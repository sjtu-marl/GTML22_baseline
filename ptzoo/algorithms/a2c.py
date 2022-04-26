import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from gym import spaces

from types import Any, Dict
from algorithms.pg import PGPolicy
from common.data import Batch
from common.models.torch import continuous, discrete


DEFAULT_CONFIG = {
    "training_config": {
        "optimizer": "Adam",
        "lr": 1e-4,
        "reward_norm": None,
        "n_repeat": 2,
        "minibatch": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "max_gae_batchsize": 256,
        "value_coef": 1.0,
        "entropy_coef": 1e-3,
        "grad_norm": 5.0,
    },
    "model_config": {
        "preprocess_net": {"net_type": None, "config": {"hidden_sizes": [64]}},
        "hidden_sizes": [64],
    },
}


class A2CPolicy(PGPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        model_config: Dict[str, Any],
        custom_config: Dict[str, Any],
        is_fixed: bool = False,
        replacement: Dict[str, nn.Module] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            model_config,
            custom_config,
            is_fixed,
            replacement,
        )

        if replacement is not None:
            self.critic = replacement["critic"]
        else:
            preprocess_net: nn.Module = self.actor.preprocess
            if isinstance(action_space, spaces.Discrete):
                self.critic = discrete.Critic(
                    preprocess_net=preprocess_net,
                    hidden_sizes=model_config["hidden_sizes"],
                    device=self.device,
                )
            elif isinstance(action_space, spaces.Box):
                self.critic = continuous.Critic(
                    preprocess_net=preprocess_net,
                    hidden_sizes=model_config["hidden_sizes"],
                    device=self.device,
                )
            else:
                raise TypeError(
                    "Unexpected action space type: {}".format(type(action_space))
                )

        self.register_state(self.critic, "critic")

    def value_function(self, batch: Batch, state=None):
        return self.critic(batch.observation, state=state)

    def action_value_function(
        self, batch: Batch, evaluate: bool, state: Any
    ) -> np.ndarray:
        """Compute the state action value with given batch. Note that the batch should include `estimate_reward` item.

        Args:
            batch (Batch): The given batch
            evaluate (bool): Evaluate mode or not
            state (Any): Hidden state

        Returns:
            np.ndarray: An array like values
        """

        assert (
            "estimate_reward" in batch
        ), "Batch has no item named `estimate_reward`: {}".format(batch.keys())
        values = self.critic(batch.observation, state=state)
        return values


import itertools

from torch import optim
from torch import nn
from torch.nn import functional as F

from types import Dict
from common.data import DataLoader, EpisodeHandler
from common.tianshou_batch import Batch
from ptzoo.common.base_trainer import Trainer


class A2CTrainer(Trainer):
    def setup(self):
        parameter_dict = self.policy.parameters()
        # concate parameters
        parameters = set(itertools.chain(*parameter_dict.values()))
        self.optimizer = getattr(optim, self.training_config["optimizer"])(
            parameters, lr=self.training_config["lr"]
        )
        self.parameters = parameters
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None
        self.ret_rms = None

    def process_fn(self, batch: Batch) -> Batch:
        state_value, next_state_value = [], []
        with torch.no_grad():
            for minibatch in batch.split(
                self.training_config.get("max_gae_batchsize", 256),
                shuffle=False,
                merge_last=True,
            ):
                state_value.append(self.policy.critic(minibatch.observation))
                next_state_value.append(self.policy.critic(minibatch.next_observation))
        batch["state_value"] = torch.cat(state_value, dim=0).flatten()  # old value
        state_value = batch.state_value.cpu().numpy()
        next_state_value = torch.cat(next_state_value, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self.training_config[
            "reward_norm"
        ]:  # unnormalize state_value & next_state_value
            eps = self.training_config["reward_norm"]["config"]["eps"]
            state_value = state_value * np.sqrt(self.ret_rms.var + eps)
            next_state_value = next_state_value * np.sqrt(self.ret_rms.var + eps)

        unnormalized_returns, advantages = EpisodeHandler.compute_episodic_return(
            batch,
            state_value,
            next_state_value,
            self.training_config["gamma"],
            self.training_config["gae_lambda"],
        )

        if self.training_config["reward_norm"]:
            batch["returns"] = unnormalized_returns / np.sqrt(self.ret_rms.var + eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch["returns"] = unnormalized_returns

        # batch.returns = to_torch_as(batch.returns, batch.state_value)
        batch["advantage"] = advantages  # to_torch_as(advantages, batch.state_value)
        assert (
            batch.advantage.shape == batch.state_value.shape == batch.returns.shape
        ), (batch.advantage.shape, batch.state_value.shape, batch.returns.shape)
        batch["logits"], _ = self.policy.actor(
            batch.observation, state=batch.get("state", None)
        )
        batch.to_torch(device=self.policy.device)
        return batch

    def train(self, dataloader: DataLoader) -> Dict[str, float]:
        actor_losses, vf_losses, ent_losses, losses = [], [], [], []
        for _ in range(self.training_config["n_repeat"]):
            for batch_dict in dataloader:
                batch = Batch(**batch_dict)
                batch = self.process_fn(batch)
                # calculate loss for actor
                logits = batch.logits
                dist = self.policy.dist_fn.proba_distribution(logits)
                log_prob = dist.log_prob(batch.action)
                log_prob = log_prob.reshape(len(batch.advantage), -1).transpose(0, 1)
                actor_loss = -(log_prob * batch.advantage).mean()
                # calculate loss for critic
                value = self.policy.critic(batch.observation).flatten()
                vf_loss = F.mse_loss(batch.returns, value)
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = (
                    actor_loss
                    + self.training_config["value_coef"] * vf_loss
                    - self.training_config["entropy_coef"] * ent_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.training_config.get("grad_norm", 0):  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self.parameters, max_norm=self.training_config["grad_norm"]
                    )
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        avg_actor_loss = sum(actor_losses) / max(1, len(actor_losses))
        avg_vf_loss = sum(vf_losses) / max(1, len(vf_losses))
        avg_ent_loss = sum(ent_losses) / max(1, len(ent_losses))
        avg_total_loss = sum(losses) / max(1, len(losses))
        return {
            "avg_actor_loss": avg_actor_loss,
            "avg_vf_loss": avg_vf_loss,
            "avg_ent_loss": avg_ent_loss,
            "avg_total_loss": avg_total_loss,
        }
