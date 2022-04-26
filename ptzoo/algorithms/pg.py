import numpy as np
import torch

from gym import spaces
from torch import nn

from typing import Any, Tuple, Optional, Union, Dict
from common.tianshou_batch import Batch
from common import misc

from common.models.torch import net, discrete, continuous
from common.base_policy import (
    Policy,
    Action,
    ActionDist,
    Logits,
)


DEFAULT_CONFIG = {
    "training_config": {
        "optimizer": "Adam",
        "lr": 1e-4,
        "reward_norm": None,
        "n_repeat": 2,
        "minibatch": 2,
        "gamma": 0.99,
        "batch_size": 64,
    },
    "model_config": {
        "preprocess_net": {"net_type": None, "config": {"hidden_sizes": [64]}},
        "hidden_sizes": [64],
    },
    "custom_config": {},
}


class PGPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        model_config: Dict[str, Any],
        custom_config: Dict[str, Any],
        is_fixed: bool = False,
        replacement: Dict[str, nn.Module] = None,
    ):
        """Build a REINFORCE policy whose input and output dims are determined by observation_space and action_space, respectively.

        Args:
            observation_space (spaces.Space): The observation space.
            action_space (spaces.Space): The action space.
            model_config (Dict[str, Any]): The model configuration dict.
            custom_config (Dict[str, Any]): The custom configuration dict.
            is_fixed (bool, optional): Indicates fixed policy or trainable policy. Defaults to False.

        Raises:
            NotImplementedError: Does not support other action space type settings except Box and Discrete.
            TypeError: Unexpected action space.
        """

        super().__init__(
            observation_space, action_space, model_config, custom_config, is_fixed
        )

        # update model preprocess_net config here
        action_shape = (
            (action_space.n,) if len(action_space.shape) == 0 else action_space.shape
        )

        if replacement is not None:
            self.actor = replacement["actor"]
        else:
            preprocess_net: nn.Module = net.make_net(
                observation_space,
                self.device,
                model_config["preprocess_net"].get("net_type", None),
                **model_config["preprocess_net"]["config"]
            )
            if isinstance(action_space, spaces.Discrete):
                self.actor = discrete.Actor(
                    preprocess_net=preprocess_net,
                    action_shape=action_shape,
                    hidden_sizes=model_config["hidden_sizes"],
                    softmax_output=False,
                    device=self.device,
                )
            elif isinstance(action_space, spaces.Box):
                self.actor = continuous.Actor(
                    preprocess_net=preprocess_net,
                    action_shape=action_shape,
                    hidden_sizes=model_config["hidden_sizes"],
                    max_action=custom_config.get("max_action", 1.0),
                    device=self.device,
                )
            else:
                raise TypeError(
                    "Unexpected action space type: {}".format(type(action_space))
                )

        self.register_state(self.actor, "actor")

    def value_function(self, state: Any, action_mask: Any = None, **kwargs):
        """Compute value. Since PG no critic, then return 0.

        Args:
            state (Any): State
            action_mask (Any, optional): Action mask. Defaults to None.
        """

        # return a real return
        return 0.0

    def action_value_function(
        self, batch: Batch, evaluate: bool, state: Any
    ) -> np.ndarray:
        raise NotImplementedError("PG Policy does not support value estimation!")

    def compute_action(
        self,
        batch: Batch,
        evaluate: bool,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs
    ) -> Batch:
        expand_dim = kwargs.get("expand_dim", False)
        with torch.no_grad():
            logits, hidden = self.actor(
                np.expand_dims(batch.observation, axis=0)
                if expand_dim
                else batch.observation,
                state=state,
            )
            if isinstance(logits, tuple):
                dist = self.dist_fn.proba_distribution(*logits)
            else:
                dist = self.dist_fn.proba_distribution(
                    logits, action_mask=batch.get("action_mask", None)
                )
            if evaluate:
                if self.action_type == "discrete":
                    act = misc.masked_logits(
                        logits, mask=batch.get("action_mask", None)
                    ).argmax(-1)
                elif self.action_type == "continuous":
                    act = logits[0]
            else:
                act = dist.sample()
            probs = dist.prob().cpu().numpy()
        return Batch(
            logits=logits.cpu().numpy(),
            action=act.cpu().numpy(),
            action_dist=probs,
            state=hidden,
        )

    def compute_actions(self, **kwargs) -> Tuple[Action, ActionDist, Logits]:
        return super().compute_actions(**kwargs)


from typing import Type


from torch import optim

from common.data import DataLoader, Batch, EpisodeHandler
from common.base_trainer import Trainer


AgentID = str


class PGTrainer(Trainer):
    def setup(self):
        self.optimizer: Type[optim.Optimizer] = getattr(
            optim, self.training_config["optimizer"]
        )(self.policy.parameters()["actor"], lr=self.training_config["lr"])
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None
        self.ret_rms = None

    def process_fn(self, batch: Batch) -> Batch:

        # v_s_ = np.full(indices.shape, self.ret_rms.mean)
        unnormalized_returns, _ = EpisodeHandler.compute_episodic_return(
            batch, gamma=self.training_config["gamma"], gae_lambda=1.0
        )

        if self.training_config["reward_norm"] and self.ret_rms is not None:
            batch["returns"] = (unnormalized_returns - self.ret_rms.mean) / np.sqrt(
                self.ret_rms.var + self._eps
            )
            self.ret_rms.update(unnormalized_returns)
        else:
            batch["returns"] = unnormalized_returns
        batch["logits"], _ = self.policy.actor(
            batch.observation, state=batch.get("state", None)
        )
        batch.to_torch(device=self.policy.device)
        return batch

    def concat_multiagent(self, agent_batch: Dict[AgentID, Dict]) -> Batch:
        res = []
        for k, v in agent_batch.items():
            res.append(Batch(**v))
        batch = Batch(res)
        return batch

    def train(self, dataloader: DataLoader) -> Dict[str, Any]:
        losses = []
        for _ in range(self.training_config["n_repeat"]):
            for batch_dict in dataloader:
                batch = Batch(**batch_dict)
                batch = self.process_fn(batch)
                self.optimizer.zero_grad()
                logits = batch.logits
                dist = self.policy.dist_fn.proba_distribution(logits)
                act = batch.action
                ret = batch.returns
                log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
                loss = -(log_prob * ret).mean()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return {"avg_loss": sum(losses) / max(1, len(losses))}
