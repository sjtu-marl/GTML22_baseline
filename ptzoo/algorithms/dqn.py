from typing import AgentID, DataArray, Any, Tuple, Dict, Type

import os
import copy

import gym
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from common.logger import Log
from common.data import Batch
from common.base_policy import Policy
from common.models.torch.net import make_net

from common.data import Batch, DataLoader
from common.schedules import LinearSchedule
from common import misc
from common.base_trainer import Trainer

DataArray = Type[np.ndarray]


DEFAULT_CONFIG = {
    "training_config": {
        "optimizer": "Adam",
        "lr": 1e-4,
        "reward_norm": None,
        "n_repeat": 2,
        "minibatch": 2,
        "gamma": 0.99,
        "update_interval": 1,
        "tau": 0.05,
        "batch_size": 64,
    },
    "model_config": {
        "net_type": "general_net",
        "config": {"hidden_sizes": [256, 256, 256, 64]},
    },
    "custom_config": {"schedule_timesteps": 10000, "final_p": 0.05, "initial_p": 1.0},
}


class DQNPolicy(Policy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any],
        custom_config: Dict[str, Any],
        is_fixed: bool = False,
        replacement: Dict = None,
    ):
        super(DQNPolicy, self).__init__(
            observation_space,
            action_space,
            model_config,
            custom_config,
            is_fixed,
            replacement,
        )

        assert isinstance(action_space, gym.spaces.Discrete)

        if replacement is not None:
            self.critic = replacement["critic"]
        else:
            self.critic: nn.Module = make_net(
                observation_space=observation_space,
                action_space=action_space,
                device=self.device,
                net_type=model_config.get("net_type", None),
                **model_config["config"]
            )

        self.use_cuda = self.custom_config.get("use_cuda", False)

        if self.use_cuda:
            self.critic = self.critic.to("cuda")

        self._eps = 1.0

        self.register_state(self._eps, "_eps")
        self.register_state(self.critic, "critic")

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, value: float):
        self._eps = value

    def action_value_function(self, batch: Batch, evaluate: bool, state: Any = None):
        with torch.no_grad():
            logits, state = self.critic(batch.observation, state)
            logits = misc.masked_logits(
                logits, mask=batch.get("action_mask"), explore=not evaluate
            )
        return logits.cpu().numpy(), state

    def compute_action(self, batch: Batch, evaluate: bool, state: Any = None):
        """Compute action in rollout stage. Do not support vector mode yet.

        Args:
            observation (DataArray): The observation batched data with shape=(n_batch, *obs_shape).
            action_mask (DataArray): The action mask batched with shape=(n_batch, *mask_shape).
            evaluate (bool): Turn off exploration or not.
            state (Any, Optional): The hidden state. Default by None.
        """

        observation = batch.observation
        action_mask = batch.get("action_mask", None)
        observation = torch.as_tensor(
            observation, device="cuda" if self.use_cuda else "cpu"
        )

        with torch.no_grad():
            logits, state = self.critic(observation)

            # do masking
            if action_mask is not None:
                mask = torch.FloatTensor(action_mask).to(logits.device)
                action_probs = misc.masked_gumbel_softmax(logits, mask)
                assert mask.shape == logits.shape, (mask.shape, logits.shape)
            else:
                action_probs = misc.gumbel_softmax(logits, hard=True)

        if not evaluate:
            if np.random.random() < self.eps:
                action_probs = (
                    np.ones((len(observation), self._action_space.n))
                    / self._action_space.n
                )
                if action_mask is not None:
                    legal_actions = np.array(
                        [
                            idx
                            for idx in range(self._action_space.n)
                            if action_mask[0][idx] > 0
                        ],
                        dtype=np.int32,
                    )
                    action = np.random.choice(legal_actions, len(observation))
                else:
                    action = np.random.choice(self._action_space.n, len(observation))
                return Batch(
                    action=action,
                    action_dist=action_probs,
                    logits=logits.cpu().numpy(),
                    state=state,
                )

        action = torch.argmax(action_probs, dim=-1).cpu().numpy()
        return Batch(
            action=action,
            action_dist=action_probs.cpu().numpy(),
            logits=logits.cpu().numpy(),
            state=state,
        )

    def parameters(self):
        return {
            "critic": self._critic.parameters(),
            # "target_critic": self._target_critic.parameters(),
        }

    def compute_actions(self, **kwargs):
        raise NotImplementedError

    def value_function(self, states, action_mask=None) -> np.ndarray:
        states = torch.as_tensor(states, device="cuda" if self.use_cuda else "cpu")
        values = self.critic(states).detach().cpu().numpy()
        if action_mask is not None:
            values[action_mask] = -1e9
        return values

    def reset(self, **kwargs):
        pass

    def save(self, path, global_step=0, hard: bool = False):
        file_exist = os.path.exists(path)
        if file_exist:
            Log.warning("\t! detected existing mode with path: {}".format(path))
        if (not file_exist) or hard:
            torch.save(self._critic.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location="cuda" if self.use_cuda else "cpu")
        self._critic.load_state_dict(state_dict)


class DQNTrainer(Trainer):
    def setup(self):
        exploration_fraction = self._training_config["exploration_fraction"]
        total_timesteps = self._training_config["total_timesteps"]
        exploration_final_eps = self._training_config["exploration_final_eps"]
        self.fixed_eps = self._training_config.get("pretrain_eps")
        self.pretrain_mode = False

        self.exploration = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * total_timesteps),
            initial_p=1.0 if self.fixed_eps is None else self.fixed_eps,
            final_p=exploration_final_eps,
        )

        optim_cls = getattr(torch.optim, self.training_config["optimizer"])
        self.target_critic = copy.deepcopy(self.policy.critic)
        self.optimizer: torch.optim.Optimizer = optim_cls(
            self.policy.critic.parameters(), lr=self.training_config["critic_lr"]
        )

    def set_pretrain(self, pmode=True):
        self.pretrain_mode = pmode

    def process_fn(
        self, batch: Dict[AgentID, Dict[str, DataArray]]
    ) -> Dict[AgentID, Dict[str, DataArray]]:
        policy = self.policy.to(
            "cuda" if self.policy.custom_config.get("use_cuda", False) else "cpu",
            use_copy=False,
        )
        # set exploration rate for policy
        if not self._training_config.get("param_noise", False):
            update_eps = self.exploration.value(self.counter)
            update_param_noise_threshold = 0.0
        else:
            update_eps = 0.0
        if self.pretrain_mode and self.fixed_eps is not None:
            policy.eps = self.fixed_eps
        else:
            policy.eps = update_eps

        return list(batch.values())[0]

    def train(self, batch):
        batch = Batch(**self.process_fn(batch))
        batch.to_torch(dtype=torch.float32, device=self.policy.device)

        state_action_values, _ = self.policy.critic(batch.observation)
        state_action_values = state_action_values.gather(
            -1, batch.action.long().view((-1, 1))
        ).view(-1)

        next_state_q, _ = self.target_critic(batch.next_observation)
        next_action_mask = batch.get("next_action_mask", None)

        if next_action_mask is not None:
            illegal_action_mask = 1.0 - next_action_mask
            # give very low value to illegal action logits
            illegal_action_logits = -illegal_action_mask * 1e9
            next_state_q += illegal_action_logits

        next_state_action_values = next_state_q.max(-1)[0]
        expected_state_values = (
            batch.reward
            + self._training_config["gamma"]
            * (1.0 - batch.done)
            * next_state_action_values
        )

        self.optimizer.zero_grad()
        loss = F.mse_loss(state_action_values, expected_state_values.detach())
        loss.backward()
        self.optimizer.step()

        misc.soft_update(
            self.target_critic, self.policy.critic, tau=self._training_config["tau"]
        )

        return {
            "loss": loss.detach().item(),
            "mean_target": expected_state_values.mean().cpu().item(),
            "mean_eval": state_action_values.mean().cpu().item(),
            "min_eval": state_action_values.min().cpu().item(),
            "max_eval": state_action_values.max().cpu().item(),
            "max_target": expected_state_values.max().cpu().item(),
            "min_target": expected_state_values.min().cpu().item(),
            "mean_reward": batch.reward.mean().cpu().item(),
            "min_reward": batch.reward.min().cpu().item(),
            "max_reward": batch.reward.max().cpu().item(),
            "eps": self.policy.eps,
        }
