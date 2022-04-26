"""
Impelementation of interaction interfaces. For the connection between policies and envirnoments.

@status: passed
@author: Ming Zhou
@organization: SJTU-MARL
@date: 2021/08/15
"""

from typing import Sequence, Union, Any, Callable, Dict, Tuple, Type
from types import LambdaType
from dataclasses import dataclass

import gym
import numpy as np
import ray

from common.logger import Log
from common.data import Batch, DataArray
from common.base_policy import Policy, Action, ActionDist, Logits
from common.sampler import SamplerInterface


DataArray = Type[np.ndarray]
AgentID = str
DEFAULT_OBSERVATION_ADAPTER = lambda x, obs_spec: x
DEFAULT_ACTION_ADAPTER = lambda x: x


class InteractInterface:
    def __init__(self) -> None:
        pass

    def compute_action(self, **kwargs) -> Any:
        raise NotImplementedError

    def transform_observation(self, **kwargs) -> Any:
        raise NotImplementedError

    def reset_behavior_state(self, **kwargs):
        raise NotImplementedError


@dataclass
class AgentInterface(InteractInterface):
    policy_name: str
    """human-readable policy name"""

    policy: Union[ray._raylet.ObjectRef, Policy]
    """policy, could be remote or local instance."""

    observation_space: gym.spaces.Space
    """observation space"""

    action_space: gym.spaces.Space
    """action space"""

    observation_adapter: Callable = lambda x, obs_spec: x
    """for some environments which require observation reconstruction."""

    action_adapter: Callable = lambda x: x
    """for some environments which require action mapping from abstraction represention from policy to
    human readable actions
    """

    is_active: bool = False
    """ is current agent interface use active policy as behavior policy or not."""

    def __post_init__(self):
        self._warn_time = 0
        self._discrete = type(self.action_space) in [
            gym.spaces.Discrete,
            gym.spaces.MultiDiscrete,
        ]

    def compute_action(
        self,
        observation: Sequence[Any],
        action_mask: Sequence[DataArray] = None,
        state: np.ndarray = None,
        evaluate: bool = True,
        rank: int = 0,
    ) -> Sequence[Any]:
        """Compute action with given a sequence of observation and action_mask. If `policy_id` is not None.
        Agent will retrieve policy tagged with `policy_id` from its policy pool.

        Note:
            For fixed policy / policy_pool (and policy pool has no learnable dist), `evaluate` will be fored as True.

        Args:
            observation (Sequence[Any]): A sequence of transformed observations.
            action_mask (Sequence[DataArray]): A sequence of action masks.
            evalute: (bool): Use evaluation mode or not.
            rank (int, optional): Indicate which policy level for action compute.

        Raises:
            TypeError: Unsupported policy type.

        Returns:
            Sequence[Any]: A sequence of actions.
        """

        obs_shape = self.policy.preprocessor.shape
        original_batch = observation.shape[: -len(obs_shape)]
        observation = observation.reshape((-1,) + obs_shape)
        if action_mask is not None:
            action_mask = action_mask.reshape((-1,) + action_mask.shape)

        evaluate = evaluate | (
            not self.is_active
        )  # True if (not self.is_active or self.policy.is_fixed) else evaluate

        if isinstance(self.policy, Policy):
            # (batch, innershape)
            batch = self.policy.compute_action(
                Batch(observation=observation, action_mask=action_mask),
                evaluate=evaluate,
                state=state,
                rank=rank,
            )
        elif isinstance(self.policy, dict):
            raise DeprecationWarning(
                "Policy dict for agent interface has been deprecated, do not use it!"
            )
        else:
            raise TypeError(f"Unexpected policy type: {type(self.policy)}")
        action, action_dist, logits, state = self.reshape_return(batch, original_batch)
        return action, action_dist, logits, state

    def reshape_return(self, batch: Batch, original_batch: Sequence[int]):
        # FIXME(ming): we need to return action_dist here.
        if isinstance(batch.action, np.ndarray):
            if len(batch.action.shape) > 1:
                a_shape = batch.action.shape[1:]
            else:
                a_shape = ()
        else:
            a_shape = ()

        if len(batch.logits.shape) > 1:
            start = 1
        else:
            start = 0

        logits_shape = batch.logits.shape[start:]
        dist_shape = batch.action_dist.shape[start:]

        if batch.state is not None:
            state_shape = batch.state.shape[start:]
            reshaped_state = batch.state.reshape(original_batch + state_shape)
        else:
            reshaped_state = None

        return (
            batch.action.reshape(original_batch + a_shape)
            if isinstance(batch.action, np.ndarray)
            else batch.action,
            batch.action_dist.reshape(original_batch + dist_shape),
            batch.logits.reshape(original_batch + logits_shape),
            reshaped_state,
        )

    def action_mask(self, raw_observation) -> DataArray:
        """Generate an action mask from raw observation.

        Args:
            raw_observation ([type]): Raw environment observation.

        Returns:
            DataArray: A returned action mask.
        """

        shape = (self.action_space.n,) if self._discrete else self.action_space.shape
        if isinstance(raw_observation, dict):
            legal_actions = raw_observation.get("legal_actions")
            if legal_actions is not None:
                legal_actions = (
                    legal_actions[raw_observation["current_player"]]
                    if raw_observation.get("current_player") is not None
                    else legal_actions
                )
                action_mask = np.zeros(shape)
                # FIXME: multi dim
                action_mask[legal_actions] = 1
            else:
                action_mask = raw_observation.get("action_mask")
                if action_mask is None:
                    action_mask = np.ones(shape)
                # assert action_mask is not None, f"Cannot find action mask in raw observation: {raw_observation}!"
            return action_mask
        else:
            return np.ones(shape)

    def transform_observation(self, observation) -> DataArray:
        observation = self.observation_adapter(observation, self.observation_space)

        if self.policy.preprocessor is not None:
            return self.policy.preprocessor.transform(observation)
        else:
            if self._warn_time == 0:
                Log.warning(
                    "AgentInterface:: No callable preprocessor, will return the original observation"
                )
                self._warn_time += 1
            return observation

    def reset_behavior_state(self, **kwargs):
        self.policy.reset(is_active=self.is_active, **kwargs)


class AgentInterfaceManager(InteractInterface):
    def __init__(
        self,
        agent_interfaces: Dict[AgentID, AgentInterface],
        agent_mapping: LambdaType,
    ) -> None:
        self.agent_interfaces = agent_interfaces
        self.agent_mapping = agent_mapping
        self.sampler = None

    def register_sampler(self, sampler: SamplerInterface):
        self.sampler = sampler

    def compute_action(
        self, **kwargs
    ) -> Tuple[Dict[AgentID, Action], Dict[AgentID, ActionDist], Dict[AgentID, Logits]]:
        observation = kwargs["observation"]
        action_mask = kwargs["action_mask"]
        hidden_state = kwargs["state"]
        evaluate = kwargs["evaluate"]
        rank = kwargs["rank"]
        action_mask = action_mask or {aid: None for aid in observation}
        agent_ids = kwargs.get("agents", None) or list(observation.keys())
        action, action_dist, logits, next_hidden_state = {}, {}, {}, {}

        for aid in agent_ids:
            ppid = self.agent_mapping(aid)
            interface = self.agent_interfaces[ppid]
            (
                action[aid],
                action_dist[aid],
                logits[aid],
                next_hidden_state[aid],
            ) = interface.compute_action(
                observation=observation[aid],
                action_mask=action_mask[aid],
                state=hidden_state[aid],
                evaluate=evaluate,
                rank=rank,
            )
        return action, action_dist, logits, next_hidden_state

    def compute_action_mask(self, **kwargs):
        observation = kwargs["observation"]
        agents = kwargs.get("agents") or list(observation.keys())

        res = {}
        for aid in agents:
            ppid = self.agent_mapping(aid)
            res[aid] = self.agent_interfaces[ppid].action_mask(observation[aid])

        return res

    def add_batches(self, **kwargs):
        self.sampler.add_batches(kwargs["buffers"])

    def add_transition(self, **kwargs):
        vector_mode = kwargs["vector_mode"]
        transition = kwargs["transition"]
        self.sampler.add_transition(vector_mode=vector_mode, **transition)

    def transform_observation(self, **kwargs) -> Dict[AgentID, DataArray]:
        observation = kwargs["observation"]
        agent_ids = kwargs.get("agents", None) or list(observation.keys())
        res = {}
        for aid in agent_ids:
            ppid = self.agent_mapping(aid)
            res[aid] = self.agent_interfaces[ppid].transform_observation(
                observation[aid]
            )
        return res

    def reset_behavior_state(self, **kwargs):
        # FIXME(ming): here maybe args overlapping
        for aid, interface in self.agent_interfaces.items():
            interface.reset_behavior_state(
                evaluate=kwargs["evaluate"],
                policy_id=kwargs["policy_mapping"].get(aid, None),
            )
        return
