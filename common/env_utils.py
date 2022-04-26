"""
ABC environment class and the implementation of environrment vectorization.

@status: passed
@author: Ming Zhou
@organization: SJTU-MARL
@date: 2021/08/15
"""

import gym

from typing import Dict, List, Any, Type, Union, Tuple

from common.data import DataArray


AgentID = str


class Environment:
    def __init__(self, **configs):
        self.is_sequential = False
        self.env_id = configs["env_id"]
        self.scenario_config = configs["scenario_config"]
        self._extra_returns = []
        self._trainable_agents = None
        self._configs = configs
        self._env = None

    @property
    def env(self):
        return self._env

    def get_trainable_agents(self):
        return self.trainable_agents

    def get_extra_returns(self):
        return self.extra_returns

    @staticmethod
    def from_sequential_game(env, **kwargs):
        raise NotImplementedError

    @property
    def possible_agents(self):
        return self._env.possible_agents

    @property
    def trainable_agents(self) -> Union[Tuple, None]:
        """Return trainble agents, if registered return a tuple, otherwise None"""
        return self._trainable_agents

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._env.observation_spaces

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._env.action_spaces

    @property
    def extra_returns(self):
        return self._extra_returns

    def agent_to_group(self, agent_id) -> str:
        return agent_id

    def reset(self, *args, **kwargs):
        return self._env.reset()

    def step(self, actions: Dict[AgentID, Any]):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def get_episode_info(self) -> Dict[str, Any]:
        """Return a dict of episode statistic information.

        Returns:
            Dict[str, Any]: A dict of episode statistic information
        """

        raise NotImplementedError

    def close(self):
        self._env.close()

    def seed(self, seed: int = None):
        pass


"""
The `VectorEnv` is an interface that integrates multiple environment instances to support parllel rollout 
with shared multi-agent policies. Currently, the `VectorEnv` support parallel rollout for environment which steps in simultaneous mode.
"""

import logging
import gym
import numpy as np
import copy
import ray

from collections import defaultdict

from typing import Dict, Any, List, Tuple
from common.logger import Log
from common.env_utils import Environment


logger = logging.getLogger(__name__)
DataArray = Type[np.ndarray]
AgentID = str


class AgentItems:
    def __init__(self) -> None:
        self._data = {}
        self._cache = None

    def update(self, agent_items: Dict[AgentID, Any]):
        self._cache = None
        for k, v in agent_items.items():
            if k not in self._data:
                self._data[k] = []
            self._data[k].append(v)

    def cleaned_data(self):
        if self._cache is None:
            self._cache = {k: np.stack(v) for k, v in self._data.items()}
        return self._cache


class VectorEnv:
    def __init__(
        self,
        env_desc,
        num_envs: int = 0,
        use_remote: bool = True,
        resource_config: Dict = None,
    ):
        """Create a vector environment.

        Args:
            observation_spaces (Dict[AgentID, gym.Space]): A dict of agent observation spaces.
            action_spaces (Dict[AgentID, gym.Space]): A dict of agent action spaces.
            creator (type): The handler to create environment.
            configs (Dict[str, Any]): Environment configuration
            num_envs (int, optional): The number of nested environments. Defaults to 0.
        """

        env_config = env_desc["config"]

        self.observation_spaces = env_config["observation_spaces"]
        self.action_spaces = env_config["action_spaces"]
        self.possible_agents = env_config["possible_agents"]

        self._num_envs = num_envs
        self._creator = env_desc["creator"]
        self._configs = copy.deepcopy(env_config)
        self._envs: List[Environment] = []
        self._group = env_config.get("group", None)
        self._agent_to_group = {}
        self._use_remote = use_remote if num_envs > 1 else False

        if use_remote:
            assert (
                ray.is_initialized()
            ), "Remote environment is trigger, but ray hasn't been initialized."
            resources = ray.available_resources()
            Log.info("Ray available resources: %s", resources)
            if resource_config is not None:
                self._creator = ray.remote(**resource_config)(self._creator).remote
            else:
                self._creator = ray.remote(
                    # num_cpus=2,
                    num_cpus=None,
                    num_gpus=None,
                    memory=None,
                    object_store_memory=None,
                    resources=None,
                )(self._creator).remote

        if self._group is not None:
            for gk, _agents in self._group.items():
                self._agent_to_group.update(dict.fromkeys(_agents, gk))
        else:
            self._agent_to_group = {aid: aid for aid in self.possible_agents}

        if num_envs > 0:
            self._envs.append(self._creator(**self._configs))
            self._envs.extend(
                [self._creator(**self._configs) for _ in range(num_envs - 1)]
            )

        self._limits = len(self._envs)

    @property
    def trainable_agents(self):
        if self._use_remote:
            return ray.get(self._envs[0].get_trainable_agents.remote())
        return self._envs[0].trainable_agents

    @property
    def num_envs(self) -> int:
        """The total number of environments"""

        return self._num_envs

    @property
    def envs(self) -> List[Environment]:
        """Return a limited list of enviroments"""

        return self._envs[: self._limits]

    @property
    def extra_returns(self) -> List[str]:
        """Return extra columns required by this environment"""
        if self._use_remote:
            return ray.get(self.envs[0].get_extra_returns.remote())
        else:
            return self.envs[0].extra_returns

    @property
    def env_creator(self):
        return self._creator

    @property
    def env_configs(self):
        return self._configs

    @property
    def limits(self):
        return self._limits

    @classmethod
    def from_envs(cls, envs: List, config: Dict[str, Any]):
        """Generate vectorization environment from exisiting environments."""

        observation_spaces = envs[0].observation_spaces
        action_spaces = envs[0].action_spaces

        vec_env = cls(observation_spaces, action_spaces, type(envs[0]), config, 0)
        vec_env.add_envs(envs=envs)

        return vec_env

    def add_envs(self, envs: List = None, num: int = 0):
        """Add exisiting `envs` or `num` new environments to this vectorization environment.
        If `envs` is not empty or None, the `num` will be ignored.
        """

        from gym.spaces import Discrete

        if envs and len(envs) > 0:
            for env in envs:
                self._validate_env(env)
                self._envs.append(env)
                self._num_envs += 1
            logger.debug(f"added {len(envs)} exisiting environments.")
        elif num > 0:
            for _ in range(num):
                self._envs.append(self.env_creator(**self.env_configs))
                self._num_envs += 1
            logger.debug(f"created {num} new environments.")

    def reset(self, limits: int = None) -> Dict:
        # if limits is not None:
        #     assert limits > 0, limits
        #     self._limits = limits
        transitions = defaultdict(lambda: AgentItems())
        if self._use_remote:
            rets = ray.get([env.reset.remote() for env in self.envs])
            for ret in rets:
                for k, agent_items in ret.items():
                    transitions[k].update(agent_items)
        else:
            for i, env in enumerate(self.envs):
                ret = env.reset()
                for k, agent_items in ret.items():
                    transitions[k].update(agent_items)
        data = {k: v.cleaned_data() for k, v in transitions.items()}
        return data

    def step(self, actions: Dict[AgentID, List]) -> Dict[str, DataArray]:
        transitions = defaultdict(lambda: AgentItems())
        rets = []
        for i, env in enumerate(self.envs):
            if self._use_remote:
                rets.append(
                    env.step.remote(
                        {
                            _agent: array[i] if len(array.shape) > 0 else array
                            for _agent, array in actions.items()
                        }
                    )
                )
            else:
                rets.append(
                    env.step(
                        {
                            _agent: array[i] if len(array.shape) > 0 else array
                            for _agent, array in actions.items()
                        }
                    )
                )
        if self._use_remote:
            rets = ray.get(rets)
        for ret in rets:
            for k, agent_items in ret.items():
                transitions[k].update(agent_items)

        # merge transitions by keys
        data = {k: v.cleaned_data() for k, v in transitions.items()}
        return data

    def agent_to_group(self, agent_id: AgentID) -> str:
        return self._agent_to_group[agent_id]

    def close(self):
        if self._use_remote:
            ray.get([env.close.remote() for env in self.envs])
        else:
            for env in self._envs:
                env.close()

    def seed(self, seed):
        if self._use_remote:
            ray.get([env.seed.remote(seed) for env in self.envs])
        else:
            for env in self._envs:
                env.seed(seed)
