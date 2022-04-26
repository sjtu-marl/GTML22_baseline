import importlib
import gym
import numpy as np

from typing import Dict
from pettingzoo.utils.wrappers import BaseWrapper

from common.data import EpisodeKeys


# class ClipWrapper(BaseWrapper):
#     def __init__(self, max_cycles: int, env: object):
#         super().__init__(env)

#         self.old_observation_spaces = env.observation_spaces
#         self.max_cycles = max_cycles
#         # self.observation_spaces =
#         self.time_step = 0

#     @property
#     def observation_spaces(self):
#         return {
#             aid: gym.spaces.Dict(
#                 {
#                     "time": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
#                     "opponent_act_emb": obs_space,
#                 }
#             )
#             for aid, obs_space in self.old_observation_spaces.items()
#         }

#     def observe(self, agent):
#         obs = super().observe(agent)
#         obs = {"time": self.time_step / self.max_cycles, "opponent_act_emb": obs}
#         return obs

#     def step(self, action):
#         self.time_step += 1
#         return super().step(action)

#     def reset(self):
#         self.time_step = 0
#         super().reset()


def creator(**kwargs):
    env_id = kwargs["env_id"]
    scenario_config = kwargs.get("scenario_config", {})
    env_module = importlib.import_module(f"pettingzoo.{env_id}")
    env = env_module.env(**scenario_config)
    # env = ClipWrapper(max_cycles=scenario_config.get("max_cycles", 100), env=env)
    return env


def env_desc_gen(env_id: str, scenario_config: Dict):
    env = creator(env_id=env_id, scenario_config=scenario_config)

    res = {
        "creator": creator,
        "config": {
            "env_id": env_id,
            "possible_agents": env.possible_agents,
            "action_spaces": env.action_spaces,
            "observation_spaces": env.observation_spaces,
            "scenario_config": scenario_config,
        },
    }

    env.close()
    return res
