"""
Rollout functions (sequential/simultaneous).

@status: passed
@author: Ming Zhou
@organization: SJTU-MARL
@date: 2021/08/15
"""

from typing import Dict, Tuple, Any, Sequence, List, Type
from collections import defaultdict

import time
import collections

import ray
import numpy as np

from pettingzoo import AECEnv

from common.logger import Log, monitor
from common.data import Episode, EpisodeKeys
from common.agent_interface import InteractInterface
from common.env_utils import VectorEnv


PolicyID = str
AgentID = str


# standard time step description, for sequential rollout
_TimeStep = collections.namedtuple(
    "_TimeStep", "observation, action_mask, reward, action, done, logits"
)


def general_call(caller, remote_interact: bool, **kwargs):
    if remote_interact:
        return ray.get(caller.remote(**kwargs))
    else:
        return caller(**kwargs)


def is_episode_termiante(
    evaluate: bool,
    done: bool,
    cnt: int,
    total_cnt: int,
    fragment_length: int,
    max_step: int,
) -> bool:
    """Judge whether current episode should be terminate. For evaluation mode, we determine termination as \
        `done or (cnt >= max_step)`, for other cases, we determine termination as \
            `done or (cnt >= max_step) or (total_cnt >= fragment_lengt)`.

    Args:
        evaluate (bool): Evaluation mode or not
        done (bool): Environment done signal
        cnt (int): Current frame number
        total_cnt (int): Current total frame number
        fragment_length (int): The maximum of frame number
        max_step (int): The maximum episode length

    Returns:
        bool: Terminate or not
    """

    if evaluate:
        return done or (cnt >= max_step)
    else:
        return done or (cnt >= max_step) or (total_cnt >= fragment_length)


def is_global_terminate(
    evaluate: bool,
    episode_num: int,
    total_cnt: int,
    fragment_length: int,
    max_episode: int,
) -> bool:
    """Judge whether current rollout procedure should be terminated. For evaluation mode, we determine termination as \
        `episode_num >= max_episode`, for other cases, we determine termination as `(total_cnt >= fragment_length) or \
            (episode_num >= max_episode)`.

    Args:
        evaluate (bool): Evaluation mode or not.
        episode_num (int): Current episode number
        total_cnt (int): Current total frame number
        fragment_length (int): The maximum of frame number
        max_episode (int): The maximum number of collected episode

    Returns:
        bool: Terminate or not
    """

    if evaluate:
        return episode_num >= max_episode
    else:
        return (total_cnt >= fragment_length) or (episode_num >= max_episode)


def simultaneous_rollout(
    agent_interface: InteractInterface,
    env_description: Dict[str, Any],
    rollout_config: Dict,
    agent_policy_mapping: Dict[AgentID, PolicyID] = None,
    evaluate: bool = False,
    seed: int = None,
    env: type = None,
    remote_interact: bool = False,
    ones_require_buffer: List = None,
    rank: int = 0,
) -> Dict[str, Any]:
    """Run simultanenous rollout. This rollout procedure supports only for games/environments that \
        executes in simultaneous manner.

    Args:
        agent_interface (InteractInterface): Any instance derived from `InteractInterface`
        env_description (Dict[str, Any]): The environment description
        rollout_config (Dict): A dict which describes the runtime rollout configuration
        agent_policy_mapping (Dict[AgentID, PolicyID], optional): If each agent has more than one policy, this attribute should \
            be specified. Defaults to None.
        evaluate (bool, optional): Indicate evaluation mode or not. Defaults to False.
        seed (int, optional): Random seed for environment initialization. Defaults to None.
        env (type, optional): Environment instance, could be derived from `Environment` or `VectorEnv`, if it has been specified \
            this procedure will run with this environment instance, otherwise construct a new one from the given \
                `env_description`. Defaults to None.
        remote_interact (bool, optional): Indicate the given `agent_interface` is a ray object or not. Defaults to False.
        ones_require_buffer (List, optional): Indicate which runtime agent wanna record buffer for training, `None` means all \
            of them require. Defaults to None.
        rank (int, optional): If there are more than one active policy for each agent to do training, we need to determined \
            which rank you wanna, 0 means the lowest rank, and the larger the higher. Defaults to 0.

    Returns:
        Dict[str, Any]: A dict of statistic information

    Yields:
        Dict[str, Any]: Some internal running information if training or evaluation required
    """

    groups = env_description["config"].get("group", None)
    fragment_length = rollout_config["fragment_length"]
    max_step = rollout_config["max_step"]
    train_every = rollout_config["train_every"]
    max_episode = rollout_config["max_episode"] if evaluate else float("inf")
    truncated = rollout_config.get("truncated", False) if not evaluate else False
    episodic = True if train_every < 0 else False

    env = env or env_description["creator"](**env_description["config"])
    env.seed(seed)

    mean_episode_reward = defaultdict(list)
    mean_episode_len = []
    episode_num = 0
    total_cnt = 0
    n_increament = 0

    agent_policy_mapping = agent_policy_mapping or {}
    ones_require_buffer = ones_require_buffer or []

    if not evaluate:
        agents_monitored = ones_require_buffer
    else:
        agents_monitored = (
            env.possible_agents if groups is None else list(groups.keys())
        )

    start = time.time()
    while not is_global_terminate(
        evaluate, episode_num, total_cnt, fragment_length, max_episode
    ):
        rets = (
            env.reset(limits=max_episode) if isinstance(env, VectorEnv) else env.reset()
        )

        general_call(
            agent_interface.reset_behavior_state,
            remote_interact,
            evaluate=evaluate,
            policy_mapping=agent_policy_mapping,
        )

        done = False
        cnt = 0

        observations = general_call(
            agent_interface.transform_observation,
            remote_interact,
            observation=rets[EpisodeKeys.OBSERVATION.value],
        )

        episode_reward = defaultdict(lambda: 0.0)
        is_episode_done = False
        hidden_states = dict.fromkeys(env.possible_agents, None)

        while not is_episode_done:
            action_masks = {}  # grouped obs does not support action masks
            if groups is not None:
                # group observations
                grouped_obs = {}
                for k, _agents in groups.items():
                    grouped_obs.update(
                        dict.fromkeys(
                            _agents, np.stack([observations[_k] for _k in _agents])
                        )
                    )
                actions, logits, hidden_states = general_call(
                    agent_interface.compute_action,
                    remote_interact,
                    observation=grouped_obs,
                    action_mask=None,
                    evaluate=evaluate,
                    rank=rank,
                )
            else:
                for aid, observation in observations.items():
                    action_masks[aid] = (
                        rets[EpisodeKeys.ACTION_MASK.value][aid]
                        if rets.get(EpisodeKeys.ACTION_MASK.value) is not None
                        else None
                    )
                actions, action_dist, logits, hidden_states = general_call(
                    agent_interface.compute_action,
                    remote_interact,
                    observation=observations,
                    action_mask=action_masks,
                    state=hidden_states,
                    evaluate=evaluate,
                    rank=rank,
                )
            rets = env.step(actions)
            # ============== handle next_frame ================
            next_observations = general_call(
                agent_interface.transform_observation,
                remote_interact,
                observation=rets[EpisodeKeys.OBSERVATION.value],
            )
            for aid, r in rets[EpisodeKeys.REWARD.value].items():
                if isinstance(r, np.ndarray):
                    r = np.mean(r)
                episode_reward[aid] += r

            # record time step if
            done = (
                any(list(rets[EpisodeKeys.DONE.value].values())[0])
                if isinstance(env, VectorEnv)
                else any(list(rets[EpisodeKeys.DONE.value].values()))
            )
            total_cnt += env.limits if isinstance(env, VectorEnv) else 1
            cnt += 1
            n_increament += env.limits if isinstance(env, VectorEnv) else 1
            is_episode_done = is_episode_termiante(
                evaluate, done, cnt, total_cnt, fragment_length, max_step
            )

            if not evaluate:
                transition = {
                    "observation": observations,
                    "reward": rets[EpisodeKeys.REWARD.value],
                    "action": actions,
                    "done": rets[EpisodeKeys.DONE.value]
                    if not truncated
                    else dict.fromkeys(
                        rets[EpisodeKeys.DONE.value].keys(),
                        [is_episode_done] * env.limits,
                    ),
                    "next_observation": next_observations,
                    "logits": logits,
                }
                if list(action_masks.values())[0] is not None:
                    transition["action_mask"] = action_masks
                general_call(
                    agent_interface.add_transition,
                    remote_interact,
                    vector_mode=isinstance(env, VectorEnv) and env.limits > 1,
                    transition=transition,
                    rank=rank,
                )
            observations = next_observations

            if not evaluate and not episodic and not remote_interact:
                if n_increament % train_every == 0:
                    # reset n increament
                    n_increament = 0
                    yield {"timesteps": total_cnt, "step": cnt}

            if cnt % 100 == 0:
                cur_time = time.time()
                fps = total_cnt / (cur_time - start)
                Log.debug(
                    "FPS: {:.3} TOTAL_CNT: {} MAX_STEP: {} FRAGMENT_LENGTH: {}".format(
                        fps,
                        total_cnt,
                        max_step,
                        fragment_length,
                    )
                )

        episode_num += env.limits if isinstance(env, VectorEnv) else 1

        for aid, v in episode_reward.items():
            mean_episode_reward[aid].append(v)

        mean_episode_len.append(cnt)

    mean_episode_reward = {
        aid: sum(v) / len(v)
        for aid, v in mean_episode_reward.items()
        if aid in agents_monitored
    }
    mean_episode_len = sum(mean_episode_len) / len(mean_episode_len)

    res = {"total_timesteps": total_cnt, "num_episode": episode_num}
    if evaluate:
        res.update(
            {
                "reward": mean_episode_reward,
                "episode_len": mean_episode_len,
            }
        )
    return res


AECEnvType = Type[AECEnv]


def sequential_rollout(
    agent_interface: InteractInterface,
    env_description: Dict[str, Any],
    rollout_config: Dict[str, Any],
    agent_policy_mapping: Dict[AgentID, PolicyID] = {},
    evaluate: bool = False,
    seed: int = None,
    env: AECEnvType = None,
    remote_interact: bool = False,
    ones_require_buffer: List = None,
    rank: int = 0,
):
    """Run sequential rollout. This rollout procedure supports only for games/environments that \
        executes in sequential manner, like Poker Game, Chess Game and etc.

    Args:
        agent_interface (InteractInterface): Any instance derived from `InteractInterface`
        env_description (Dict[str, Any]): The environment description
        rollout_config (Dict): A dict which describes the runtime rollout configuration
        agent_policy_mapping (Dict[AgentID, PolicyID], optional): If each agent has more than one policy, this attribute should \
            be specified. Defaults to None.
        evaluate (bool, optional): Indicate evaluation mode or not. Defaults to False.
        seed (int, optional): Random seed for environment initialization. Defaults to None.
        env (AECEnvType, optional): AECEnvironment instance, could be only derived from `AECEnvironment`, if it has been specified \
            this procedure will run with this environment instance, otherwise construct a new one from the given \
                `env_description`. Defaults to None.
        remote_interact (bool, optional): Indicate the given `agent_interface` is a ray object or not. Defaults to False.
        ones_require_buffer (List, optional): Indicate which runtime agent wanna record buffer for training, `None` means all \
            of them require. Defaults to None.
        rank (int, optional): If there are more than one active policy for each agent to do training, we need to determined \
            which rank you wanna, 0 means the lowest rank, and the larger the higher. Defaults to 0.

    Returns:
        Dict[str, Any]: A dict of statistic information

    Yields:
        Dict[str, Any]: Some internal running information if training or evaluation required
    """

    groups = env_description["config"].get("group", None)
    fragment_length = rollout_config["fragment_length"]
    max_step = rollout_config["max_step"]
    train_every = rollout_config["train_every"]
    max_episode = rollout_config["max_episode"] if evaluate else float("inf")
    episodic = True if train_every < 0 else False

    Log.debug("Creating game %s", env_description["config"]["env_id"])
    env = env or env_description["creator"](**env_description["config"])
    env.seed(seed)

    agent_policy_mapping = agent_policy_mapping or {}
    ones_require_buffer = ones_require_buffer or env.possible_agents

    if not evaluate:
        agents_monitored = ones_require_buffer
    else:
        agents_monitored = (
            env.possible_agents if groups is None else list(groups.keys())
        )

    total_cnt = {agent: 0 for agent in agents_monitored}
    mean_episode_reward = defaultdict(list)
    mean_episode_len = defaultdict(list)
    episode_num = 0

    while (
        any(
            [
                agent_total_cnt < fragment_length
                for agent_total_cnt in total_cnt.values()
            ]
        )
        and episode_num < max_episode
    ):
        env.reset()
        general_call(
            agent_interface.reset_behavior_state,
            remote_interact,
            evaluate=evaluate,
            policy_mapping=agent_policy_mapping,
            rank=rank,
        )

        cnt = defaultdict(lambda: 0)
        agent_episode = defaultdict(list)
        episode_reward = defaultdict(lambda: 0.0)

        Log.debug(
            "\t++ [sequential rollout {}/{}] start new episode ++".format(
                list(total_cnt.values()), fragment_length
            )
        )
        last_action_dist = {player_id: None for player_id in env.agents}
        last_logits = {player_id: None for player_id in env.agents}
        hidden_state = dict.fromkeys(env.possible_agents, None)

        for player_id in env.agent_iter():
            observation, pre_reward, done, info = env.last()
            action_mask = general_call(
                agent_interface.compute_action_mask,
                remote_interact,
                observation={player_id: observation},
                agents=[player_id],
            )[player_id]
            observation = general_call(
                agent_interface.transform_observation,
                remote_interact,
                observation={player_id: observation},
                agent=[player_id],
            )[player_id]
            if not done:
                tmp = general_call(
                    agent_interface.compute_action,
                    remote_interact,
                    observation={player_id: observation},
                    action_mask={player_id: action_mask},
                    state=hidden_state,
                    evaluate=evaluate,
                    agents=[player_id],
                    rank=rank,
                )
                action, action_dist, logits, hidden_state[player_id] = [
                    e[player_id] for e in tmp
                ]
            else:
                action = None
                action_dist = None
                logits = None
                hidden_state = dict.fromkeys(env.possible_agents, None)
            env.step(action)
            Log.debug(
                "\t\t[agent={}] action={} action_dist={} logits={} pre_reward={}".format(
                    player_id, action, action_dist, logits, pre_reward
                )
            )
            if not evaluate and player_id in ones_require_buffer:
                # print(observation.shape, action_mask.shape, pre_reward, action, action_dist, logits)
                agent_episode[player_id].append(
                    _TimeStep(
                        observation,
                        action_mask,
                        pre_reward,
                        action
                        if action is not None
                        else env.action_spaces[player_id].sample(),
                        done,
                        # action_dist
                        # if action_dist is not None
                        # else last_action_dist[player_id],
                        logits if logits is not None else last_logits[player_id],
                    )
                )
            last_action_dist[player_id] = action_dist
            last_logits[player_id] = logits
            episode_reward[player_id] += pre_reward
            # print("prerew fed", pre_reward, player_id)
            cnt[player_id] += 1

            if all([agent_cnt >= max_step for agent_cnt in cnt.values()]):
                break
        Log.debug(
            "\t++ [sequential rollout] episode end at step={} ++".format(dict(cnt))
        )
        episode_num += 1
        if not evaluate:
            buffers = {}
            for player, data_tups in agent_episode.items():
                (
                    observations,
                    action_masks,
                    pre_rewards,
                    actions,
                    dones,
                    # action_dists,
                    logits,
                ) = tuple(map(np.stack, list(zip(*data_tups))))

                rewards = pre_rewards[1:].copy()
                dones = dones[1:].copy()
                next_observations = observations[1:].copy()
                next_action_masks = action_masks[1:].copy()
                next_logits = logits[1:].copy()

                observations = observations[:-1].copy()
                action_masks = action_masks[:-1].copy()
                actions = actions[:-1].copy()
                # action_dists = action_dists[:-1].copy()
                logits = logits[:-1].copy()

                buffers[player] = Episode(
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    action_masks,
                    dones,
                    # action_dists,
                    logits,
                    extras={
                        "next_action_mask": next_action_masks,
                        "next_logits": next_logits,
                    },
                ).clean_data()
            general_call(
                agent_interface.add_batches, remote_interact, buffers=buffers, rank=rank
            )
            if not episodic and not remote_interact:
                if (
                    agent_interface.sampler.is_ready()
                    and agent_interface.sampler.size % train_every == 0
                ):
                    yield {"timesteps": cnt[agents_monitored[0]]}

        # pack into batch
        for agent in agents_monitored:
            total_cnt[agent] += cnt[agent]

        for k, v in episode_reward.items():
            mean_episode_reward[k].append(v)
            mean_episode_len[k].append(cnt[k])

    mean_episode_reward = {
        k: sum(v) / len(v)
        for k, v in mean_episode_reward.items()
        if k in agents_monitored
    }
    mean_episode_len = {
        k: sum(v) / len(v) for k, v in mean_episode_len.items() if k in agents_monitored
    }

    if evaluate:
        return {
            "reward": mean_episode_reward,
            "episode_len": mean_episode_len,
        }
    else:
        res = {
            "total_timesteps": total_cnt[agents_monitored[0]],
            "num_episode": episode_num,
        }
        return res


def get_rollout_func(type_name: str) -> type:
    """Return rollout function by name.

    Args:
        type_name (str): The type name of rollout function. Could be `sequential` or `simultaneous`.

    Raises:
        TypeError: Unsupported rollout func type.

    Returns:
        type: Rollout caller.
    """

    handler = None
    if type_name == "simultaneous":
        handler = simultaneous_rollout
    elif type_name == "sequential":
        handler = sequential_rollout
    else:
        raise TypeError("Unsupported rollout func type: %s" % type_name)
    return handler


class Evaluator:
    def __init__(self, env_desc, n_env: int = 1, use_remote_env: bool = False) -> None:
        """Initialize an evaluator with given environment configuration. Specifically, `env_desc` for instance generation,
        `n_env` indicates the number of environments, >1 triggers VectorEnv. `use_remote_env` for VectorEnv mode only.
        """

        if n_env > 1:
            self.env = VectorEnv(env_desc, n_env, use_remote=use_remote_env)
        else:
            self.env = env_desc["creator"](**env_desc["config"])

        self.env_desc = env_desc

    def terminate(self):
        self.env.close()

    def run(
        self,
        policy_mappings: List[Dict[AgentID, PolicyID]],
        max_step: int,
        fragment_length: int,
        agent_interface: InteractInterface,
        rollout_caller: type,
        seed: int = None,
        max_episode: int = 10,
    ) -> Sequence[Tuple[Dict, Dict]]:
        """Accept a sequence of policy mapping description, then evaluate them sequentially.

        Args:
            policy_mappings (List[Dict[AgentID, PolicyID]]): A sequence of policy mapping, describes the policy selection by all agents.
            max_step (int): Max step of one episode.
            fragment_length (int): Fragment length of a data batch.
            agent_interface (InteractInterface): An instance of Interactinterfaces.
            rollout_caller (type): Rollout callback function, could be `simultaneous` or `sequential`.
            seed (int, optional): Random seed. Default by None
            max_episode (int, optional): Indicates the maximum of episodes, only be activated for `evaluate=True`. Default by 10.

        Returns:
            Sequence[Tuple[Dict, Dict]]: A sequence of evaluation feedabck, corresponding to policy mappings.
        """

        res = []

        if isinstance(self.env, VectorEnv):
            self.env._limits = max_episode

        rollout_config = dict(
            fragment_length=fragment_length,
            max_step=max_step,
            train_every=-1,  # mute generator
            max_episode=max_episode,
        )

        if policy_mappings is not None:
            for policy_mapping in policy_mappings:
                rets = monitor(enable_timer=True, enable_returns=True, prefix="\t")(
                    rollout_caller
                )(
                    agent_interface=agent_interface,
                    env_description=self.env_desc,
                    rollout_config=rollout_config,
                    agent_policy_mapping=policy_mapping,
                    evaluate=True,
                    seed=seed,
                    env=self.env,
                )
                reward = 0
                try:
                    while True:
                        _ = next(rets)
                except StopIteration as e:
                    reward = e.value
                res.append((policy_mapping, reward))
        else:
            rets = monitor(enable_timer=True, enable_returns=True, prefix="\t")(
                rollout_caller
            )(
                agent_interface=agent_interface,
                env_description=self.env_desc,
                rollout_config=rollout_config,
                agent_policy_mapping=None,
                evaluate=True,
                seed=seed,
                env=self.env,
            )
            try:
                while True:
                    _ = next(rets)
            except StopIteration as e:
                res = [(None, e.value)]
        return res
