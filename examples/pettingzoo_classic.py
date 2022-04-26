"""
Training script for Pettingzoo poker games.

@status: passed
@author: Ming Zhou
@organization: SJTU-MARL
@date: 2022/04/26 01:04:42 PM
"""

from typing import Dict, Any

import os
import argparse
import gym

import numpy as np

from torch.utils import tensorboard
from common.preprocessor import get_preprocessor

from ptzoo import env_desc_gen
from ptzoo.algorithms import pg
from common.agent_interface import AgentInterface, AgentInterfaceManager
from common.sampler import get_sampler
from common.rollout import get_rollout_func, Evaluator, sequential_rollout
from common.logger import write_to_tensorboard
from common.data import EpisodeKeys


def basic_sampler_config(
    observation_space: gym.Space,
    action_space: gym.Space,
    preprocessor: object,
    capacity: int = 1000,
    learning_starts: int = 64,
):
    sampler_config = {
        "dtypes": {
            EpisodeKeys.ACTION_MASK.value: float,
            EpisodeKeys.REWARD.value: float,
            EpisodeKeys.NEXT_OBSERVATION.value: float,
            EpisodeKeys.DONE.value: bool,
            EpisodeKeys.OBSERVATION.value: float,
            EpisodeKeys.ACTION.value: int,
            # EpisodeKeys.ACTION_DIST.value: float,
            EpisodeKeys.NEXT_ACTION_MASK.value: float,
        },
        "data_shapes": {
            EpisodeKeys.ACTION_MASK.value: (action_space.n,),
            EpisodeKeys.REWARD.value: (),
            EpisodeKeys.NEXT_OBSERVATION.value: preprocessor.shape,
            EpisodeKeys.DONE.value: (),
            EpisodeKeys.OBSERVATION.value: preprocessor.shape,
            EpisodeKeys.ACTION.value: (),
            # EpisodeKeys.ACTION_DIST.value: (action_space.n,),
            EpisodeKeys.NEXT_ACTION_MASK.value: (action_space.n,),
        },
        "capacity": capacity,
        "learning_starts": learning_starts,
    }

    return sampler_config


def main(args: argparse.Namespace):
    env_desc = argparse.Namespace(
        **env_desc_gen(env_id=args.env_id, scenario_config={})
    )

    obs_spaces = env_desc.config["observation_spaces"]
    act_spaces = env_desc.config["action_spaces"]
    rollout_config = {
        "fragment_length": args.fragment_length,
        "train_every": -1,  # -1 means episodic, > 0 means train policy every `train_every` time step
        "max_episode": args.max_episode,  # for evaluation
        "max_step": args.max_step,
    }

    trainers = {
        aid: pg.PGTrainer(
            learning_mode="on_policy",
            training_config=pg.DEFAULT_CONFIG["training_config"],
            policy_instance=pg.PGPolicy(
                observation_space=obs_spaces[aid],
                action_space=act_spaces[aid],
                model_config=pg.DEFAULT_CONFIG["model_config"],
                custom_config=pg.DEFAULT_CONFIG["custom_config"],
                is_fixed=False,
            ),
        )
        for aid in env_desc.config["possible_agents"]
    }

    policies = {k: v.policy for k, v in trainers.items()}

    selected_obs_space = list(obs_spaces.values())[0]
    sampler = get_sampler(
        env_desc.config["possible_agents"],
        sampler_config=basic_sampler_config(
            selected_obs_space,
            action_space=list(act_spaces.values())[0],
            preprocessor=get_preprocessor(selected_obs_space)(selected_obs_space),
            capacity=args.memory_size,
            learning_starts=args.learning_starts,
        ),
    )

    # pack agent interfaces into agent manager
    agent_manager = AgentInterfaceManager(
        {
            k: AgentInterface(
                policy_name="pg",
                policy=policies[k],
                observation_space=obs_spaces[k],
                action_space=act_spaces[k],
                is_active=True,
            )
            for k in env_desc.config["possible_agents"]
        },
        agent_mapping=lambda agent: agent,
    )
    # register sampler for experience collection
    agent_manager.register_sampler(sampler)
    evaluator = Evaluator(
        env_desc.__dict__, n_env=1, use_remote_env=args.use_remote_env
    )
    writer = tensorboard.SummaryWriter(log_dir=args.log_dir)

    # init evaluator for
    for epoch_th in range(args.num_epoch):
        if (epoch_th + 1) % 5 == 0:
            e_info = evaluator.run(
                None,
                max_step=args.max_step,
                fragment_length=args.fragment_length,
                agent_interface=agent_manager,
                rollout_caller=sequential_rollout,
            )[0][1]
            write_to_tensorboard(
                writer, e_info, global_step=epoch_th, prefix="evaluation"
            )
            print("eval at epoch={}: {}".format(epoch_th, e_info))

        generator = sequential_rollout(
            agent_manager,
            env_desc.__dict__,
            rollout_config,
            evaluate=False,
            seed=args.seed,
        )

        try:
            while True:
                _ = next(generator)
        except StopIteration as e:
            info = e.value

        # training
        buffers = sampler.get_buffer()
        t_info = {}
        for agent_id, trainer in trainers.items():
            t_info[agent_id] = trainer(time_step=epoch_th, buffer=buffers[agent_id])
        write_to_tensorboard(writer, t_info, global_step=epoch_th, prefix="training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training poker games from Petting Zoo.")
    parser.add_argument(
        "--env_id",
        type=str,
        default="classic.leduc_holdem_v4",
        help="registered environment id in pettingzoo.",
    )
    parser.add_argument(
        "--fragment_length", type=int, default=20, help="data block size for training."
    )
    parser.add_argument(
        "--max_episode",
        type=int,
        default=5,
        help="indicate how many episodes will be generated for evaluation.",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=20,
        help="indicates how many cycles for each episode.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=5,
        help="indicate how many environments will be used for evaluation.",
    )
    parser.add_argument(
        "--use_remote_env",
        action="store_true",
        help="enable remote enviornment caller or not.",
    )
    parser.add_argument(
        "--num_epoch", type=int, default=100, help="training epoch num."
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=1000,
        help="specification of the memory size.",
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=10,
        help="specification minimal size of training start.",
    )
    parser.add_argument("--seed", type=int, default=1, help="environment seed")
    parser.add_argument(
        "--log_dir", type=str, default="./gtml_results", help="log directory."
    )

    args = parser.parse_args()

    main(args)
