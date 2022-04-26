from abc import ABCMeta, abstractmethod
from collections import ChainMap, defaultdict
from enum import IntEnum
from typing import Dict, Sequence, Any, Union, Tuple, Callable, Type
import gym
import random
import torch
import torch.nn as nn
import numpy as np

from gym import spaces
from torch.distributions.categorical import Categorical

from common.data import DataArray, EpisodeKeys
from common.preprocessor import get_preprocessor
from common.data import tensor_cast, Batch
from common.distributions import make_proba_distribution, Distribution


PolicyID = str
AgentID = str
DataArray = Type[np.ndarray]


class PolicyStatus(IntEnum):
    """PolicyStatus indicates the policy status as FIXED(0), LOWEST_ACTIVE(1) and ACTIVE(2).
    FIXED means policy cannot be trained anymore; LOWEST_ACTIVE means that a policy can be trained
    by playing against fixed policies; ACTIVE means that a policy can be trained by playing against
    either FIXED, LOWEST_ACTIVE, ACTIVE policies.

    Args:
        IntEnum ([type]): ...
    """

    FIXED = 0
    LOWEST_ACTIVE = 1
    ACTIVE = 2


class SimpleObject:
    def __init__(self, obj, name):
        assert hasattr(obj, name), f"Object: {obj} has no such attribute named `{name}`"
        self.obj = obj
        self.name = name

    def load_state_dict(self, v):
        setattr(self.obj, self.name, v)

    def state_dict(self):
        value = getattr(self.obj, self.name)
        return value


Action = Any
ActionDist = Any
Logits = Any


class Policy(metaclass=ABCMeta):
    def __init__(
        self,
        observation_space,
        action_space,
        model_config,
        custom_config,
        is_fixed: bool = False,
    ):
        _locals = locals()
        _locals.pop("self")
        self._init_args = _locals
        self._observation_space = observation_space
        self._action_space = action_space
        self._model_config = model_config or {}
        self._custom_config = custom_config or {}
        self._state_handler_dict = {}
        self._is_fixed = is_fixed
        self._preprocessor = get_preprocessor(
            observation_space,
            mode=self._custom_config.get("preprocess_mode", "flatten"),
        )(observation_space)

        self._device = torch.device(
            "cuda" if self._custom_config.get("use_cuda") else "cpu"
        )

        self._policy_status = PolicyStatus.ACTIVE
        self._registered_networks: Dict[str, nn.Module] = {}

        self.use_cuda = self._custom_config.get("use_cuda", False)
        self.dist_fn: Distribution = make_proba_distribution(
            action_space=action_space,
            use_sde=custom_config.get("use_sde", False),
            dist_kwargs=custom_config.get("dist_kwargs", None),
        )

        if isinstance(action_space, spaces.Discrete):
            self.action_type = "discrete"
        elif isinstance(action_space, spaces.Box):
            self.action_type = "continuous"
        else:
            raise NotImplementedError(
                "Does not support other action space type settings except Box and Discrete. {}".format(
                    type(action_space)
                )
            )

    @property
    def model_config(self):
        return self._model_config

    @property
    def status(self) -> PolicyStatus:
        return self._policy_status

    def switch_status_to(self, status: PolicyStatus):
        """Update policy status to `status`.

        Args:
            status (PolicyStatus): A PolicyStatus signal.
        """

        self._policy_status = status

    @property
    def device(self) -> str:
        return self._device

    @property
    def custom_config(self) -> Dict[str, Any]:
        return self._custom_config

    @property
    def is_fixed(self) -> bool:
        """Checkout whether current policy is fixed or not. Fixed means cannot be trained.

        Returns:
            bool: A bool value indicates whether current policy is trainable or not.
        """

        return self._is_fixed

    # TODO(ming): will be replaced with PolicyStatus
    @is_fixed.setter
    def is_fixed(self, value: bool):
        self._is_fixed = value

    def set_fixed(self, **kwargs):
        """Set policy be fixed."""

        self._is_fixed = True
        # then convert to cpu
        if "device" in kwargs:
            self.to(device=kwargs["device"], use_copy=False)

    @property
    def target_actor(self):
        return self._target_actor

    @target_actor.setter
    def target_actor(self, value: Any):
        self._target_actor = value

    @property
    def actor(self):
        return self._actor

    @actor.setter
    def actor(self, value: Any):
        self._actor = value

    @property
    def critic(self):
        return self._critic

    @critic.setter
    def critic(self, value: Any):
        self._critic = value

    @property
    def target_critic(self):
        return self._target_critic

    @target_critic.setter
    def target_critic(self, value: Any):
        self._target_actor = value

    def value_function(self, **kwargs):
        pass

    def action_value_function(self, batch: Batch, evaluate: bool, state: Any):
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict outside.

        Args:
            state_dict (Dict[str, Any]): A dict of states.
        """

        for k, v in state_dict.items():
            self._state_handler_dict[k].load_state_dict(v)

    def state_dict(self):
        """Return state dict in real time"""

        res = {k: v.state_dict() for k, v in self._state_handler_dict.items()}
        return res

    def register_state(self, obj: Any, name: str) -> None:
        """Register state of obj. Called in init function to register model states.

        Example:
            >>> class CustomPolicy(Policy):
            ...     def __init__(
            ...         self,
            ...         registered_name,
            ...         observation_space,
            ...         action_space,
            ...         model_config,
            ...         custom_config
            ...     ):
            ...     # ...
            ...     actor = MLP(...)
            ...     self.register_state(actor, "actor")

        Args:
            obj (Any): Any object, for non `torch.nn.Module`, it will be wrapped as a `Simpleobject`.
            name (str): Humanreadable name, to identify states.

        Raises:
            errors.RepeatedAssignError: [description]
        """

        # if not isinstance(obj, nn.Module):
        if obj.__class__.__module__ == "builtins":
            obj = SimpleObject(self, name)
        if self._state_handler_dict.get(name, None) is not None:
            raise errors.RepeatedAssignError(
                f"state handler named with {name} is not None."
            )
        self._state_handler_dict[name] = obj
        if isinstance(obj, nn.Module):
            self._registered_networks[name] = obj

    def deregister_state(self, name: str):
        if self._state_handler_dict.get(name) is None:
            print(f"No such state tagged with: {name}")
        else:
            self._state_handler_dict.pop(name)
            print(f"Deregister state tagged with: {name}")

    @property
    def preprocessor(self):
        return self._preprocessor

    @abstractmethod
    def compute_action(
        self, batch: Batch, evaluate, state: Any = None, **kwargs
    ) -> Tuple[Action, ActionDist, Logits, Any]:
        pass

    def save(self, path, global_step=0, hard: bool = False):
        state_dict = {"global_step": global_step, **self.state_dict()}
        torch.save(state_dict, path)

    def load(self, path: str):
        state_dict = torch.load(path)
        print(
            f"[Model Loading] Load policy model with global step={state_dict.pop('global_step')}"
        )
        self.load_state_dict(state_dict)

    def reset(self, **kwargs):
        """Reset parameters or behavior policies."""

        pass

    @classmethod
    def copy(cls, instance, replacement: Dict):
        return cls(replacement=replacement, **instance._init_args)

    @property
    def registered_networks(self) -> Dict[str, nn.Module]:
        return self._registered_networks

    def to(self, device: str = None, use_copy: bool = False) -> "Policy":
        """Convert policy to a given device. If `use_copy`, then return a copy. If device is None, do not change device.

        Args:
            device (str): Device identifier.
            use_copy (bool, optional): User copy or not. Defaults to False.

        Raises:
            NotImplementedError: Not implemented error.

        Returns:
            Policy: A policy instance
        """

        if device is None:
            device = "cpu" if not self.use_cuda else "cuda"

        cond1 = "cpu" in device and self.use_cuda
        cond2 = "cuda" in device and not self.use_cuda

        if "cpu" in device:
            use_cuda = False
        else:
            use_cuda = self._custom_config.get("use_cuda", False)

        replacement = {}
        if cond1 or cond2:
            # retrieve networks here
            for k, v in self.registered_networks.items():
                _v = v.to(device)
                if not use_copy:
                    setattr(self, k, _v)
                else:
                    replacement[k] = _v
        else:
            # fixed bug: replacement cannot be None.
            for k, v in self.registered_networks.items():
                replacement[k] = v

        if use_copy:
            # FIXME(ming): copy has bug, when policy is trainable!!!!!!
            ret = self.copy(self, replacement=replacement)
        else:
            self.use_cuda = use_cuda
            ret = self

        return ret

    def parameters(self) -> Dict[str, Dict]:
        """Return trainable parameters."""

        res = {}
        for name, net in self.registered_networks.items():
            res[name] = net.parameters()
        return res

    def named_parameters(self) -> Dict[str, Dict]:
        """Return a dict of named parameters.

        Returns:
            Dict[str, Dict]: A dict of parameters for each network.
        """

        res = {}
        for name, net in self.registered_networks.items():
            res[name] = net.named_parameters()
        return res

    def update_parameters(self, parameter_dict: Dict[str, Any]):
        """Update local parameters with given parameter dict.

        Args:
            parameter_dict (Dict[str, Parameter]): A dict of paramters
        """

        for k, parameters in parameter_dict.items():
            target = self.registered_networks[k]
            for target_param, param in zip(target.parameters(), parameters):
                target_param.data.copy_(param.data)


class AggPolicy(Policy):
    def __init__(
        self,
        agent_id: AgentID,
        policies: Dict[PolicyID, Policy],
        prior: Dict[PolicyID, float],
        action_func: Callable,
        enable_variance_control: bool,
    ):

        selected = list(policies.values())[0]
        observation_space = selected._observation_space
        action_space = selected._action_space
        model_config = selected._model_config
        custom_config = selected._custom_config

        super(AggPolicy, self).__init__(
            observation_space, action_space, model_config, custom_config, is_fixed=True
        )

        prior_candidates = []
        support_candidates = []

        for pid, prob in prior.items():
            support = policies[pid]
            if isinstance(support, AggPolicy):
                scaled_prior, supports = self.return_scaled_supports(prob)
                prior_candidates.append(scaled_prior)
                support_candidates.append(supports)
            else:
                prior_candidates.append({pid: prob})
                support_candidates.append({pid: support})

        # merge prior and supports
        self.policies = dict(ChainMap(*support_candidates))
        self.prior = defaultdict(lambda: 1.0)
        for item in prior_candidates:
            for k, v in item.items():
                self.prior[k] *= v

        # filtered_prior = {}
        # filtered_poliies = {}
        filtered_prior = []
        filtered_poliies = []
        for k, v in self.prior.items():
            if np.isclose(v, 0.0):
                continue
            # filtered_prior[k] = v
            # filtered_poliies[k] = self.policies[k]
            filtered_prior.append(v)
            filtered_poliies.append(self.policies[k])

        self.filtered_prior = filtered_prior
        self.filtered_policies = filtered_poliies
        # filter zero weighted policy
        self.action_func = action_func
        self.enable_variance_control = enable_variance_control

    def return_scaled_supports(self, scale: float):
        """Return scaled supports and clean this instance.

        Args:
            scale (float): A float for scaling.

        Returns:
            Tuple[Dict, Dict]: A tuple of dicts.
        """

        supports = {}
        new_prior = {}
        for k in self.prior.keys():
            new_prior[k] = self.prior[k] * scale
            supports[k] = self.policies[k]

        # self.policies = {}
        # self.prior = {}
        # self.action_func = None

        return new_prior, supports

    def to(self, device: str = None, use_copy: bool = False) -> "Policy":
        self.filtered_policies = []
        self.filtered_prior = []
        for k, v in self.policies.items():
            self.policies[k] = v.to(device, use_copy)
            if np.isclose(self.prior[k], 0.0):
                continue
            self.filtered_policies.append(self.policies[k])
            self.filtered_prior.append(self.prior[k])
        return self

    def compute_action(
        self, batch: Batch, evaluate, state: Any = None, **kwargs
    ) -> Tuple[Action, ActionDist, Logits, Any]:
        return self.action_func(
            self.filtered_policies,
            self.filtered_prior,
            batch,
            evaluate,
            state,
            self.enable_variance_control,
        )

    def value_function(self, **kwargs):
        raise NotImplementedError


def pack_action_to_policy(
    action_set: Sequence[Any],
    observation_space: gym.Space = None,
    is_fixed: bool = True,
    distribution: np.ndarray = None,
) -> Policy:
    """Pack an action set (maybe a dict) as a policy, without NN estimator.

    Args:
        action_set (Sequence[Any]): A sequence of an action set / policy set.
        observation_space (gym.Space, optional): Observation space, if the policy is stateless, it can be None. Defaults to None.
        is_fixed (bool, optional): Indicate whether current policy is fixed or not.
        distribution (np.ndarray, optional): Policy placeholder if `is_fixed` is True. Default by None.

    Returns:
        Policy: A customized policy instance.
    """

    class custom_policy(Policy):
        def __init__(self, observation_space, action_set, is_fixed):
            super(custom_policy, self).__init__(
                observation_space, None, None, None, is_fixed
            )

            self.action_is_policy = isinstance(action_set[0], Policy)
            self.action_set = np.asarray(action_set, dtype=type(action_set[0]))

            self.weights = make_proba_distribution(
                gym.spaces.Discrete(len(self.action_set))
            )

            logits = torch.tensor(
                np.random.random(size=(1, len(self.action_set))),
                requires_grad=True,
                dtype=torch.float32,
            )
            self.logits = logits
            self.weights = self.weights.proba_distribution(logits)

            if not self.is_fixed:
                self.optimizer = torch.optim.SGD([self.logits], lr=1.0)
            else:
                assert (
                    distribution is not None
                ), "Fixed policy cannot be initialized with an empty distribution over the action space."
                assert (
                    distribution.dtype == np.float32
                ), "Data type should be `np.float32`, while {} found".format(
                    distribution.dtype
                )
                self.weights.distribution = Categorical(
                    probs=torch.from_numpy(distribution)
                )
                self.optimizer = None

        def compute_action(
            self, observation, action_mask, evaluate
        ) -> Tuple[Action, ActionDist]:
            if evaluate:
                # TODO(ming): argmax
                idx = (
                    torch.argmax(self.weights.distribution.probs, dim=-1)
                    if not self.is_fixed
                    else self.weights.sample()
                )
                action = self.action_set[idx]
            else:
                idx = self.weights.sample()
                action = self.action_set[idx]

            prob = (
                self.weights.distribution.probs.detach()
                .numpy()
                .squeeze()
                .astype(np.float32)
            )
            if self.action_is_policy:
                action, action_dist = action.compute_action(
                    observation, action_mask, evaluate
                )
                prob = np.asarray(action_dist, dtype=np.float32)  # * prob[idx]

            return action, prob

        def compute_actions(self, observation):
            indices = self.weights.sample(sample_shape=(len(observation),))
            actions = self.action_set[indices]

            if self.action_is_policy:
                actions = [e.compute_actions(observation) for e in actions]

            return actions

        @property
        def br(self):
            """BR means non-zero argmax support.

            Raises:
                NotImplementedError: [description]
            """
            idx = self.weights.mode()
            br = self.action_set[idx]
            return br

        def reset(self, **kwargs):
            self.is_fixed = kwargs.get("is_fixed", self.is_fixed)
            policy_id = kwargs.get("policy_id", None)

            # make prob be one hot
            # assert type(policy_id) == type(self.action_set[0]), (type(policy_id), type(self.action_set[0]))
            # prob = [0.] * len(self.action_set)
            # chosen_idx = np.argwhere(self.action_set == policy_id)
            # assert len(chosen_idx) > 0, (self.action_set, policy_id, chosen_idx)
            # prob[chosen_idx] = 1.
            # self.weights.distribution = Categorical(
            #     probs=torch.Tensor(prob)
            # )

        @tensor_cast()
        def optimize(self, batch: Dict[str, DataArray]) -> Dict[str, float]:
            """Policy gradient optimization.

            Args:
                batch (Dict[str, DataArray]): Sample batch.

            Returns:
                Dict[str, float]: Optimization report.
            """

            assert (
                self.is_fixed is False
            ), "Policy has been fixed, cannot be optimized anymore!"
            actions = batch[EpisodeKeys.ACTION.value]
            rewards = batch[EpisodeKeys.ACC_REWARD.value]

            # WARNING: currently we have no need to convert actions to the corresponding indices in weights since
            #   the learnable logits is a full action set.
            log_probs = self.weights.log_prob(actions).reshape_as(rewards)
            loss = -torch.sum(log_probs * rewards)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print("logits: ", self.logits)
            self.weights = self.weights.proba_distribution(self.logits)
            self.optimizer = torch.optim.SGD([self.logits], lr=1.0)
            return {
                "loss": loss.item(),
                "mean_reward": rewards.mean().item(),
                "max_reward": rewards.max().item(),
                "min_reward": rewards.min().item(),
                "entropy": self.weights.entropy().item(),
            }

    return custom_policy(observation_space, action_set, is_fixed)
