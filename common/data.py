"""
Data structure.

@status: passed
@author: Ming Zhou
@organization: SJTU-MARL
@date: 2021/08/15
"""

import enum
import copy


from typing import Dict, Callable, Sequence, Any, Union, Optional, Type
from numbers import Number
from dataclasses import dataclass, fields

import torch
import numpy as np

from numba import njit
from torch.utils.data import Dataset, DataLoader

from common.logger import Log
from common.tianshou_batch import Batch, _parse_value


AgentID = str
DataArray = Type[np.ndarray]


class EpisodeKeys(enum.Enum):
    OBSERVATION = "observation"
    REWARD = "reward"
    ACC_REWARD = "accumulative_reward"
    NEXT_OBSERVATION = "next_observation"
    ACTION = "action"
    ACTION_MASK = "action_mask"
    ACTION_LOGITS = "logits"
    NEXT_ACTION_LOGITS = "next_logits"
    NEXT_ACTION_MASK = "next_action_mask"
    DONE = "done"
    ACTION_DIST = "action_distribution"
    GLOBAL_STATE = "global_state"
    NEXT_GLOBAL_STATE = "next_global_state"
    STATE_VALUE = "state_value"
    INFO = "info"


@dataclass
class Episode:
    observation: DataArray
    action: DataArray
    reward: DataArray
    next_observation: DataArray
    action_mask: DataArray
    done: DataArray
    # action_distribution: DataArray
    logits: DataArray
    extras: Dict[str, DataArray] = None

    def __post_init__(self):
        """Check length consistency."""

        # check shape
        lens_dict = {
            field.name: len(getattr(self, field.name))
            for field in fields(self)
            if field.name is not "extras" and getattr(self, field.name) is not None
        }
        assert (
            len(set(lens_dict.values())) == 1
        ), f"Inconsistency between fields: {lens_dict}"
        if self.extras is not None:
            lens_dict = {k: len(v) for k, v in self.extras.items()}
            lens_set = set(lens_dict.values())
            assert len(lens_set) == 1 and lens_set.pop() == len(
                self.observation
            ), f"Inconsitency in extras: {lens_dict} expected length is: {len(self.observation)}"

    def clean_data(self):
        res = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if field.name is not "extras" and v is not None:
                res[field.name] = v
        if self.extras is not None:
            for k, v in self.extras.items():
                if v is not None:
                    res[k] = v
        for k, v in res.items():
            Log.debug("cleaned data for %s with shape %s", k, v.shape)
        return res

    def visualize(self):
        """Visualize current episode in tabular.

        ------------------    -------------- -----
        name                  shape          bytes
        'observation'         (n_batch, ...) xxx
        'action'              (n_batch, ...) xxx
        'reward'              (n_batch,)     xxx
        'next_observation'    (n_batch, ...) xxx
        'action_mask'         (n_batch, ...) xxx
        'done'                (n_batch,)     xxx
        'action_distribution' (n_batch, ...) xxx
        ------------------ -------------- -----
        """

        raise NotImplementedError


class Memory(Dataset):
    def __init__(self, extra_keys: Sequence[str] = None):
        """Initialize a memory dataset.

        Args:
            extra_keys (Sequence[str], optional): Extra keys for data training. Defaults to None.
        """

        self.observation = []
        self.action = []
        self.reward = []
        self.done = []
        self.next_observation = []

        self.data_keys = ["observation", "action", "reward", "done", "next_observation"]
        if extra_keys is not None:
            for key in extra_keys:
                setattr(self, key, [])
            self.data_keys.extend(extra_keys)
        self.data_keys = tuple(self.data_keys)

    def __len__(self) -> int:
        return len(self.done)

    def __getitem__(self, idx: int) -> Batch:
        res = {}
        for k in self.data_keys:
            data = getattr(self, k)
            data = np.array(data[idx], dtype=np.float32)
            res[k] = data
        return res

    def get_all(self, device: str = None) -> Dict:
        res = {}
        for k in self.datakeys:
            res[k] = getattr(self, k)
        return res

    def save_all(
        self, observation, action, reward, done, next_observation, **extra_data
    ):
        local_kwargs = locals()
        local_kwargs.update(extra_data)

        for k in self.data_keys:
            setattr(self, k, local_kwargs[k])

    def save_transition(
        self, observation, action, reward, done, next_observation, **extra_data
    ):
        local_kwargs = locals()
        local_kwargs.update(**extra_data)
        for k in self.data_keys:
            data = getattr(self, k)
            data.append(local_kwargs[k])

    def clear_memory(self):
        for k in self.data_keys:
            del getattr(self, k)[:]

    @classmethod
    def from_dict(cls, dict_data: Dict[str, np.ndarray]) -> "Memory":
        """Generate a memory instance from a given buffer dict.

        Args:
            dict_data (Dict[str, np.ndarray]): A buffer dict.

        Returns:
            Memory: A memory instance.
        """

        instance = cls(extra_keys=list(dict_data.keys()))
        instance.save_all(**dict_data)
        return instance


class MultiAgentMemory(Memory):
    def __init__(self, agent_ids: Sequence[AgentID], extra_keys: Sequence[str] = None):
        super().__init__(extra_keys)
        self.agent_memory = {
            agent: Memory(extra_keys=extra_keys) for agent in agent_ids
        }

    def __len__(self) -> Dict[AgentID, int]:
        return sum(map(len, self.agent_memory.values()))

    def __getitem__(self, idx: int):
        # concate here
        res = {k: [] for k in self.data_keys}
        for v in self.agent_memory.values():
            for k, _v in v[idx].items():
                res[k].append(_v)
        for k, v in res.items():
            res[k] = np.stack(v).squeeze()
        return res

    def get_all(self, device: str = None) -> Dict:
        return {agent: v.get_all(device) for agent, v in self.agent_memory.items()}

    def save_all(self, agent_buffer: Dict[AgentID, Dict], copy: bool = True):
        for agent, buffer in agent_buffer.items():
            if isinstance(buffer, Memory):
                self.agent_memory[agent] = buffer.copy() if copy else buffer
            else:
                self.agent_memory[agent].save_all(**buffer)

    def save_transition(self, agent_trans: Dict[AgentID, Dict]):
        for agent, trans in agent_trans.items():
            self.agent_memory[agent].save_transition(**trans)

    def clear_memory(self):
        for v in self.agent_memory.values():
            v.clear_memory()

    @classmethod
    def from_dict(
        cls, dict_data: Dict[AgentID, Dict[str, np.ndarray]], copy: bool = False
    ):
        """Generate a memory instance from a given buffer dict.

        Args:
            dict_data (Dict[str, np.ndarray]): A buffer dict.

        Returns:
            Memory: A memory instance.
        """

        selected = list(dict_data.values())[0]
        if isinstance(selected, Memory):
            all_keys = selected.data_keys
        else:
            all_keys = list(selected.keys())
        instance = cls(agent_ids=list(dict_data.keys()), extra_keys=all_keys)
        instance.save_all(dict_data, copy)
        return instance


class NumpyDataset(Dataset):
    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        dtypes: Dict[str, Any],
        capacity: int,
    ):
        registered_data_keys = list(data_shapes.keys())

        self.data = {
            k: np.zeros((capacity,) + shape, dtype=dtypes[k])
            for k, shape in data_shapes.items()
        }
        self.capacity = capacity
        self.registered_data_keys = registered_data_keys
        self.runtime_episode_buffer = {}

        self._size = 0
        self._flag = 0

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Batch:
        res = {}
        for k in self.registered_data_keys:
            res[k] = self.data[k][idx]
        return res

    def save_episodes(self, episodes: Sequence[Any]):
        """Accept a sequence of episodes, then save them batch by batch as vectorized=False.

        Args:
            episodes (Sequence[Any]): A sequence of episodes
        """

        for episode in episodes:
            self.save_batch(False, episode)

    def save_transition(self, **transition):
        idx = self._flag
        for k, v in transition.items():
            if k not in self.data:
                self.data[k] = np.zeros((self.capacity,) + v.shape, dtype=v.dtype)
            self.data[k][idx] = v

    def clear(self):
        self._size = 0
        self._flag = 0

    @classmethod
    def from_dict(cls, dict_data: Dict[str, np.ndarray]) -> "Memory":
        """Generate a memory instance from a given buffer dict.

        Args:
            dict_data (Dict[str, np.ndarray]): A buffer dict.

        Returns:
            Memory: A memory instance.
        """

        instance = cls(extra_keys=list(dict_data.keys()))
        instance.save_all(**dict_data)
        return instance


def default_dtype_mapping(dtype):
    # FIXME(ming): cast 64 to 32?
    if dtype in [np.int32, np.int64, int]:
        return torch.int32
    elif dtype in [float, np.float32]:
        return torch.float32
    elif dtype == np.float64:
        return torch.float64
    elif dtype in [bool, np.bool_]:
        return torch.float32
    else:
        raise NotImplementedError(f"dtype: {dtype} has no transmission rule.") from None


# wrap with type checking
def _walk(caster, v):
    if isinstance(v, Episode):
        v = v.__dict__
    elif isinstance(v, Dict):
        for k, _v in v.items():
            v[k] = _walk(caster, _v)
    else:
        v = caster(v)
    return v


def tensor_cast(
    custom_caster: Callable = None,
    callback: Callable = None,
    dtype_mapping: Dict = None,
    device="cpu",
):
    """Casting the inputs of a method into tensors if needed.

    Note:
        This function does not support recursive iteration.

    Args:
        custom_caster (Callable, optional): Customized caster. Defaults to None.
        callback (Callable, optional): Callback function, accepts returns of wrapped function as inputs. Defaults to None.
        dtype_mapping (Dict, optional): Specify the data type for inputs which you wanna. Defaults to None.

    Returns:
        Callable: A decorator.
    """
    dtype_mapping = dtype_mapping or default_dtype_mapping
    cast_to_tensor = custom_caster or (
        lambda x: torch.FloatTensor(x.copy()).to(
            device=device, dtype=dtype_mapping(x.dtype)
        )
        if not isinstance(x, torch.Tensor)
        else x
    )

    def decorator(func):
        def wrap(self, *args, **kwargs):
            new_args = []
            for i, arg in enumerate(args):
                new_args.append(_walk(cast_to_tensor, arg))
            for k, v in kwargs.items():
                kwargs[k] = _walk(cast_to_tensor, v)
            rets = func(self, *new_args, **kwargs)
            if callback is not None:
                callback(rets)
            return rets

        return wrap

    return decorator


def to_torch(
    x: Any,
    dtype: Optional[torch.dtype] = None,
    device: Union[str, int, torch.device] = "cpu",
) -> Union[Batch, torch.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(
        x.dtype.type, (np.bool_, np.number)
    ):  # most often case
        x = torch.from_numpy(x).to(device)  # type: ignore
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)  # type: ignore
    elif isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, (dict, Batch)):
        x = Batch(x, copy=True) if isinstance(x, dict) else copy.deepcopy(x)
        x.to_torch(dtype, device)
        return x
    elif isinstance(x, (list, tuple)):
        return to_torch(_parse_value(x), dtype, device)
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")


@njit
def _gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape, dtype=np.float32)
    delta = rew + v_s_ * gamma - v_s
    discount = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(rew.shape[0] - 1, -1, -1):
        gae = delta[i] + discount[i] * gae
        returns[i] = gae
    return returns


@njit
def _nstep_return(
    rew: np.ndarray,
    end_flag: np.ndarray,
    target_q: np.ndarray,
    indices: np.ndarray,
    gamma: float,
    n_step: int,
) -> np.ndarray:
    gamma_buffer = np.ones(n_step + 1)
    for i in range(1, n_step + 1):
        gamma_buffer[i] = gamma_buffer[i - 1] * gamma
    target_shape = target_q.shape
    bsz = target_shape[0]
    # change target_q to 2d array
    target_q = target_q.reshape(bsz, -1)
    returns = np.zeros(target_q.shape)
    gammas = np.full(indices[0].shape, n_step)
    for n in range(n_step - 1, -1, -1):
        now = indices[n]
        gammas[end_flag[now] > 0] = n + 1
        returns[end_flag[now] > 0] = 0.0
        returns = rew[now].reshape(bsz, 1) + gamma * returns
    target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
    return target_q.reshape(target_shape)


class EpisodeHandler:
    @staticmethod
    def gae_return(
        state_value,
        next_state_value,
        reward,
        done,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):

        adv = _gae_return(
            state_value, next_state_value, reward, done, gamma, gae_lambda
        )
        # assert (
        #     adv.shape
        #     == state_value.shape
        #     == next_state_value.shape
        #     == reward.shape
        #     == done.shape
        # ), (
        #     adv.shape,
        #     state_value.shape,
        #     next_state_value.shape,
        #     reward.shape,
        #     done.shape,
        # )
        return adv

    @staticmethod
    def compute_episodic_return(
        batch: Batch,
        state_value: np.ndarray = None,
        next_state_value: np.ndarray = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        rew = batch.reward.cpu().numpy()
        done = batch.done.cpu().numpy()

        if next_state_value is None:
            assert np.isclose(gae_lambda, 1.0)
            next_state_value = np.zeros_like(rew)
        else:
            # mask next_state_value
            next_state_value = next_state_value * (1.0 - done)

        state_value = (
            np.roll(next_state_value, 1) if state_value is None else state_value
        )

        # XXX(ming): why we clip the unfinished index?
        # end_flag = batch.done.copy()
        # truncated
        # end_flag[np.isin(indices, buffer.unfinished_index())] = True
        if gae_lambda == 0:
            returns = rew + gamma * next_state_value
        else:
            advantage = EpisodeHandler.gae_return(
                state_value,
                next_state_value,
                rew,
                done,
                gamma,
                gae_lambda,
            )
            returns = advantage + state_value
        # normalization varies from each policy, so we don't do it here
        return returns, advantage
