"""
For experience collection.

@status: passed
@author: Ming Zhou
@organization: SJTU-MARL
@date: 2021/08/15
"""

from typing import Any, Dict, Sequence, Union, List
from types import LambdaType

import os
import copy
import pickle
import os.path as osp

import numpy as np

from common.logger import Log
from common.data import Memory


AgentID = str


class SamplerInterface:
    def __init__(
        self, use_data_alignment: bool = False, learning_starts: int = -1
    ) -> None:
        self._use_data_alignment = use_data_alignment
        self._learning_starts = learning_starts

    @property
    def size(self):
        raise NotImplementedError

    @property
    def capacity(self):
        raise NotImplementedError

    @property
    def trainable_agents(self):
        raise NotImplementedError

    def is_ready(self):
        return self.size >= self._learning_starts

    def add_transition(self, vector_mode=False, **kwargs):
        raise NotImplementedError

    def add_batches(self, agent_batch: Dict[AgentID, Any]):
        raise NotImplementedError

    def get_buffer(
        self,
        size: int = -1,
        shuffle: bool = False,
        agent_filter: Sequence[AgentID] = None,
    ) -> Dict[AgentID, Memory]:
        """Return a dict of memory, mapping from agent to memory.

        Returns:
            Dict[AgentID, Memory]: A dict of memory
        """
        raise NotImplementedError

    def sample(
        self, idxes=None, batch_size=-1, agent_filter=None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        raise NotImplementedError

    def reset(self, trainable_agents: List[str]):
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError


class Sampler(SamplerInterface):
    """Default sampler, save batches for agents, do sampling for trainable agents, but ignores
    the agent interactions.
    """

    def __init__(
        self,
        agent_ids,
        dtypes,
        data_shapes,
        use_data_alignment: bool = False,
        capacity: int = 1000,
        learning_starts: int = 64,
        data_preprocessor: type = None,
    ):
        super(Sampler, self).__init__(use_data_alignment, learning_starts)
        self._agent_batch: Dict[str, Dict] = {
            aid: {
                k: np.zeros((capacity,) + shape, dtype=dtypes[k])
                for k, shape in data_shapes.items()
            }
            for aid in agent_ids
        }
        self._capacity = capacity
        self._trainable_agents = agent_ids
        self._pickle_name = "samples.pkl"
        self._flag = 0
        self._size = 0
        self._data_preprocessor = data_preprocessor or (lambda x: x)

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def trainable_agents(self):
        return self._trainable_agents

    def get_buffer(
        self,
        size: int = -1,
        shuffle: bool = False,
        agent_filter: Sequence[AgentID] = None,
    ) -> Dict[AgentID, Dict]:
        """Return a dict of memory, mapping from agent to memory.

        Returns:
            Dict[AgentID, Memory]: A dict of memory
        """

        res: Dict[AgentID, Memory] = {}
        size = min(self.size, size) if size < 0 else self.size
        agent_filter = agent_filter or list(self._agent_batch.keys())
        if shuffle:
            indices = np.random.choice(self.size, size)
            for agent in agent_filter:
                dict_data = self._agent_batch[agent]
                res[agent] = {k: v[indices] for k, v in dict_data.items()}
        else:
            for agent in agent_filter:
                dict_data = self._agent_batch[agent]
                res[agent] = {k: v[:size] for k, v in dict_data.items()}
        return res

    def add_transition(self, vector_mode: bool = False, **kwargs):
        for k, v in kwargs.items():
            for agent, batch in self._agent_batch.items():
                if not vector_mode:
                    batch[k][self._flag] = copy.deepcopy(v[agent])
                    new_length = 1
                    end = self._flag + new_length
                else:
                    new_length = len(v[agent])
                    if self._flag + new_length > self._capacity:
                        start = 0
                        end = (self._flag + new_length) % self._capacity
                        n_repeat = (self._flag + new_length) // self.capacity
                        if n_repeat > 1:
                            batch[k] = np.concatenate([batch[k][: self._flag], v])[
                                : -self.capacity
                            ]
                        else:
                            batch[k][self._flag :] = copy.deepcopy(
                                v[agent][self._flag - self._capacity :]
                            )
                    else:
                        start = self._flag
                        end = (self._flag + new_length) % self.capacity
                    batch[k][start:end] = copy.deepcopy(v[agent][start - end :])
        self._flag = end
        self._size = min(self._size + new_length, self._capacity)

    def add_batches(
        self, agent_batch: Dict[AgentID, Any], data_preprocessor: type = None
    ):
        """Add a dict of agent batch. Users can specify a data preprocessor to rectify the raw data.

        Args:
            agent_batch (Dict[AgentID, Any]): A dict of agent batch.
            data_preprocessor (type, optional): A callback for data preprocessor. Defaults to None.
        """

        data_preprocessor = data_preprocessor or self._data_preprocessor

        for aid, batch in agent_batch.items():
            agent_buffer = self._agent_batch[aid]
            # FIXME(ming): we should not use keys from external batches
            #   but registered keys
            batch = data_preprocessor(batch)
            insert_length = 0
            for k in agent_buffer.keys():
                v = batch[k]
                insert_length = len(v)
                Log.debug(f"insert data length is {insert_length}")

                if insert_length + self._flag > self._capacity:
                    # Log.warning("Use cliphere ... %s %s %s", insert_length, self._flag, self._capacity)
                    a = (self._flag + insert_length) % self._capacity
                    b = self._flag
                    start, end = min(a, b), max(a, b)
                    v = v[-(self._capacity - end + start) :]
                    agent_buffer[k] = np.concatenate(
                        [agent_buffer[k][start:end], v], axis=0
                    )
                    assert len(agent_buffer[k]) == self._capacity, (
                        len(agent_buffer[k]),
                        self._capacity,
                        k,
                        start,
                        end,
                        self._flag,
                        len(v),
                    )
                else:
                    v = v.squeeze()
                    agent_buffer[k][self._flag : self._flag + insert_length] = v.copy()
        self._flag = (self._flag + insert_length) % self._capacity
        self._size = min(self._size + insert_length, self._capacity)

    def sample(
        self, idxes=None, batch_size=-1, agent_filter=None
    ) -> Dict[AgentID, Dict[str, np.ndarray]]:
        """Sample N batches for trainable agents whose size is batch_size"""

        batch_dict = {}
        agent_filter = agent_filter or self._trainable_agents
        if self._size < batch_size or not self.is_ready():
            Log.warning(
                "No enough data, returns `None`: (size=%s, batch_size=%s)",
                self._size,
                batch_size,
            )
            batch_dict = {aid: None for aid in agent_filter}
            return batch_dict

        if idxes is None and self._use_data_alignment:
            batch_size = self._size if batch_size < 0 else batch_size
            idxes = np.random.choice(self._size, batch_size)

        for agent in agent_filter:
            size = self._size
            batch_size = size if batch_size < 0 else batch_size
            _idxes = idxes if idxes is not None else np.random.choice(size, batch_size)
            batch = {k: v[_idxes] for k, v in self._agent_batch[agent].items()}
            batch_dict[agent] = batch

        return batch_dict

    def reset(self, trainable_agents: List[str]):
        """Reset trainable agents"""
        self._trainable_agents = trainable_agents

    def clean(self):
        self._size = 0
        self._flag = 0

    def save(self, path: str):
        """Accept root directory path"""

        if not osp.exists(path):
            os.makedirs(path)

        with open(osp.join(path, self._pickle_name), "wb") as f:
            pickle.dump(self._agent_batch, f)

    def load(self, path: str):
        """Accept root directory path"""

        with open(osp.join(path, self._pickle_name), "rb") as f:
            self._agent_batch = pickle.load(f)


class HeteroSampler(SamplerInterface):
    def __init__(self, agent_ids, config_lambda_func):
        self._samplers = {
            aid: Sampler([aid], **config_lambda_func(aid)) for aid in agent_ids
        }
        self._trainable_agents = agent_ids

    @property
    def size(self) -> Dict[AgentID, int]:
        return min([v.size for v in self._samplers.values()])

    @property
    def capacity(self) -> int:
        return {aid: s.capacity for aid, s in self._samplers.items()}

    @property
    def trainable_agents(self):
        return self._trainable_agents

    def is_ready(self):
        return all([v.is_ready() for v in self._samplers.values()])

    def add_transition(self, vector_mode=False, **kwargs):
        _ = [
            sampler.add_transition(vector_mode=vector_mode, **kwargs)
            for sampler in self._samplers.values()
        ]

    def add_batches(
        self, agent_batch: Dict[AgentID, Any], data_preprocessor: type = None
    ):
        """Merge batches by agents"""

        for aid, batch in agent_batch.items():
            self._samplers[aid].add_batches({aid: batch}, data_preprocessor)

    def is_ready(self):
        return all([sampler.is_ready() for sampler in self._samplers.values()])

    def sample(
        self, idxes=None, batch_size=-1, agent_filter=None
    ) -> Dict[AgentID, Dict[str, np.ndarray]]:
        """Sample N batches for trainable agents whose size is batch_size"""

        batch_dict = {}
        agent_filter = agent_filter or self._trainable_agents
        for agent in agent_filter:
            batch_dict.update(self._samplers[agent].sample(idxes, batch_size, [agent]))
        return batch_dict

    def reset(self, trainable_agents: List[str]):
        """Reset trainable agents"""
        self._trainable_agents = trainable_agents

    def clean(self):
        _ = [s.clean() for s in self._samplers.values()]

    def save(self, path: str):
        """Accept root directory path"""

        _ = [s.save(path) for s in self._samplers.values()]

    def load(self, path: str):
        """Accept root directory path"""

        _ = [s.load(path) for s in self._samplers.values()]


def get_sampler(
    keys: Sequence[AgentID], sampler_config: Union[Dict, LambdaType]
) -> SamplerInterface:
    """Return a sampler determined by homogeneous or heterogenous agent settings.

    Args:
        keys (Sequence[AgentID]): Agent keys.
        sampler_config (Union[Dict, LambdaType]): Sampler config description.

    Returns:
        SamplerInterface: A sampler instance.
    """

    assert isinstance(keys, (list, tuple)), keys
    if isinstance(sampler_config, LambdaType):
        if len(keys) == 1:
            sampler = Sampler(keys, **sampler_config(keys[0]))
        else:
            sampler = HeteroSampler(keys, sampler_config)
    else:
        sampler = Sampler(keys, **sampler_config)

    return sampler
