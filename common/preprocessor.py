"""
Gym space preprocessor.

@status: passed
@author: Ming Zhou
@organization: SJTU-MARL
@date: 2021/08/15
"""

from abc import ABCMeta, abstractmethod
from typing import Sequence, Tuple, List
from gym import spaces
from functools import reduce

import operator
import numpy as np

from common.logger import Log


class Preprocessor(metaclass=ABCMeta):
    def __init__(self, space: spaces.Space):
        self._original_space = space

    @abstractmethod
    def transform(self, data) -> np.ndarray:
        pass

    @abstractmethod
    def write(self, array, offset, data):
        pass

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError


class NoPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Box):
        super(NoPreprocessor, self).__init__(space)
        self._size = reduce(operator.mul, space.shape)

    @property
    def shape(self):
        return self._original_space.shape

    @property
    def size(self):
        return self._size

    def transform(self, data) -> np.ndarray:
        return data

    def write(self, array, offset, data):
        pass


class DictFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Dict):
        super(DictFlattenPreprocessor, self).__init__(space)
        self._preprocessors = {}

        for k, _space in space.spaces.items():
            self._preprocessors[k] = get_preprocessor(_space)(_space)

        self._size = sum([prep.size for prep in self._preprocessors.values()])

    @property
    def shape(self):
        return (self._size,)

    @property
    def size(self):
        return self._size

    def transform(self, data) -> np.ndarray:
        """Transform support multi-instance input"""
        if not isinstance(data, Sequence):
            array = np.zeros(self.shape)
            self.write(array, 0, data)
        else:
            array = np.zeros((len(data),) + self.shape)
            for i in range(len(array)):
                self.write(array[i], 0, data[i])
        return array

    def write(self, array, offset, data):
        if isinstance(data, dict):
            for k, _data in data.items():
                size = self._preprocessors[k].size
                array[offset : offset + size] = self._preprocessors[k].transform(_data)
                offset += size
        else:
            raise TypeError(f"Unexpected type: {type(data)}")


class TupleFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Tuple):
        super(TupleFlattenPreprocessor, self).__init__(space)
        self._preprocessors = []
        for k, _space in space.spaces:
            self._preprocessors.append(get_preprocessor(_space)(_space))
        self._size = sum([prep.size for prep in self._preprocessors])

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    def transform(self, data) -> np.ndarray:
        if isinstance(data, List):
            array = np.zeros((len(data),) + self.shape)
            for i in range(len(array)):
                self.write(array[i], 0, data[i])
        else:
            array = np.zeros(self.shape)
            self.write(array, 0, data)
        return array

    def write(self, array, offset, data):
        if isinstance(data, Tuple):
            for _data, prep in zip(data, self._preprocessors):
                array[offset : offset + prep.size] = prep.transform(_data)
        else:
            raise TypeError(f"Unexpected type: {type(data)}")


class BoxFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Box):
        super(BoxFlattenPreprocessor, self).__init__(space)
        self._size = reduce(operator.mul, space.shape)

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    def transform(self, data) -> np.ndarray:
        # if isinstance(data, list):
        #     array = np.stack(data)
        #     array = array.reshape((len(array), -1))
        #     return array
        # else:
        array = np.asarray(data).reshape((-1,))
        return array

    def write(self, array, offset, data):
        pass


class DiscreteFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Discrete):
        super(DiscreteFlattenPreprocessor, self).__init__(space)
        self._size = space.n

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    def transform(self, data) -> np.ndarray:
        # convert to one hot
        array = np.zeros(self.size, dtype=np.float)
        data = int(data)
        array[data] = 1.0
        return array

    def write(self, array, offset, data):
        pass


class Mode:
    FLATTEN = "flatten"
    STACK = "stack"


def get_preprocessor(space: spaces.Space, mode: str = Mode.FLATTEN):
    if mode == Mode.FLATTEN:
        if isinstance(space, spaces.Dict):
            # Log.debug("Use DictFlattenPreprocessor")
            return DictFlattenPreprocessor
        elif isinstance(space, spaces.Tuple):
            # Log.debug("Use TupleFlattenPreprocessor")
            return TupleFlattenPreprocessor
        elif isinstance(space, spaces.Box):
            # Log.debug("Use BoxFlattenPreprocessor")
            return BoxFlattenPreprocessor
        elif isinstance(space, spaces.Discrete):
            return DiscreteFlattenPreprocessor
        else:
            raise TypeError(f"Unexpected space type: {type(space)}")
    elif mode == Mode.STACK:
        if isinstance(space, spaces.Box):
            return NoPreprocessor
    else:
        raise ValueError(f"Unexpected mode: {mode}")
