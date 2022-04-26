from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Sequence, Union

from common.logger import Log
from common.sampler import SamplerInterface
from common.data import Memory, DataLoader, MultiAgentMemory
from common.base_policy import Policy


AgentID = str
DEFAULT_TRAINING_CONFIG = {"batch_size": 32}


class Trainer(metaclass=ABCMeta):
    def __init__(
        self,
        learning_mode: str,
        training_config: Dict[str, Any] = None,
        policy_instance: Policy = None,
    ):
        """Initialize a trainer for a type of policies.

        Args:
            learning_mode (str): Learning mode inidication, could be `off_policy` or `on_policy`.
            training_config (Dict[str, Any], optional): The training configuration. Defaults to None.
            policy_instance (Policy, optional): A policy instance, if None, we must reset it. Defaults to None.
        """

        self._policy = policy_instance

        if training_config is None:
            Log.warning(
                "No training config specified, will use default={}".format(
                    DEFAULT_TRAINING_CONFIG
                )
            )
            self._training_config = DEFAULT_TRAINING_CONFIG
        else:
            self._training_config = training_config

        self._step_counter = 0
        self._learning_mode = learning_mode

        assert self._learning_mode in [
            "off_policy",
            "on_policy",
        ], "You must specify the learning mode as `off_policy` or `on_policy`, while the accepted parameter is: {}".format(
            self._learning_mode
        )

        if policy_instance is not None:
            self.setup()

    @property
    def policy(self):
        return self._policy

    @property
    def training_config(self) -> Dict[str, Any]:
        return self._training_config

    @property
    def counter(self):
        return self._step_counter

    @abstractmethod
    def setup(self):
        """Set up optimizers here."""

    @abstractmethod
    def train(self, batch) -> Dict[str, float]:
        """Run training, and return info dict.

        Args:
            batch (Dict): A dict of batch

        Returns:
            Dict[str, float]: A dict of information
        """

    def on_policy_train(
        self, buffer, sampler, shuffle, agent_filter, multi_agent: bool = False
    ):
        if sampler is not None:
            if sampler.size < self._training_config["batch_size"]:
                batch_size = self._training_config["batch_size"]
                Log.warning(
                    f"No enough training data. size={sampler.size} batch_size={batch_size}"
                )
                return {}
            else:
                dataloader = DataLoader(
                    MultiAgentMemory.from_dict(
                        sampler.get_buffer(
                            size=-1, shuffle=shuffle, agent_filter=agent_filter
                        )
                    ),
                    shuffle=shuffle,
                    batch_size=self.training_config["batch_size"],
                )
        else:
            dataloader = DataLoader(
                MultiAgentMemory.from_dict(buffer)
                if multi_agent
                else Memory.from_dict(buffer),
                shuffle=shuffle,
                batch_size=self.training_config["batch_size"],
            )

        feedback = self.train(dataloader)
        return feedback

    def __call__(
        self,
        time_step: int,
        buffer: Union[Dict[AgentID, Memory], Dict[AgentID, Dict]] = None,
        sampler: SamplerInterface = None,
        agent_filter: Sequence = None,
    ) -> Dict[str, Any]:
        """Implement the training Logic here, and return the computed loss results.

        Args:
            time_step (int): The training time step.
            buffer (Union[Dict[AgentID, Memory], Dict[AgentID, Dict]], Optional): The give data buffer
            agent_filter (Sequence[AgentID], Optional): Determine which agents are governed by \
                this trainer. In single agent mode, there will be only one agents be \
                    transferred. Activated only when `sampler` is not None.
            sampler: (Sampler, Optional): A sampler instance for sampling. Defaults to None.

        Returns:
            Dict: A dict of training feedback. Could be agent to dict or string to any scalar/vector datas.
        """

        # assert loss func has been initialized
        self._step_counter = time_step
        # assert self.loss_func is not None
        # batch sampling
        if self._learning_mode == "on_policy":
            feedback = self.on_policy_train(buffer, sampler, False, agent_filter)
        else:
            if buffer is None:
                buffer = sampler.sample(
                    batch_size=self._training_config["batch_size"],
                    agent_filter=agent_filter,
                )
            feedback = self.train(buffer)

        return feedback

    def reset(self, policy_instance=None, configs=None, learning_mode: str = None):
        """Reset current trainer, with given policy instance, training configuration or learning mode.

        Note:
            Becareful to reset the learning mode, since it will change the sample behavior. Specifically, \
                the `on_policy` mode will sample datas sequntially, which will return a `torch.DataLoader` \
                    to the method `self.train`. For the `off_policy` case, the sampler will sample data \
                        randomly, which will return a `dict` to 

        Args:
            policy_instance (Policy, optional): A policy instance. Defaults to None.
            configs (Dict[str, Any], optional): A training configuration used to update existing one. Defaults to None.
            learning_mode (str, optional): Learning mode, could be `off_policy` or `on_policy`. Defaults to None.
        """

        self._step_counter = 0
        if policy_instance is not self._policy:
            self._policy = policy_instance or self._policy
            self.setup()

        if configs is not None:
            self.training_config.update(configs)

        if learning_mode in ["off_policy", "on_policy"]:
            self._learning_mode = learning_mode
