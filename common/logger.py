"""
Logging helpers.

@status: passed
@author: Ming Zhou
@organization: SJTU-MARL
@date: 2021/08/15
"""

from typing import Dict, Sequence, Union
from argparse import Namespace

import time
import logging
import os

from colorlog import ColoredFormatter

from torch.utils import tensorboard


settings = Namespace(LOG_LEVEL=logging.INFO)


Log = logging.getLogger("expground")
Log.setLevel(settings.LOG_LEVEL)
Log.handlers = []  # No duplicated handlers
Log.propagate = False  # workaround for duplicated logs in ipython
log_level = settings.LOG_LEVEL

stream_handler = logging.StreamHandler()
stream_handler.setLevel(log_level)

stream_formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s][%(process)05d][%(levelname)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "white,bold",
        "INFOV": "cyan,bold",
        "WARNING": "yellow",
        "ERROR": "red,bold",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)
stream_handler.setFormatter(stream_formatter)
Log.addHandler(stream_handler)


def init_file_logger(experiment_dir_: str):
    """Initialize a file logger with given experiment directory.

    Args:
        experiment_dir_ (str): Local experiment logging directory.
    """

    file_handler = logging.FileHandler(os.path.join(experiment_dir_, "sf_log.txt"))
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        fmt="[%(asctime)s][%(process)05d] %(message)s", datefmt=None, style="%"
    )
    file_handler.setFormatter(file_formatter)
    Log.addHandler(file_handler)


def monitor(enable_timer: bool = False, enable_returns: bool = True, prefix: str = ""):
    """Monitor will record the time consumption and print return.

    Args:
        enable_timer (bool, optional): Enable timer or not. Defaults to False.
        enable_returns (bool, optional): Enable return print. Defaults to True.
        prefix (str, optional): Prefix as hint. Defaults to "".
    """

    def decorator(func):
        def wrap(*args, **kwargs):
            Log.debug(f"entering func: {func.__name__}")
            # _________ enter _________
            mess = prefix
            start = time.time()
            rets = func(*args, **kwargs)
            end = time.time()
            if enable_timer:
                mess += f"time consumption = {end - start}s "
            if enable_returns:
                mess += str(rets)
            # _________ exit __________
            Log.debug(mess)
            return rets

        return wrap

    return decorator


def write_to_tensorboard(
    writer: tensorboard.SummaryWriter,
    info: Dict,
    global_step: Union[int, Dict],
    prefix: str,
):
    """Write learning info to tensorboard.

    Args:
        writer (tensorboard.SummaryWriter): The summary writer instance.
        info (Dict): The information dict.
        global_step (int): The global step indicator.
        prefix (str): Prefix added to keys in the info dict.
    """
    if writer is None:
        return

    dict_step = isinstance(global_step, Dict)

    prefix = f"{prefix}/" if len(prefix) > 0 else ""
    for k, v in info.items():
        if isinstance(v, dict):
            # add k to prefix
            write_to_tensorboard(
                writer,
                v,
                global_step if not dict_step else global_step[k],
                f"{prefix}{k}",
            )
        elif isinstance(v, Sequence):
            raise NotImplementedError(
                f"Sequence value cannot be logged currently: {v}."
            )
        elif v is None:
            continue
        else:
            writer.add_scalar(
                f"{prefix}{k}",
                v,
                global_step=global_step
                if not dict_step
                else global_step["global_step"],
            )
