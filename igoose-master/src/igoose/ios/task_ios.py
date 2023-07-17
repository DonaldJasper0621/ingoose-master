import importlib

import torch

from igoose.tasks import bases


def load_task(checkpoint_path, strict: bool = True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    module = importlib.import_module(checkpoint[bases.CHECKPOINT_KEY_TASK_MODULE])
    cls = getattr(module, checkpoint[bases.CHECKPOINT_KEY_TASK_NAME])
    return cls.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu", strict=strict
    )
