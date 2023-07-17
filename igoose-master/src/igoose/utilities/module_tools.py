from torch import nn


def set_submodule(module: nn.Module, target: str, submodule: nn.Module):
    if not target:
        raise ValueError("Empty target.")

    parent_target, _, name = target.rpartition(".")
    setattr(module.get_submodule(parent_target), name, submodule)
