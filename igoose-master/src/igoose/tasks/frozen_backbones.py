from typing import Any, Mapping

from hydra import utils as hydra_utils

from igoose.tasks import bases


class FrozenBackboneTask(bases.TaskBase):
    def __init__(self, backbone_conf: Mapping[str, Any]):
        super().__init__({}, {})

        self.save_hyperparameters()

        self.backbone = hydra_utils.instantiate(backbone_conf)

        self.eval()
        self.requires_grad_(requires_grad=False)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return self.backbone(batch)

    def configure_optimizers(self):
        return None
