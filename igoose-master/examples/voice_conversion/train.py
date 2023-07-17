import hydra
from hydra import utils
import omegaconf
import pytorch_lightning as pl


@hydra.main(config_path=None, config_name="train", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    if cfg.get("seed") is not None:
        pl.seed_everything(seed=cfg.seed, workers=True)

    datamodule = utils.instantiate(cfg.data_module_conf)
    task = utils.instantiate(cfg.task_conf)
    trainer = utils.instantiate(cfg.trainer_conf)
    trainer.fit(task, datamodule=datamodule)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
