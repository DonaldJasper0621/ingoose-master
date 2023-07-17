from hydra import utils


def build_optimizer_and_scheduler_configuration(
    params, optimizer_conf, scheduler_configuration_conf
):
    optimizer_conf = dict(optimizer_conf)
    optimizer_conf.setdefault("params", params)
    optimizer_target = optimizer_conf.pop("_target_")
    optimizer = utils.get_method(optimizer_target)(**optimizer_conf)

    scheduler_configuration = dict(scheduler_configuration_conf)
    scheduler_conf = dict(scheduler_configuration["scheduler"])
    scheduler_conf.setdefault("optimizer", optimizer)
    scheduler_target = scheduler_conf.pop("_target_")
    scheduler = utils.get_method(scheduler_target)(**scheduler_conf)
    scheduler_configuration["scheduler"] = scheduler

    return optimizer, scheduler_configuration
