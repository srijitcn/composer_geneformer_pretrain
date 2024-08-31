from omegaconf import DictConfig

def load_params(
    yaml_path = 'parameters.yaml',
    args_list = None):
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)

    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    return cfg
