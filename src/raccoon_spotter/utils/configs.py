from kedro.config import OmegaConfigLoader

configs = OmegaConfigLoader("conf", config_patterns={"configs": ["**/configs.yml"]})[
    "configs"
]
