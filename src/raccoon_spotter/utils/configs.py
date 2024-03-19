from kedro.config import OmegaConfigLoader

loader = OmegaConfigLoader("conf")

configs = OmegaConfigLoader("conf", config_patterns={"configs": ["**/configs.yml"]})[
    "configs"
]
