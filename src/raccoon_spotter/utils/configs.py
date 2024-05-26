from kedro.config import MissingConfigException, OmegaConfigLoader

loader = OmegaConfigLoader("conf")


try:
    configs = OmegaConfigLoader(
        "conf/local", config_patterns={"configs": ["**/configs.yml"]}
    )["configs"]
except MissingConfigException:
    configs = OmegaConfigLoader(
        "conf/base", config_patterns={"configs": ["**/configs.yml"]}
    )["configs"]
