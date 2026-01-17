from pathlib import Path
from typing import cast

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.paths import configs_dir


def get_dir_files_count(data_dir: Path) -> int:
    return len(list(data_dir.rglob("*")))


def get_config(config_name: str) -> DictConfig:
    logger.info(f"Loading {config_name} config...")
    return cast(DictConfig, OmegaConf.load(configs_dir / config_name))


def prepare_feature_info() -> tuple[list[int], list[dict]]:
    feature_config = get_config("features.yaml")

    features_info = feature_config.get("features")
    if not features_info:
        msg = "Missing or empty 'features' section in features.yaml"
        logger.error(msg)
        raise ValueError(msg)

    features_to_exclude = feature_config.get("excludes", [])

    if not features_to_exclude:
        return [], features_info

    logger.info(f"Excluding features ({len(features_to_exclude)}): {features_to_exclude}")
    features_to_exclude_lwr = {name.lower() for name in features_to_exclude}
    feature_names_lwr = [f["name"].lower() for f in features_info]

    invalid_names = features_to_exclude_lwr - set(feature_names_lwr)
    if invalid_names:
        logger.warning(f"Features not found for exclusion: {sorted(invalid_names)}")

    indices_to_exclude = [i for i, name in enumerate(feature_names_lwr) if name in features_to_exclude_lwr]

    if not indices_to_exclude:
        logger.info("No matching features found to exclude.")
        return [], features_info

    filtered_features_info = [f for i, f in enumerate(features_info) if i not in indices_to_exclude]

    logger.info(f"Remaining features ({len(filtered_features_info)}): {[f['name'] for f in filtered_features_info]}")

    return indices_to_exclude, filtered_features_info
