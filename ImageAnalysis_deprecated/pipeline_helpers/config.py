# pipeline_helpers/config.py

from pathlib import Path
import yaml


def load_config(config_path: Path) -> dict:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : Path
        Path to YAML configuration file

    Returns
    -------
    dict
        Parsed configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: dict) -> None:
    """
    Perform minimal validation of configuration content.
    Raises ValueError if invalid.
    """
    # Required top-level keys
    required = {"image", "output", "tps", "parameters_01"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    # Blur validation
    blur = config["parameters_01"]["Blur"]
    if blur["type"] not in blur["valid"]:
        raise ValueError(
            f"Invalid blur type '{blur['type']}'. "
            f"Valid options: {blur['valid']}"
        )

    if blur["type"] in {"Gaussian", "Median"} and blur["ksize"] % 2 == 0:
        raise ValueError(
            f"{blur['type']} blur requires odd ksize, got {blur['ksize']}"
        )
