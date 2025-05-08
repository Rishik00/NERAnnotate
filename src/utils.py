## Actually useful imports
from typing import Dict, List
import yaml
import torch
import logging

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_section_from_yaml( file_name: str, section: str, title: str) -> Dict:
    with open(file_name, 'r') as f:
        config_data = yaml.safe_load(f)
    config = config_data.get(section, {})

    validate_config(config=config,)
    print("Loading config: ", config)
    return config

def validate_config(config: Dict) -> None:
    missing_fields = [field for field in config.keys() if config.get(field) is None]
    if missing_fields:
        raise ValueError(f"Missing required config fields: {', '.join(missing_fields)}")

def load_config(file_name: str):
    return load_section_from_yaml(file_name, 'overall_config', 'Overall Config')

def load_translator_config(file_name: str):
    return load_section_from_yaml(file_name, 'translator_config', 'Translator Config')

def load_aligner_config(file_name: str):
    return load_section_from_yaml(file_name, 'aligner_config', 'Aligner Config')

def setup_logging(log_file: str, log_level: int = logging.INFO):
    """
    Sets up logging with a console handler and a file handler.
    
    Args:
        log_file (str): The path to the log file.
        log_level (int): The logging level (default is logging.INFO).
    """
    # Create a logger object
    if 'log' not in log_file.split('.')[1]:
        log_file = log_file.split('.')[0]  + '.log'

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(
        filename=log_file,
        mode='a',
        encoding='utf-8',
    )

    # Create a formatter and set it for the handlers
    formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)

    return logger