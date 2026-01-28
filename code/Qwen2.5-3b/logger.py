import logging
import torch.nn as nn

def get_logger(filename, verbosity=0, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setLevel(level_dict[verbosity]) 
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger