# encoding=utf8
import logging
import os

FORMAT = '[MAIN] %(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'
DATEFMT = '%d-%m-%Y:%H:%M:%S'

def initialize_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt=FORMAT, datefmt=DATEFMT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(output_dir, "error.log"), "a", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter(fmt=FORMAT, datefmt=DATEFMT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(output_dir, "all.log"), "a")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt=FORMAT, datefmt=DATEFMT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
