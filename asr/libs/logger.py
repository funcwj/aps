#!/usr/bin/env python

# wujian@2019

import logging

def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    """
    Get logger instance
    """

    def get_handler(handler):
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
        handler.setFormatter(formatter)
        return handler

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if file:
        # both stdout & file
        logger.addHandler(get_handler(logging.FileHandler(name)))
        # logger.addHandler(logging.StreamHandler())
    else:
        logger.addHandler(logging.StreamHandler())
    return logger


