#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os.path as osp
import time
import sys
import logging

import torch.distributed as dist


def setup_logger(logpth, logfile_name):
    logfile = '{}-{}.log'.format(logfile_name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.CRITICAL
    if dist.is_initialized() and dist.get_rank()!=0:
        log_level = logging.WARNING
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())