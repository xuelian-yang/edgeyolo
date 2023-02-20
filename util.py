# -*- coding: utf-8 -*-

import datetime
import logging
import os.path as osp
from termcolor import colored

def setup_log(call_file):
    medium_format = (
        '[%(asctime)s] %(levelname)s : %(filename)s[%(lineno)d] %(funcName)s'
        ' >>> %(message)s'
    )

    dt_now = datetime.datetime.now()
    log_name = osp.basename(call_file).replace('.py', '.log')
    get_log_file = osp.abspath(osp.join(osp.dirname(call_file), log_name))
    logging.basicConfig(
        filename=get_log_file,
        filemode='w',
        level=logging.INFO,
        format=medium_format
    )
    logging.info('@{} created at {}'.format(get_log_file, dt_now))
    print(colored('@{} created at {}'.format(get_log_file, dt_now), 'magenta'))
