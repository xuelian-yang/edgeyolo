# -*- coding: utf-8 -*-

import datetime
import logging
import os
import os.path as osp
import PIL
from termcolor import colored


def d_print(text):
    print(colored(text, 'cyan'))


def setup_log(filename):
    medium_format = (
        '[%(asctime)s] %(levelname)s : %(filename)s[%(lineno)d] %(funcName)s'
        ' >>> %(message)s'
    )
    if not filename.lower().endswith('.log'):
        filename = filename + '.log'
    log_dir = osp.abspath(osp.join(osp.dirname(__file__), '../logs'))
    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    dt_now = datetime.datetime.now()
    get_log_file = osp.join(log_dir, filename)
    logging.basicConfig(
        filename=get_log_file,
        filemode='w',
        level=logging.INFO,
        format=medium_format
    )
    logging.info('@{} created at {}'.format(get_log_file, dt_now))
    print(colored('@{} created at {}'.format(get_log_file, dt_now), 'magenta'))


def create_gif(images, gif):
    frames = []
    for item in images:
        new_frame = PIL.Image.open(item)
        frames.append(new_frame)
    frames[0].save(gif, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0, comment=b'DAIR V2X Visualization')
