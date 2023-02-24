# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import logging
import os
import os.path as osp
import pprint
import subprocess
import sys
import time
from termcolor import colored

from multiprocessing import Process, Manager, freeze_support
from datetime import datetime as date
from loguru import logger

from glob import glob

import logging
import torch.cuda
import argparse
import cv2
import os
import os.path as osp
import sys
from termcolor import colored
import time

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from edgeyolo.detect import Detector, TRTDetector, draw
from common.util import setup_log, d_print

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("EdgeYOLO Detect parser")
    parser.add_argument("-w", "--weights", type=str, default="models/edgeyolo_coco.pth", help="weight file")
    parser.add_argument("-c", "--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("-n", "--nms-thres", type=float, default=0.55, help="nms threshold")
    parser.add_argument("--mp", action="store_true", help="use multi-process to accelerate total speed")
    parser.add_argument("--fp16", action="store_true", help="fp16")
    parser.add_argument("--no-fuse", action="store_true", help="do not fuse model")
    parser.add_argument("--input-size", type=int, nargs="+", default=[640, 640], help="input size: [height, width]")
    parser.add_argument("-s", "--source", type=str, default="temp/videos/W6.1chan3.avi", help="video source or image dir")
    parser.add_argument("--trt", action="store_true", help="is trt model")
    parser.add_argument("--legacy", action="store_true", help="if img /= 255 while training, add this command.")
    parser.add_argument("--use-decoder", action="store_true", help="support original yolox model v0.2.0")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--no-label", action="store_true", help="do not draw label")
    parser.add_argument("--save-dir", type=str, default="./output/detect/imgs/", help="image result save dir")
    parser.add_argument("--save-all", action="store_true", help="save all images")
    parser.add_argument("--fps", type=int, default=99999, help="max fps")
    args = parser.parse_args()
    for arg in vars(args):
        d_print(f'  - {arg:20s}: {getattr(args, arg)}')
    return args


def detect(args):
    logger.info(f'detect( {args} )')
    exist_save_dir = os.path.isdir(args.save_dir)

    # detector setup
    detector = TRTDetector if args.trt else Detector
    detect = detector(
        weight_file=args.weights,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        input_size=args.input_size,
        fuse=not args.no_fuse,
        fp16=args.fp16,
        use_decoder=args.use_decoder
    )
    if args.trt:
        args.batch = detect.batch_size

    # source loader setup
    if os.path.isdir(args.source):
        class DirCapture:
            def __init__(self, dir_name):
                self.imgs = []
                for img_type in ["jpg", "png", "jpeg", "bmp", "webp"]:
                    self.imgs += sorted(glob(os.path.join(dir_name, f"*.{img_type}")))
            def isOpened(self):
                return bool(len(self.imgs))
            def read(self):
                print(self.imgs[0])
                now_img = cv2.imread(self.imgs[0])
                self.imgs = self.imgs[1:]
                return now_img is not None, now_img
        source = DirCapture(args.source)
        delay = 0
    else:
        source = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        delay = 1

    all_dt = []
    dts_len = 300 // args.batch
    success = True

    win_w = 2880 // 4
    win_h = 1860 // 4

    name_win = f'OD: {osp.basename(args.source)}'
    cv2.namedWindow(name_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_win, win_w, win_h)
    cv2.moveWindow(name_win, 20, 20)

    # start inference
    count = 0
    t_start = time.time()
    while source.isOpened() and success:
        frames = []
        for _ in range(args.batch):
            success, frame = source.read()
            if not success:
                if not len(frames):
                    cv2.destroyAllWindows()
                    break
                else:
                    while len(frames) < args.batch:
                        frames.append(frames[-1])
            else:
                frames.append(frame)

        if not len(frames):
            break

        results = detect(frames, args.legacy)
        dt = detect.dt
        all_dt.append(dt)
        if len(all_dt) > dts_len:
            all_dt = all_dt[-dts_len:]
        print(f"\r{dt * 1000 / args.batch:.1f}ms  "
              f"average:{sum(all_dt) / len(all_dt) / args.batch * 1000:.1f}ms", end="      ")

        key = -1
        imgs = draw(frames, results, detect.class_names, 2, draw_label=not args.no_label)

        for img in imgs:
            cv2.imshow(name_win, img)
            count += 1

            key = cv2.waitKey(delay)
            if key in [ord("q"), 27]:
                break
            elif key == ord(" "):
                delay = 1 - delay
            elif key == ord("s") or args.save_all:
                if not exist_save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    exist_save_dir = True
                file_name = f"{str(date.now()).split('.')[0].replace(':', '').replace('-', '').replace(' ', '')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, file_name), img)
                logger.info(f"image saved to {file_name}.")
        if key in [ord("q"), 27]:
            cv2.destroyAllWindows()
            break

    logger.info(f"\ntotal frame: {count}, total average latency: {(time.time() - t_start) * 1000 / count - 1}ms")


if __name__ == "__main__":
    args = parse_args()
    time_beg = time.time()
    this_filename = osp.basename(__file__)
    setup_log(this_filename)

    detect(args)

    time_end = time.time()
    logger.warning(f'{this_filename} elapsed {time_end - time_beg} seconds')
    print(colored(f'{this_filename} elapsed {time_end - time_beg} seconds', 'yellow'))
