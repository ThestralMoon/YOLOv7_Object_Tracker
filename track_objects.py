import argparse
import logging
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

import warnings

warnings.filterwarnings("ignore")

import cv2
from tqdm import tqdm

from detections import draw, counter
from obj_detector import YOLOv7

sys.path.insert(0, './yolov7')


def track_objects(weights, classes, device, video, output):
    yolov7 = YOLOv7()
    yolov7.load(weights_path=weights, classes=classes, device=device)

    vid = cv2.VideoCapture(video)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frames_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    output = cv2.VideoWriter(output, fourcc, fps, (width, height))

    if not vid.isOpened():
        logging.error('Error Opening Video!')

    logging.info('Tracking Objects...\n')
    pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)

    try:
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                detections = yolov7.detect(frame, track=True)
                detected_frame = draw(frame, detections)
                detected_frame = counter(detected_frame, detections)
                output.write(detected_frame)
                pbar.update(1)
            else:
                break
    except KeyboardInterrupt:
        pass

    pbar.close()
    vid.release()
    output.release()
    yolov7.unload()

    logging.info(f'Completed Tracking Objects for {video}')


WEIGHTS_PATH = os.path.join(os.getcwd(), 'yolov7/yolov7.pt')
CLASSES_PATH = os.path.join(os.getcwd(), 'track_classes.yaml')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track_objects.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--classes', type=str, default='track_classes.yaml',
                        help='YAML File of Object(s) to Track (See '
                             'Documentation)')
    parser.add_argument('--device', type=str, default='cpu', help='CPU or GPU')
    parser.add_argument('--video', type=str, help='Video Path')
    parser.add_argument('--output', type=str, help='(output_video_name).mp4')
    opt = parser.parse_args()
    print(opt)

    track_objects(opt.weights, opt.classes, opt.device, opt.video, opt.output)
