import argparse
import sys

import warnings
warnings.filterwarnings("ignore")

import cv2

from detections import draw, counter
from obj_detector import YOLOv7

sys.path.insert(0, './yolov7')


def track_single_frame(weights, classes, device, image_file, output):
    yolov7 = YOLOv7()
    yolov7.load(weights_path=weights, classes=classes, device=device)

    image = cv2.imread(image_file)
    detections = yolov7.detect(image, track=True)
    detected_frame = draw(image, detections)
    detected_frame = counter(detected_frame, detections)

    num_people = 0
    for detected_obj in detections:
        if detected_obj['class'] == 'person':
            num_people += 1

    print(f'Number of People in Frame: {num_people}')

    cv2.imwrite(output, detected_frame)

    print(detections)
    yolov7.unload()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track_single_frame.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--classes', type=str, default='track_classes.yaml',
                        help='YAML File of Object(s) to Track (See '
                             'Documentation)')
    parser.add_argument('--device', type=str, default='cpu', help='CPU or GPU')
    parser.add_argument('--image_file', type=str, default='test_frame.png', help='Image File Path')
    parser.add_argument('--output', type=str, default='detected_test_frame.png', help='(output_image_name).png')
    opt = parser.parse_args()
    print(opt)

    track_single_frame(opt.weights, opt.classes, opt.device, opt.image_file, opt.output)
