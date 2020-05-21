from detector import Detector
from argparse import ArgumentParser
# from tracker import Tracker
from helpers import *


def arguments():
    parser = ArgumentParser()
    parser.add_argument('-vp', '--video_path', help='Traffic video path.')
    parser.add_argument('-mdp', '--model_path', help='Yolo model path.')
    return parser


def main():
    args = arguments().parse_args()
    cap = cv2.VideoCapture(args.video_path)
    timers = []
    for i in range(4):
        timers.append(TrafficLight())


    masks = []
    for i in range(1, 5):
        masks.append(cv2.resize(cv2.cvtColor(cv2.imread(f'mask{i}.png'), cv2.COLOR_BGR2GRAY), (1920, 1072)))

    intersected_objects = []
    prev_frame = np.zeros((1, 1, 1), dtype="uint8")
    car_detector = Detector(args.model_path)
    ret, frame = cap.read()

    while ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("", frame)
        cv2.waitKey(1)
        for i, mask in enumerate(masks):
            if intersect_line(prev_frame, gray_frame, mask):
                bboxes, _, cls_names = detect_objects(frame, car_detector)
                intersected_objs = find_intersected(mask, bboxes, cls_names)
                print(f'Lane {i+1}', len(intersected_objs))
        prev_frame = gray_frame
        ret, frame = cap.read()
    return intersected_objects


if __name__ == '__main__':
    main()
