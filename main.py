from detector import Detector
from argparse import ArgumentParser
from TrafficLight import TrafficLight
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
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('./output.mp4',
                    cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))


    timer = []
    for i in range(4):
        timer.append(TrafficLight())

    colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]

    masks = []
    for i in range(1, 5):
        masks.append(cv2.resize(cv2.cvtColor(cv2.imread(f'mask{i}.png'), cv2.COLOR_BGR2GRAY), (1920, 1072)))

    intersected_objects = []
    prev_frame = np.zeros((1, 1, 1), dtype="uint8")
    car_detector = Detector(args.model_path)
    ret, frame = cap.read()
    count = 0
    while ret:
        count += 1
        if count % 25 != 0:
            continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bboxes, _, cls_names = detect_objects(frame, car_detector)   
        for i, mask in enumerate(masks):
            if timer[i].light == 'green':
                traffic_color = (0, 255, 0)
            else:
                traffic_color = (0, 0, 255)
                
            intersected_objs = find_intersected(mask, bboxes, cls_names)
            if len(intersected_objs) >= 3 and timer[i].light == 'green':
                timer[i].change_timer('slower')
            elif len(intersected_objs) >= 3 and timer[i].light == 'red':
                timer[i].change_timer('faster')
            else:
                timer[i].change_timer('normal')
            # print(f'Lane {i+1}', len(intersected_objs))
            # print(timer[i].timer)
            for j, objt in enumerate(intersected_objs):
                x, y, w, h = objt['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), traffic_color, 2)
                cv2.putText(frame, f'{objt["cls"]}_{j+1}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, colors[i], 2)
        out.write(frame)
        cv2.imshow("SmartTrafficLight", frame)
        cv2.waitKey(1)
                
        prev_frame = gray_frame
        ret, frame = cap.read()
    out.release()
    return intersected_objects


if __name__ == '__main__':
    main()
