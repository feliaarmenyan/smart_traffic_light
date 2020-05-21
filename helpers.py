import cv2
import numpy as np

from datetime import datetime


def gamma_correct(frame):
    cv2.imshow("frame", frame)
    frame = np.float32(frame) / 255.0
    gamma = -0.3 / np.log(cv2.sumElems(frame)[0] / (frame.shape[0] * frame.shape[1]))
    frame = np.uint8(cv2.pow(frame, gamma) * 255)
    return frame


def detect_objects(frame, car_detector):
    return car_detector.predict(frame)


def intersect_line(prev_frame, frame, mask):
    if prev_frame.shape[0] == 1:
        return True
    morpho = cv2.erode(cv2.subtract(prev_frame, frame, mask=mask.astype(np.int8)) * 255, np.ones((20, 20)))
    return cv2.countNonZero(morpho) != 0


def init_traffic_dictionary():
    return {'type': {'cnt': 'Count', 'number': "Number"},
            'car': {"cnt": 0, "number": "number"},
            'car with trailer': {"cnt": 0, "number": "number"},
            'motorbike': {"cnt": 0, "number": "number"},
            'VAN': {"cnt": 0, "number": "number"},
            'train': {"cnt": 0, "number": "number"},
            'HGV': {"cnt": 0, "number": "number"},
            'HGV with trailer': {"cnt": 0, "number": "number"},
            'HGV articulated': {"cnt": 0, "number": "number"},
            'other vehicles': {"cnt": 0, "number": "number"}}


def is_object_on_line(mask, rect):
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    return cv2.countNonZero(mask[y:y + h, x:x + w]) != 0


def draw_logs(dict, bg):
    for data in dict:
        cv2.putText(bg, data['number'] if data["number"] != "number" else "", (1570, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)


def sleep_video(cap, bg, frame, ret):
    while ret:
        h = (1345 - 576) * frame.shape[0] // frame.shape[1]
        rszd = cv2.resize(frame, (1345 - 576, h))
        bg[254:254 + h, 576:1345] = rszd

        current_time = datetime.now()
        curr_time = "%s-%s-%s %s:%s:%s" % (
            current_time.month, current_time.day, current_time.year, current_time.hour, current_time.minute,
            current_time.second)
        cv2.putText(bg, curr_time, (590, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("", bg)
        # Press Q on keyboard to  exit
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
    return ret, frame


def find_intersected(mask, bboxes, cls_names):
    intersected_objs = []
    for i in range(len(bboxes)):
        if is_object_on_line(mask, bboxes[i]):
            intersected_objs.append({"box": bboxes[i], "cls": cls_names[i]})
    return intersected_objs


def draw_detection(frame, tracker_out):
    out_frame = frame
    for i in range(len(tracker_out)):
        index = tracker_out[i][0]
        box = tracker_out[i][1]["box"]
        x1, y1, w1, h1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        out_frame = cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 5)
        heigth, width, _ = out_frame.shape
        cv2.putText(out_frame, str(index), (x1, y1 - 12), 0, 1e-3 * heigth, (255, 0, 0), 5)
    return out_frame


def draw_number(bg, number, cls_name):
    cv2.putText(bg, str(cls_name), (230, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(bg, number, (1570, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return bg


def draw_cropped(out, bg):
    h = (490 - 50) * out.shape[0] // out.shape[1]
    h = min(bg.shape[0], 340 + h) - 340
    rszd = cv2.resize(out, (490 - 50, h))
    bg[340:340 + h, 50:490] = rszd
    return bg


def draw_labels(trf_dict, bg, tracker):
    t = 0
    for key, value in trf_dict.items():
        log_number = trf_dict[key]['number'] if trf_dict[key]["number"] != "number" else ""
        cv2.putText(bg, key, (1400, 400 + t * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(bg, str(trf_dict[key]['cnt']), (1600, 400 + t * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(bg, log_number, (1700, 400 + t * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        t += 1
    cv2.line(bg, (1400, 870), (1800, 870), (0, 0, 0), 3)
    cv2.putText(bg, "Traffic counter:", (1400, 910), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(bg, str(tracker.count()), (1750, 910), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return bg


def draw_frame(frame, bg, current_time):
    h = (1345 - 576) * frame.shape[0] // frame.shape[1]
    rszd = cv2.resize(frame, (1345 - 576, h))
    bg[254:254 + h, 576:1345] = rszd

    curr_time = "%s-%s-%s %s:%s:%s" % (
        current_time.month, current_time.day, current_time.year, current_time.hour, current_time.minute,
        current_time.second)
    cv2.putText(bg, curr_time, (590, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return bg, current_time


def draw_all(frame, bg, obj, trf_dict, tracker, current_time):
    bg, current_time = draw_frame(frame, bg, current_time)
    box = obj[1]["box"]
    cls_name = obj[1]["cls"]
    x, y, w, h = box[0], box[1], box[2], box[3]
    out = frame[y:y + h, x:x + w]
    bg = draw_cropped(out, bg)

    # pre_number = alpr.recognize_ndarray(out)['results']
    number = "number"
    if len(pre_number) != 0:
        number = str(pre_number[0]['plate'])
    if number != "number":
        bg = draw_number(bg, number, cls_name)
        trf_dict[cls_name]['number'] = number
    trf_dict[cls_name]['cnt'] += 1
    bg = draw_labels(trf_dict, bg, tracker)
    return bg
