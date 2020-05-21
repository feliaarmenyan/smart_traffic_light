import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'scylla-yolo-v3/python/'))
import darknet as dn
import numpy as np
import cv2


class Detector(object):
    def __init__(self, yolo_pat, confidence=0.5, threshold=0.3, gpu_id=0):
        """
        :param yolo_pat: path to YOLO directory.
        :param confidence: minimum probability to filter weak detections.
        :param threshold: threshold when applying non-maximum suppression.
        """
        self.confidence = confidence
        self.threshold = threshold
        weights_path = os.path.sep.join([yolo_pat, "yolov3-spp.weights"])
        config_path = os.path.sep.join([yolo_pat, "yolov3-spp.cfg"])
        labels_path = os.path.sep.join([yolo_pat, "coco.names"])
        dn.set_gpu(gpu_id)
        self.net = dn.load_net(config_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
        self.meta = dn.load_names(labels_path.encode('utf-8'))
        self.frames = 0
        self.labels = ['car', 'motorbike', 'bus', 'train', 'truck']

    def predict(self, frame):
        """
        :param frame: image to predict.
        :return: final bounding box, confidence of detected objects.
        """
        outputs = dn.detect(self.net, self.meta, dn.nparray_to_image(frame), self.confidence)
        boxes = []
        classes = []
        confidences = []
        for det in outputs:
            cls_name = det[0].decode('utf-8')
            if cls_name in self.labels:
                cls_name = cls_name if cls_name != 'truck' else 'HGV'
                cls_name = cls_name if cls_name != 'bus' else 'VAN'
                classes.append(cls_name)
                confidences.append(float(det[1]))
                cx, cy = int(det[2][0]), int(det[2][1])
                width, height = int(det[2][2]), int(det[2][3])
                boxes.append(np.array([cx - width // 2, cy - height // 2, width, height]))
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        boxes = np.array(boxes)
        classes = np.array(classes)
        confidences = np.array(confidences)
        if len(indexes) == 0:
            return np.array([]), np.array([]), np.array([])
        return boxes[indexes.flatten()], confidences[indexes.flatten()], classes[indexes.flatten()]
