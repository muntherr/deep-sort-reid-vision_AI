
from typing import List

import cv2
from torch import Tensor
from deep_sort_reid.types.detection import Detection


def extract_features(file_input_path: str,
                     model,
                     frames_detections: List[List[Detection]],
                     verbose=False
                     ) -> List[List[Tensor]]:
    cap = cv2.VideoCapture(file_input_path)

    if not cap.isOpened():
        print("Can't open video file")

    frame_idx = 0
    frames_features: List[List[Tensor]] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if verbose:
            print("EXTRACTING FEATURES FRAME IDX ", frame_idx)
        detections = frames_detections[frame_idx]

        features = []
        for detection in detections:
            # Images are indexed as (H, W, C)
            cropped_img = frame[int(detection.coords.start_y):int(detection.coords.end_y),
                                int(detection.coords.start_x):int(detection.coords.end_x)]

            object_features: Tensor = model(cropped_img)
            # .to(device)
            features.append(object_features.flatten())

        frames_features.append(features)
        frame_idx += 1

    return frames_features
