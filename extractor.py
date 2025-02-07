import cv2
import numpy as np
from collections import deque
from typing import Optional, List, Tuple


class FeatureExtractor:
    def __init__(self):
        self.orb_detector: cv2.ORB = cv2.ORB_create()  # type: ignore
        self.bf_matcher: cv2.BFMatcher = cv2.BFMatcher(
            cv2.NORM_HAMMING, crossCheck=False)

        self.previous_frame: Optional[np.ndarray] = None
        self.store_descriptors: deque[Tuple[List[cv2.KeyPoint], np.ndarray]] = deque(
            maxlen=2)

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        current_frame: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb_detector.detectAndCompute(
            current_frame, np.array([]))

        if descriptors is not None:
            self.store_descriptors.append(
                (list(keypoints), descriptors))

        if len(self.store_descriptors) == 2:
            (keypoints1, descriptor1), (keypoints2,
                                        descriptor2) = self.store_descriptors
            self.match_features(descriptor1, descriptor2,
                                keypoints1, keypoints2, current_frame)

        return list(keypoints), descriptors

    def match_features(self,
                       descriptors1: Optional[np.ndarray],
                       descriptors2: Optional[np.ndarray],
                       keypoints1: List[cv2.KeyPoint],
                       keypoints2: List[cv2.KeyPoint],
                       current_frame: np.ndarray
                       ) -> List[List[cv2.DMatch]]:

        if (descriptors1 is None or descriptors2 is None or
                len(descriptors1) == 0 or len(descriptors2) == 0):
            print("[WARNING] ORB failed to compute descriptors! No matches found.")
            return []

        matches: List[List[cv2.DMatch]] = self.bf_matcher.knnMatch(
            descriptors1, descriptors2, k=2)  # type: ignore

        good_matches: List[Tuple[cv2.KeyPoint, cv2.KeyPoint]] = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(
                    (keypoints1[m.queryIdx], keypoints2[m.trainIdx]))

        self.visualize_features(good_matches, current_frame)
        return matches

    def visualize_features(self, good_matches: List[Tuple[cv2.KeyPoint, cv2.KeyPoint]], frame: np.ndarray) -> None:
        current_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for kp1, kp2 in good_matches:
            x1, y1 = int(kp1.pt[0]), int(kp1.pt[1])
            x2, y2 = int(kp2.pt[0]), int(kp2.pt[1])

            print(f"Matching points: ({x1}, {y1}) ↔ ({x2}, {y2})")

            cv2.circle(current_frame, (x1, y1), 3, (0, 255, 0), -1)
            cv2.circle(current_frame, (x2, y2), 3, (0, 255, 0), -1)

            cv2.line(current_frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

        cv2.imshow("Feature Matches", current_frame)
        cv2.waitKey(1)
