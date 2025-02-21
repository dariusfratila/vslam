import cv2
import numpy as np
import logging
import open3d as o3d


from collections import deque
from typing import Optional, List, Tuple
from rich.logging import RichHandler

logging.basicConfig(
    format="{levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
    handlers=[RichHandler(show_time=False)]
)

logger = logging.getLogger("SLAM")


class FeatureExtractor:
    def __init__(self, K: np.ndarray, WIDTH: int, HEIGHT: int):
        self.orb_detector: cv2.ORB = cv2.ORB_create()  # type: ignore
        self.FLANN_INDEX_LSH = 6
        self.WIDTH: int = WIDTH
        self.HEIGHT: int = HEIGHT
        self.index_params: dict = dict(
            algorithm=self.FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        self.search_params: dict = dict(checks=50)
        self.flann_matcher: cv2.FlannBasedMatcher = cv2.FlannBasedMatcher(
            self.index_params, self.search_params)
        self.K: np.ndarray = K
        logger.info(f"Camera matrix: {self.K}")

        self.keyframes: List[np.ndarray] = []
        self.trajectory: List[np.ndarray] = []

        self.global_pose: np.ndarray = np.eye(4, dtype=np.float64)
        self.last_keyframe_pose: np.ndarray = np.eye(4, dtype=np.float64)
        self.current_camera_pose: np.ndarray = np.zeros(3, dtype=np.float64)

    def extract_features(self, frame: np.ndarray):
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(current_frame, 1000, 0.01, 5)

        if corners is not None:
            keypoints: List[cv2.KeyPoint] = [cv2.KeyPoint(
                float(c[0][0]), float(c[0][1]), size=20) for c in corners]
            keypoints, descriptors = self.orb_detector.compute(
                current_frame, keypoints)  # type: ignore
        else:
            keypoints, descriptors = self.orb_detector.detectAndCompute(
                current_frame, None)  # type: ignore

        logger.info(
            f'Extracted {len(keypoints)} features from the current frame')

        if not self.keyframes:
            self.trajectory.append(self.current_camera_pose[:2])  # xy values
            self.keyframes.append(
                (keypoints, descriptors, self.global_pose.copy()))
            logger.info("Initialized with first keyframe at pose [0, 0, 0]")
            return keypoints, descriptors, None

        prev_keypoints, prev_descriptors, prev_pose = self.keyframes[-1]
        good_matches = self.match_features(
            prev_descriptors, descriptors, prev_keypoints, keypoints, current_frame)

        logger.info(
            f'Matching with the previous keyframe: {len(good_matches)}')

        if len(good_matches) >= 8:
            pts1 = np.array([kp1.pt for kp1, _ in good_matches])
            pts2 = np.array([kp2.pt for _, kp2 in good_matches])
            camera_motion = self.estimate_camera_motion(pts1, pts2)

            if camera_motion is not None:
                extrinsic_matrix, inlier_pts1, inlier_pts2 = camera_motion

                self.global_pose = prev_pose @ extrinsic_matrix
                R, t = self.global_pose[:3, :3], self.global_pose[:3, 3]
                self.current_camera_pose = -R.T @ t
                self.trajectory.append(self.current_camera_pose[:2])

                if len(good_matches) < 40:
                    self.keyframes.append(
                        (keypoints, descriptors, self.global_pose.copy()))
                    logger.info(
                        "Matches < 70, added new keyframe to maintain tracking")

                else:
                    logger.info(
                        "Matches >= 70, sufficient overlap with last keyframe, no new keyframe added")

            else:
                logger.warning('Failed to estimate camera motion')
                if self.trajectory:
                    self.trajectory.append(self.trajectory[-1])
        else:
            logger.warning('Not enough matches to estimate camera motion')
            if self.trajectory:
                self.trajectory.append(self.trajectory[-1])

        return keypoints, descriptors, None

    def estimate_camera_motion(self, pts1: np.ndarray, pts2: np.ndarray):
        if pts1.shape[0] < 8 or pts2.shape[0] < 8:
            logger.warning(
                "[WARNING] Not enough points to estimate camera motion!")
            return None, None, None

        essential_matrix, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1
        )

        if essential_matrix is None:
            logger.warning("[WARNING] Essential matrix is None!")
            return None, None, None

        inlier_pts1: np.ndarray = pts1[mask.ravel() == 1]
        inlier_pts2: np.ndarray = pts2[mask.ravel() == 1]

        if inlier_pts1.shape[0] < 8 or inlier_pts2.shape[0] < 8:
            logger.warning(
                "[WARNING] Not enough inlier points to estimate camera motion")
            return None, None, None

        _, R, t, new_mask = cv2.recoverPose(
            essential_matrix, inlier_pts1, inlier_pts2, self.K
        )

        final_inlier_pts1: np.ndarray = inlier_pts1[new_mask.ravel() == 255]
        final_inlier_pts2: np.ndarray = inlier_pts2[new_mask.ravel() == 255]

        extrinsic_matrix: np.ndarray = np.eye(4, dtype=np.float64)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = t.ravel()

        return extrinsic_matrix, final_inlier_pts1, final_inlier_pts2

    def triangulation(self, pts1: np.ndarray, pts2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        pts1_2D = pts1.T
        pts2_2D = pts2.T

        pts_4D = cv2.triangulatePoints(P1, P2, pts1_2D, pts2_2D)

        pts_4D /= pts_4D[3]

        X_world = pts_4D[:3].T

        return X_world

    def match_features(self,
                       descriptors1: Optional[np.ndarray],
                       descriptors2: Optional[np.ndarray],
                       keypoints1: List[cv2.KeyPoint],
                       keypoints2: List[cv2.KeyPoint],
                       current_frame: np.ndarray
                       ) -> List[Tuple[cv2.KeyPoint, cv2.KeyPoint]]:

        if (descriptors1 is None or descriptors2 is None or
                len(descriptors1) == 0 or len(descriptors2) == 0):
            logger.warning(
                "[WARNING] ORB failed to compute descriptors! No matches found.")
            return []

        matches: List[List[cv2.DMatch]] = self.flann_matcher.knnMatch(
            descriptors1, descriptors2, k=2)  # type: ignore

        good_matches: List[Tuple[cv2.KeyPoint, cv2.KeyPoint]] = []

        # print(len(matches))

        for _, pair in enumerate(matches):
            if len(pair) < 2:
                continue

            m, n = pair
            if m.distance < 0.50 * n.distance:
                good_matches.append(
                    (keypoints1[m.queryIdx], keypoints2[m.trainIdx]))

        self.visualize_features(good_matches, current_frame)
        return good_matches

    def visualize_features(self, good_matches: List[Tuple[cv2.KeyPoint, cv2.KeyPoint]], frame: np.ndarray) -> None:
        current_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for kp1, kp2 in good_matches:
            x1, y1 = int(kp1.pt[0]), int(kp1.pt[1])
            x2, y2 = int(kp2.pt[0]), int(kp2.pt[1])

            # print(f"Matching points: ({x1}, {y1}) ↔ ({x2}, {y2})")

            cv2.circle(current_frame, (x1, y1), 3, (0, 255, 0), -1)
            cv2.circle(current_frame, (x2, y2), 3, (0, 255, 0), -1)

            cv2.line(current_frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

        cv2.imshow("Feature Matches", current_frame)
        cv2.waitKey(1)
