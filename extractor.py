import cv2
import numpy as np
import logging

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
    def __init__(self, WIDTH: int, HEIGHT: int):
        self.orb_detector: cv2.ORB = cv2.ORB_create()  # type: ignore
        self.FLANN_INDEX_LSH = 6
        self.WIDTH: int = WIDTH
        self.HEIGHT: int = HEIGHT
        self.index_params: dict = dict(
            algorithm=self.FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        self.search_params: dict = dict(checks=50)
        self.flann_matcher: cv2.FlannBasedMatcher = cv2.FlannBasedMatcher(
            self.index_params, self.search_params)
        self.K: np.ndarray = np.array(
            [[554, 0, WIDTH // 2], [0, 554, HEIGHT // 2], [0, 0, 1]])
        self.previous_frame: Optional[np.ndarray] = None
        self.store_descriptors: deque[Tuple[List[cv2.KeyPoint], np.ndarray]] = deque(
            maxlen=2)

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        current_frame: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners: Optional[np.ndarray] = cv2.goodFeaturesToTrack(
            current_frame, 3000, 0.001, 3)

        if corners is not None:
            keypoints: List[cv2.KeyPoint] = [cv2.KeyPoint(
                float(c[0][0]), float(c[0][1]), size=20) for c in corners]
            keypoints, descriptors = self.orb_detector.compute(
                current_frame, keypoints)  # type: ignore
        else:
            keypoints, descriptors = self.orb_detector.detectAndCompute(
                current_frame, None)  # type: ignore

        if descriptors is not None:
            self.store_descriptors.append(
                (list(keypoints), descriptors))

        if len(self.store_descriptors) == 2:
            (keypoints1, descriptor1), (keypoints2,
                                        descriptor2) = self.store_descriptors

            good_features: List[Tuple[cv2.KeyPoint, cv2.KeyPoint]] = self.match_features(descriptor1, descriptor2,
                                                                                         keypoints1, keypoints2, current_frame)

            print(f"Number of good features: {len(good_features)}")

            if len(good_features) >= 8:

                pts1: np.ndarray = np.float32(
                    [kp1.pt for kp1, _ in good_features])  # type: ignore
                pts2: np.ndarray = np.float32(
                    [kp2.pt for _, kp2 in good_features])  # type: ignore

                pts1_homogeneous: np.ndarray = np.hstack(
                    [pts1, np.ones((pts1.shape[0], 1))])
                pts2_homogeneous: np.ndarray = np.hstack(
                    [pts2, np.ones((pts2.shape[0], 1))])

                pts1_normalized = (np.linalg.inv(self.K) @
                                   pts1_homogeneous.T).T[:, :2]
                pts2_normalized = (np.linalg.inv(self.K) @
                                   pts2_homogeneous.T).T[:, :2]

                extrinsic_matrix: Optional[np.ndarray] = self.estimate_camera_motion(
                    pts1_normalized, pts2_normalized)

                if extrinsic_matrix is None:
                    logger.warning(
                        "[WARNING] Failed to estimate camera motion")

                else:
                    # logger.info(f'Extrinsic matrix: {extrinsic_matrix}')
                    # logger.info(
                    #     f'Extrinsic matrix shape: {extrinsic_matrix.shape}')
                    triangulated_points: np.ndarray = self.triangulation(
                        pts1_normalized, pts2_normalized, np.eye(4), extrinsic_matrix)
                    logger.info(
                        f"Triangulated points shape: {triangulated_points.shape}")

        return keypoints, descriptors

    def estimate_camera_motion(self, pts1: np.ndarray, pts2: np.ndarray) -> Optional[np.ndarray]:
        if pts1.shape[0] < 8 or pts2.shape[0] < 8:
            logger.warning(
                "[WARNING] Not enough points to estimate camera motion!")
            return None

        essential_matrix, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1)

        if essential_matrix is None:
            logger.warning("[WARNING] Essential matrix is None!")
            return None

        logger.info(f"Essential matrix: {essential_matrix}")

        inlier_pts1: np.ndarray = pts1[mask.ravel() == 1]
        inlier_pts2: np.ndarray = pts2[mask.ravel() == 1]

        if inlier_pts1.shape[0] < 8 or inlier_pts2.shape[0] < 8:
            logger.warning(
                "[WARNING] Not enough inlier points to estimate camera motion")
            return None

        _, R, t, mask = cv2.recoverPose(
            essential_matrix, inlier_pts1, inlier_pts2, self.K)

        extrinsic_matrix: np.ndarray = np.hstack((R, t.reshape(3, 1)))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

        return extrinsic_matrix

    def triangulation(self, pts1: np.ndarray, pts2: np.ndarray, pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        pts1_3D = np.ones((3, pts1.shape[0]))
        pts2_3D = np.ones((3, pts2.shape[0]))

        pts1_3D[0], pts1_3D[1] = pts1[:, 0], pts1[:, 1]
        pts2_3D[0], pts2_3D[1] = pts2[:, 0], pts2[:, 1]

        P1 = self.K @ pose1[:3, :]
        P2 = self.K @ pose2[:3, :]

        P1 = P1.astype(np.float32)
        P2 = P2.astype(np.float32)

        pts_4D = cv2.triangulatePoints(P1, P2, pts1_3D[:2, :], pts2_3D[:2, :])

        pts_4D /= pts_4D[3]

        return pts_4D[:3].T

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
            if m.distance < 0.6 * n.distance:
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
