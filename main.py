import os
import cv2
import numpy as np
import logging

from typing import List
from extractor import FeatureExtractor
from utils import parse_arguments, load_camera_calibration, undistort_image, validate_frames_path
from rich.logging import RichHandler

import open3d as o3d

open3d_pcd = o3d.geometry.PointCloud()

logging.basicConfig(
    format="{levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
    handlers=[RichHandler(show_time=False)]
)

logger = logging.getLogger("SLAM_MAIN")


def process_frame(image_path: str, feature_extractor: FeatureExtractor, K: np.ndarray, D: np.ndarray):
    image: np.ndarray = cv2.imread(image_path)

    if image is None:
        logger.warning(f"[WARNING] Failed to load image: {image_path}")
        return

    image = undistort_image(image, K, D)
    _, _, triangulated_points = feature_extractor.extract_features(image)
    return triangulated_points


if __name__ == "__main__":
    frames_path: str = parse_arguments()

    image_files: List[str] = validate_frames_path(frames_path)

    K, D = load_camera_calibration()

    WIDTH = 1280
    HEIGHT = 720
    feature_extractor = FeatureExtractor(K, WIDTH, HEIGHT)

    vis = o3d.visualization.Visualizer()  # type: ignore
    vis.create_window(width=WIDTH, height=HEIGHT)

    pcd = o3d.geometry.PointCloud()
    points = np.random.rand(10, 3)
    pcd.points = o3d.utility.Vector3dVector(points)

    vis.add_geometry(pcd)

    for image_file in image_files:
        image_path = os.path.join(frames_path, image_file)
        triangulated_points = process_frame(
            image_path, feature_extractor, K, D)
        # logger.info(f"Triangulated points: {triangulated_points}")

        if triangulated_points is not None:
            pcd.points.extend(triangulated_points)
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

    cv2.destroyAllWindows()
    vis.destroy_window()
