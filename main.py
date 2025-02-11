import os
import cv2
import numpy as np
import logging

from typing import List
from extractor import FeatureExtractor
from utils import parse_arguments, load_camera_calibration, undistort_image, validate_frames_path
from rich.logging import RichHandler

logging.basicConfig(
    format="{levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
    handlers=[RichHandler(show_time=False)]
)

logger = logging.getLogger("SLAM_MAIN")

def process_frame(image_path: str, feature_extractor: FeatureExtractor, K: np.ndarray, D: np.ndarray) -> None:
    image: np.ndarray = cv2.imread(image_path)

    if image is None:
        logger.warning(f"[WARNING] Failed to load image: {image_path}")
        return

    image = undistort_image(image, K, D)
    feature_extractor.extract_features(image)

if __name__ == "__main__":
    frames_path: str = parse_arguments()

    image_files: List[str] = validate_frames_path(frames_path)

    K, D = load_camera_calibration()

    WIDTH, HEIGHT = 1408, 376
    feature_extractor = FeatureExtractor(WIDTH, HEIGHT)

    logger.info(f"Processing {len(image_files)} frames from {frames_path}")

    for image_file in image_files:
        image_path = os.path.join(frames_path, image_file)
        process_frame(image_path, feature_extractor, K, D)

    cv2.destroyAllWindows()
