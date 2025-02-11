import os
import cv2
import numpy as np
import argparse
import logging

from typing import Tuple, List
from rich.logging import RichHandler

logging.basicConfig(
    format="{levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
    handlers=[RichHandler(show_time=False)]
)

logger = logging.getLogger("SLAM_UTILS")

def parse_arguments() -> str:
  parser = argparse.ArgumentParser(description="run slam on a sequence of images")
  parser.add_argument("frames_path", type=str, help="directory path to the frames")
  args = parser.parse_args()
  return args.frames_path

def load_camera_calibration() -> Tuple[np.ndarray, np.ndarray]:
    K: np.ndarray = np.array([
        [788.63, 0, 687.16],
        [0, 786.38, 317.75],
        [0, 0, 1]
    ]) # from kitti dataset

    D: np.ndarray = np.array([-0.344441, 0.141678, 0.000414, -0.000222, -0.029608])
    return K, D

def undistort_image(image: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    return cv2.undistort(image, K, D)

def validate_frames_path(frames_path: str) -> List[str]:
    if not os.path.isdir(frames_path):
        logger.error(f"[ERROR] Path does not exist: {frames_path}")
        exit(1)

    image_files: List[str] = sorted([f for f in os.listdir(frames_path) if f.endswith(".png")])

    if len(image_files) == 0:
        logger.error(f"[ERROR] No images found in {frames_path}")
        exit(1)

    return image_files
