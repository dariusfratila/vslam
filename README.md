# Visual SLAM System

## Overview
This repository contains the implementation of a **visual SLAM system** that processes image sequences to simultaneously localize the camera and construct a 3D map of the environment in real time. The project leverages robust computer vision techniques—including ORB-based feature extraction, FLANN matching, essential matrix estimation, and triangulation—combined with Open3D for dynamic visualization.

---

## Features
- **Real-Time Camera Localization:** Estimate camera pose through essential matrix recovery and pose estimation algorithms.
- **Robust Feature Extraction:** Use ORB for detecting and matching keypoints, ensuring accurate tracking across frames.
- **3D Reconstruction:** Apply triangulation methods to generate precise 3D point clouds from sequential image data.
- **Modular Pipeline:** Incorporates camera calibration, image undistortion, and efficient data handling for seamless integration.

---

## Code Structure
```plaintext
VSLAM_PROJECT/
│
├── extractor.py           # Module for feature extraction, matching, and camera motion estimation
├── utils.py               # Utility functions for camera calibration, image undistortion, and data validation
├── main.py                # Main script for processing image sequences and visualizing the 3D map
├── README.md              # Project documentation and setup instructions
