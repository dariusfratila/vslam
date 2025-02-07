import time
import cv2
from extractor import FeatureExtractor

WIDTH = 640
HEIGHT = 480

feature_extractor = FeatureExtractor()


def process_frame(image):
    image = cv2.resize(image, (WIDTH, HEIGHT))
    keypoints, descriptors = feature_extractor.extract_features(image)

    cv2.imshow("Feature extraction", image)
    cv2.waitKey(1)


if __name__ == "__main__":
    print("[INFO] Starting SLAM system...")
    cap = cv2.VideoCapture("test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame)
        # time.sleep(10)

    cap.release()
    cv2.destroyAllWindows()
