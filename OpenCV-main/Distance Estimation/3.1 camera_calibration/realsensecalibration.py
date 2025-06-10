import cv2 as cv
import os
import pyrealsense2 as rs
import numpy as np

# Chessboard settings
Chess_Board_Dimensions = (9, 6)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Directory to save images
image_path = "/home/deb/Downloads/mdpi/images"
if not os.path.isdir(image_path):
    os.makedirs(image_path)
    print(f'"{image_path}" Directory is created')
else:
    print(f'"{image_path}" Directory already exists.')

# RealSense pipeline setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Image counter
n = 0

def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)
    return image, ret

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        frame = np.asanyarray(color_frame.get_data())
        copyFrame = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        image, board_detected = detect_checker_board(frame, gray, criteria, Chess_Board_Dimensions)

        cv.putText(
            image,
            f"saved_img : {n}",
            (30, 40),
            cv.FONT_HERSHEY_PLAIN,
            1.4,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

        cv.imshow("frame", image)
        cv.imshow("copyFrame", copyFrame)

        key = cv.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s") and board_detected:
            cv.imwrite(f"{image_path}/image{n}.png", copyFrame)
            print(f"saved image number {n}")
            n += 1

finally:
    pipeline.stop()
    cv.destroyAllWindows()
    print("Total saved Images:", n)
