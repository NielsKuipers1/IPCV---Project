import cv2
import numpy as np
import os

# function to detect lines
def detect_corners(gray,frame):
    points = []
    corners = cv2.cornerHarris(gray, 50, 3, 0.01) 
    corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dilated = cv2.dilate(thresh, None, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(gray, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            points.append((cx,cy))
    return gray, points

def detect_lines(frame, boundaries):

    # Step 1: Color thresholding to isolate white lines
    combined_mask = np.zeros(frame.shape[:2], dtype="uint8")
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(frame, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Step 2: Apply the mask and convert to grayscale
    output = cv2.bitwise_and(frame, frame, mask=combined_mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    intersections = []
    # Step 3: Edge detection to get line contours
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    edges = cv2.dilate(edges, (3,3), iterations=10)
    line_image = np.copy(combined_mask) * 0  # creating a blank to draw lines on
    # Step 4: Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=150, minLineLength=450, maxLineGap=50)
    points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)        
    line_image_colored = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)
    gray = frame * line_image_colored
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # gray, points = detect_corners(gray,frame)
    gray = skeleton(line_image)
    return frame, gray, line_image

def skeleton(image):
    # Threshold the image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Initialize a skeleton (empty image)
    skeleton = np.zeros(binary.shape, dtype=np.uint8)

    # Create a cross-shaped structuring element for morphological operations
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Perform the skeletonization
    while True:
        # Erode the image
        eroded = cv2.erode(binary, element)
        # Open the eroded image
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary, temp)
        # Add the result to the skeleton
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()

        # If there are no white pixels left, break from the loop
        if cv2.countNonZero(binary) == 0:
            break
    return skeleton

# initiate video 
video_path = 'video2.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# defenition of parameters
frame_count = 0
frames_per_segment = int(fps * 2.5)  
total_segments = 8
total_frames = frames_per_segment * total_segments
boundaries = [([170, 170, 100], [255, 255, 255])] # define threshold to only use the white lines 



while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break
    # Step 1: corner and line detection (localize calibration points)
    # frame, intersections = detect_corners(frame,boundaries)
    output_frame, gray, line_image = detect_lines(frame, boundaries)
    # Step 2 Intrinsic camera calibration using the known 3D positions of reference objects
    
    # Step 3 External camera calibration: the pose of the camera relative to the 3D reference objects

    # Step 4 Tracking of 2D points and/or lines in the movie.

    # Step 5 Based on these tracked points and/or lines, camera pose tracking during the movie.

    # Step 6 Projection of a virtual banner which has a rectangular shape in the real world and located near a court line

    cv2.imshow("Detected Lines", output_frame)
    cv2.imshow("gray", gray)
    cv2.imshow("image", line_image)
    cv2.waitKey(25)
    frame_count += 1 
cap.release()
cv2.destroyAllWindows()