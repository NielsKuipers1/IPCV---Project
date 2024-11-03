import cv2
import numpy as np
import os

def detect_corners(frame, boundaries, lines, max_distance=100):
    intersections = []

    # Remove colors of lower values (the lines are white)
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)
    
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, 9, 3, 0.01) 
    corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dilated = cv2.dilate(thresh, None, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_points = []
    max_line_length = 1000
    min_line_length = 300
    ccorner = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # Check distance to both ends of the lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate distances to both endpoints
                distance_to_start = np.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
                distance_to_end = np.sqrt((cx - x2) ** 2 + (cy - y2) ** 2)
                
                if distance_to_start <= max_distance or distance_to_end <= max_distance:

                    # TODO: check if point is a cornerpoint, if add it to corner_points and draw it
                    rectHalfWidth = 10
                    roi = gray[cy-rectHalfWidth:cy+rectHalfWidth,cx-rectHalfWidth:cx+rectHalfWidth]
                    print(roi.shape)
                    if isCorner(roi,rectHalfWidth):
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        corner_points.append((cx, cy))

                    intersections.append(contour)

                    break  # No need to check other lines for this contour
    horizontal_lines = []
    margin = 10
    # Track processed points
    processed = set()

    for i in range(len(corner_points)):
        if corner_points[i] in processed:
            continue
        
        line = [corner_points[i]]
        processed.add(corner_points[i])
        
        # Check the next points to see if they are within the margin
        for j in range(i + 1, len(corner_points)):
            if abs(corner_points[j][1] - corner_points[i][1]) <= margin and corner_points[j] not in processed:
                line.append(corner_points[j])
                processed.add(corner_points[j])

        # Check if there are at least 4 points in this line
        if len(line) >= 4:
            horizontal_lines.append(line)

        # for line in horizontal_lines:
        #     for (x, y) in line:
                # cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)  # Red circle for horizontal line points

    # for i in range(len(corner_points)):
    #     for j in range(i + 1, len(corner_points)):
    #         p1 = corner_points[i]
    #         p2 = corner_points[j]
    #         line_length = np.linalg.norm(np.array(p2) - np.array(p1))  # Calculate line length
    #         if line_length >= min_line_length:
    #             if is_line_white(gray, p1, p2):
    #                 cv2.line(frame, p1, p2, (0, 255, 0), 2)  # Draw line
    #                                 # Classify lines as horizontal or vertical  
    return frame, ccorner

def binaryIm(frame, thresh):
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameBinary = cv2.threshold(frameGray, thresh, 255, cv2.THRESH_BINARY)[1]
    return frameBinary

def isCorner(frame, rectHalfWidth):
    leftSide = False
    upperSide = False
    rightSide = False
    lowerSide = False

   
    for i in range(0,frame.shape[0]-1):
        # check leftside
        if frame[0,i] == 255:
            leftSide = True
            break
    for i in range(0,frame.shape[0]-1):
        # check upperside
        if frame[i,0] == 255:
            upperSide = True
            break
    for i in range(0,frame.shape[0]-1):
        # check rightside
        if frame[frame.shape[0]-1,i] == 255:
            rightSide = True
            break
    for i in range(0,frame.shape[0]-1):
        # check lowerside
        if frame[i,frame.shape[0]-1] == 255:
            lowerSide = True
            break
    iscorner = False
    if (leftSide and upperSide) or (upperSide and rightSide) or (rightSide and lowerSide) or (lowerSide and leftSide):
        iscorner = True
    return iscorner

def is_line_white(frame, p1, p2, color_threshold=200, white_percentage=0.9):
    line_points = np.linspace(p1, p2, num=100, dtype=int)  # Sample 100 points along the line
    white_count = 0
    total_points = len(line_points)
    
    for point in line_points:
        x, y = point
        # Ensure coordinates are within the frame boundaries
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            # Check if the pixel is "white enough" by comparing to the threshold
            if (frame[y, x] >= color_threshold).all():
                white_count += 1
    
    # Check if the white count meets the required percentage
    return (white_count / total_points) >= white_percentage

def detect_lines(frame, boundaries, min_line_length=80):
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

    # Step 3: Edge detection to get line contours
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Step 4: Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    # Draw detected lines on the frame
    # Filter out short lines
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length >= min_line_length:  # Check if line length meets the minimum requirement
                filtered_lines.append(line)
                # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw long lines only
    return lines, filtered_lines

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

kernel_size = (1, 1)  # Adjust kernel size based on the desired blurring
sigma = 0  # Auto-calculate sigma based on kernel size

while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break
    # Step 1: corner and line detection (localize calibration points)
    blurred_frame = cv2.GaussianBlur(frame, kernel_size, sigma)
    output_frame, detected_lines = detect_lines(blurred_frame, boundaries)
    output_frame, intersections = detect_corners(blurred_frame,boundaries,detected_lines)

    thresh = 200
    testOutput = binaryIm(blurred_frame, thresh)
    
    # Step 2 Intrinsic camera calibration using the known 3D positions of reference objects
    
    # Step 3 External camera calibration: the pose of the camera relative to the 3D reference objects

    # Step 4 Tracking of 2D points and/or lines in the movie.

    # Step 5 Based on these tracked points and/or lines, camera pose tracking during the movie.

    # Step 6 Projection of a virtual banner which has a rectangular shape in the real world and located near a court line

    cv2.imshow("Detected corners", output_frame)
    cv2.waitKey(25)
    frame_count += 1 
cap.release()
cv2.destroyAllWindows()