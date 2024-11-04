import cv2
import numpy as np
from scipy.spatial import KDTree
import os
import math

def detect_corners(frame):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Harris Corner Detection
    gray_float = np.float32(blurred)
    corners_harris = cv2.cornerHarris(gray_float, blockSize=5, ksize=7, k=0.04)
    corners_harris = cv2.dilate(corners_harris, None)
    threshold = 0.003 * corners_harris.max()
    corner_coords = np.argwhere(corners_harris > threshold)
    points = np.array(corner_coords)

    # Group and filter points
    grouped_points = group_close_points(points, distance_threshold=22)
    flat_points = np.array([np.mean(group, axis=0) for group in grouped_points])
    isolated_points = filter_isolated_points(flat_points, distance_threshold=40)

    return isolated_points

def filter_isolated_points(points, distance_threshold):
    tree = KDTree(points)
    isolated_points = []
    for i, point in enumerate(points):
        neighbors = tree.query_ball_point(point, r=distance_threshold)
        if len(neighbors) == 1:  # Only keep points with no neighbors within the threshold
            isolated_points.append(point)
    return np.array(isolated_points)

def group_close_points(points, distance_threshold):
    tree = KDTree(points)
    visited = set()
    grouped = []
    for i, point in enumerate(points):
        if i in visited:
            continue
        indices = tree.query_ball_point(point, r=distance_threshold)
        cluster = points[indices]
        grouped.append(cluster)
        visited.update(indices)
    return grouped

def draw_horizontal_lines(frame, corners, margin=20):
    horizontal_lines = []
    processed = set()  # Track processed points
    leftmost_rightmost_pairs = []  # Store the leftmost and rightmost points of each line

    for i in range(len(corners)):
        if tuple(corners[i]) in processed:
            continue
        
        line = [corners[i]]
        processed.add(tuple(corners[i]))  # Store as a tuple for immutability
        
        # Check the next points to see if they are within the margin
        for j in range(i + 1, len(corners)):
            if abs(corners[j][0] - corners[i][0]) <= margin and tuple(corners[j]) not in processed:
                line.append(corners[j])
                processed.add(tuple(corners[j]))

        # Check if there are at least 4 points in this line
        if len(line) >= 4:
            horizontal_lines.append(line)
            # Draw the line between the two furthest points in the line
            line = np.array(line)
            leftmost = line[line[:, 1].argmin()]
            rightmost = line[line[:, 1].argmax()]
            # cv2.line(frame, (int(leftmost[1]), int(leftmost[0])), (int(rightmost[1]), int(rightmost[0])), (0, 255, 0), 2)
            
            # Append each leftmost and rightmost pair for labeling later
            leftmost_rightmost_pairs.append((leftmost, rightmost))

    return leftmost_rightmost_pairs  # Return pairs of points for all detected lines

def label_corners(frame, leftmost_rightmost_pairs):
    # Get the vertical center of the frame
    frame_height = frame.shape[0]
    vertical_center = frame_height / 2

    labeled_corners = []  # Initialize a list to hold labeled corners


    for (leftmost, rightmost) in leftmost_rightmost_pairs:
        if leftmost is not None:
            x_left, y_left = int(leftmost[1]), int(leftmost[0])
            # Determine position relative to center
            position_left = "above" if y_left < vertical_center else "below"
            label_left = f"Leftmost ({position_left})"
            # cv2.putText(frame, label_left, (x_left + 5, y_left - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Append to labeled corners
            labeled_corners.append(((x_left, y_left), label_left))
        
        if rightmost is not None:
            x_right, y_right = int(rightmost[1]), int(rightmost[0])
            # Determine position relative to center
            position_right = "above" if y_right < vertical_center else "below"
            label_right = f"Rightmost ({position_right})"
            # cv2.putText(frame, label_right, (x_right + 5, y_right - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Append to labeled corners
            labeled_corners.append(((x_right, y_right), label_right))

    return labeled_corners  # Return the list of labeled corners

def draw_labeled_corners(frame, labeled_corners):
    """
    Draw the labeled corners on the frame.
    
    Parameters:
    - frame: The current frame of the video.
    - labeled_corners: A list of tuples containing corner coordinates and labels.
    """
    for (corner, label) in labeled_corners:
        x, y = corner
        # cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw corners in red
        # cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
# Determine begin and endpoint of ad
def getBeginEndCoords(beginAdProcent, endAdProcent, lowerCorner, upperCorner):
    xBegin = int(lowerCorner[0]+beginAdProcent*(upperCorner[0]-lowerCorner[0]))
    yBegin = int(lowerCorner[1]+beginAdProcent*(upperCorner[1]-lowerCorner[1]))
    xEnd = int(lowerCorner[0]+endAdProcent*(upperCorner[0]-lowerCorner[0]))
    yEnd = int(lowerCorner[1]+endAdProcent*(upperCorner[1]-lowerCorner[1]))
    return (xBegin,yBegin), (xEnd,yEnd)

# Determine from begin and endpoint of add the distance between the points and the angle
def getLengthAndAngle(beginPoint, endPoint):
    dx = endPoint[0] - beginPoint[0]
    dy = endPoint[1] - beginPoint[1]
    
    # Calculate the distance
    length = math.sqrt(dx**2 + dy**2)
    
    # Calculate the angle in radians 
    angle = math.atan2(dy, dx)
    return length,angle


# Determine the rectangle where ad needs to be projected on
def getParallelogramPoints(beginPoint, endPoint, height, angle, color=(255, 255, 255), thickness=2):
    
    # Calculate dx and dy for the first side
    dx = endPoint[0] - beginPoint[0]
    dy = endPoint[1] - beginPoint[1]

    # Calculate the angle in radians
    angle_rad = math.radians(angle)

    # Calculate the two remaining points based on the length and angle
    point3 = (int(endPoint[0] - height * math.cos(angle_rad - math.atan2(dy, dx))),
              int(endPoint[1] - height * math.sin(angle_rad - math.atan2(dy, dx))))
    point4 = (int(beginPoint[0] - height * math.cos(angle_rad - math.atan2(dy, dx))),
              int(beginPoint[1] - height * math.sin(angle_rad - math.atan2(dy, dx))))
    return point3, point4

# def debugDisplay(frame, beginLine, endLine, adBeginPoint, adEndPoint, adBeginUpperPoint, adEndUpperPoint):
    # cv2.line(frame, endLine, beginLine, color=(0, 255, 0), thickness=2)
    # cv2.circle(frame, adBeginPoint, radius=5, color=(0,0,255))
    # cv2.circle(frame, adEndPoint, radius=5, color=(0,0,255))
    # cv2.circle(frame, adBeginUpperPoint, radius=5, color=(0,0,255))
    # cv2.circle(frame, adEndUpperPoint, radius=5, color=(0,0,255))

# Map ad to the rectangle
def mapIm2Line(frame, image, beginLine, endLine, beginAdProcent, endAdProcent):
    adBeginPoint, adEndPoint = getBeginEndCoords(beginAdProcent, endAdProcent, beginLine, endLine)
    lengthLowerSide, angleLowerSide = getLengthAndAngle(adBeginPoint,adEndPoint)
    adBeginUpperPoint, adEndUpperPoint = getParallelogramPoints(adBeginPoint, adEndPoint, lengthLowerSide, angleOfAd, color=(255, 255, 255), thickness=2)

    # debugDisplay(frame,beginLine,endLine,adBeginPoint,adEndPoint,adBeginUpperPoint,adEndUpperPoint)

    paraArray = np.array([[adBeginUpperPoint[0], adBeginUpperPoint[1]], [adEndUpperPoint[0], adEndUpperPoint[1]], [adBeginPoint[0], adBeginPoint[1]], [adEndPoint[0], adEndPoint[1]]], dtype=np.float32)
    overlayH, overlayW = image.shape[:2]
    imageArray = np.array([[overlayW, 0],[0, 0],[0, overlayH], [overlayW, overlayH]], dtype=np.float32)

    homographyMatrix, _ = cv2.findHomography(imageArray, paraArray)
    warpedImage = cv2.warpPerspective(image, homographyMatrix, (frame.shape[1], frame.shape[0]))
    mask = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    maskInv = cv2.bitwise_not(mask)
    frameBg = cv2.bitwise_and(frame, frame, mask=maskInv)
    combinedFrame = cv2.add(frameBg, warpedImage)
    return combinedFrame



output_video_path = 'video3_edit.mp4'

video_path = 'video3.mp4'
cap = cv2.VideoCapture(video_path)
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_segment = int(fps * 2.5)
total_segments = 8
total_frames = frames_per_segment * total_segments
i = 1
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

overlay_image = cv2.imread('image.png')
beginAdProcent = 0.6
endAdProcent = 0.8
angleOfAd = 0 # angle that ad makes with the perpendicular line to the plotted line in degrees

while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    corners = detect_corners(frame)  # Get detected corners
    for corner in corners:
        x, y = int(corner[1]), int(corner[0])  # Convert to (x, y) for drawing
        # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw in green for detected corners
    
    # Draw horizontal lines and retrieve all leftmost and rightmost pairs
    leftmost_rightmost_pairs = draw_horizontal_lines(frame, corners)

    # Label the leftmost and rightmost corners for each detected line
    labeled_corners = label_corners(frame, leftmost_rightmost_pairs)
    if frame_count < 1:
        previous_corners= labeled_corners
    else:
        if len(labeled_corners) < 4:
            # Action for when the size is smaller than 4
            labeled_corners = previous_corners
            print("1Labeled corners are less than 4")
        elif len(labeled_corners) > 4:
            # Action for when the size is larger than 4
            labeled_corners = previous_corners
            print("1Labeled corners are greater than 4")
        else:  # This will handle the case when the size is exactly 4
            # Action for when the size is equal to 4
            print("1Labeled corners are equal to 4")
    previous_corners = labeled_corners
    draw_labeled_corners(frame, labeled_corners)
    # print(labeled_corners)
    # print(labeled_corners[0][0])

    combined_frame = mapIm2Line(frame, overlay_image, labeled_corners[2][0], labeled_corners[0][0], beginAdProcent, endAdProcent)
    out.write(combined_frame)
    cv2.imshow('frame', combined_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
