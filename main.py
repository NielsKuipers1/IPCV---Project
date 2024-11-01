import cv2
import numpy as np
import os

# def detect_intersections(frame,boundaries):
#     intersections = []

#     # remove colors of lower values (the lines are white)
#     for (lower, upper) in boundaries:
#         lower = np.array(lower, dtype="uint8")
#         upper = np.array(upper, dtype="uint8")
#         mask = cv2.inRange(frame, lower, upper)
#         output = cv2.bitwise_and(frame, frame, mask=mask)
#     gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
#     corners = cv2.cornerHarris(gray, 9, 3, 0.01) 
#     corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     _, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     dilated = cv2.dilate(thresh, None, iterations=1)
#     contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
#     for contour in contours:
#         moments = cv2.moments(contour)
#         if moments["m00"] != 0:
#             cx = int(moments["m10"] / moments["m00"])
#             cy = int(moments["m01"] / moments["m00"])
#             # cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
#             intersections.append(contour)
#     return frame, intersections


# def extend_line(line, frame_shape):
#     """ Extend the line to the edges of the frame. """
#     x1, y1, x2, y2 = line[0]
    
#     # Calculate the slope and intercept of the line
#     if x1 == x2:  # vertical line
#         return [(x1, 0), (x1, frame_shape[0])]
#     if y1 == y2:  # horizontal line
#         return [(0, y1), (frame_shape[1], y1)]
    
#     slope = (y2 - y1) / (x2 - x1)
#     intercept = y1 - slope * x1

#     # Calculate intersection points with the borders
#     points = []
    
#     # Check y-intercept (x = 0)
#     y0 = int(intercept)  
#     if 0 <= y0 < frame_shape[0]:
#         points.append((0, y0))
        
#     # Check x-intercept (y = 0)
#     x0 = int(-intercept / slope)  
#     if 0 <= x0 < frame_shape[1]:
#         points.append((x0, 0))

#     # Check right edge (x = frame_shape[1])
#     y1 = slope * frame_shape[1] + intercept
#     if 0 <= y1 < frame_shape[0]:
#         points.append((frame_shape[1], int(y1)))

#     # Check bottom edge (y = frame_shape[0])
#     x2 = (frame_shape[0] - intercept) / slope  
#     if 0 <= x2 < frame_shape[1]:
#         points.append((int(x2), frame_shape[0]))
#     return points

# def draw_intersection(frame, positions):
#     if positions is None:
#         return frame
#     print(positions)
#     for (x, y) in positions:
#         print(x)
#         cv2.circle(frame, (x, y), 10, (0, 0, 225), -1)
#     return frame


# def find_corners(intersections,shape):
#     # Initialize corners with None values
#     top_left = top_right = bottom_left = bottom_right = None
    
#     # Initialize extreme values
#     min_x = min_y = float('inf')
#     max_x = max_y = -float('inf')
#     tl_x = tl_y = float('inf')
#     # print(shape)
#     # Loop through the intersections to find the corners
    
#     for (x, y) in intersections:
#         if abs(x) < shape[0] and abs(y) < shape[1]:
#             # Top-left corner (smallest x, smallest y)
#             if x + y < tl_x + tl_y:
#                 top_left = (x, y)
#                 tl_x, tl_y = x, y

#             # # Top-right corner (largest x, smallest y)
#             # if x - y > max_x - min_y:
#             #     top_right = (x, y)
#             #     tr_x = x
#             #     tr_y = y

#             # # Bottom-left corner (smallest x, largest y)
#             # if x - y < min_x - max_y:
#             #     bottom_left = (x, y)

#             # # Bottom-right corner (largest x, largest y)
#             # if x + y > max_x + max_y:
#             #     bottom_right = (x, y)
#             #     max_x, max_y = x, y
#     # if top_left is None or top_right is None or bottom_left is None or bottom_right is None: 
#     #     corners = None
#     # else:
#     corners = top_left
#     print(top_left)
#     cv2.circle(frame, corners, 10, (100, 0, 225), -1)
#     return corners


# def find_intersections(lines):
#     intersections = []
#     for i in range(len(lines)):
#         for j in range(i + 1, len(lines)):
#             x1, y1 = lines[i][0]
#             x2, y2 = lines[i][1]
#             x3, y3 = lines[j][0]
#             x4, y4 = lines[j][1]

#             # Calculate the intersection point using determinants
#             denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#             if denom == 0:  # Lines are parallel
#                 continue
            
#             px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
#             py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            
#             intersections.append((int(px), int(py)))
#     return intersections



def detect_lines(frame, boundaries):
    extended_lines = []
    intersections = []
    corners = []
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)
    kernel_size=1
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    kernel = np.ones(kernel_size, np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=6)
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 120, None, 20, 10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def Harris_Corner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,5,3,0.04)
    image[dst>0.01*dst.max()]=[0,0,255]
    return image


video_path = 'video2.mp4'
cap = cv2.VideoCapture(video_path)
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_segment = int(fps * 2.5)  
total_segments = 8
total_frames = frames_per_segment * total_segments
boundaries = [([180, 180, 100], [255, 255, 255])]


while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect_lines(frame,boundaries)
    frame = Harris_Corner(frame)
    cv2.imshow('frame',frame)
    cv2.waitKey(25)
    frame_count += 1 
cap.release()
cv2.destroyAllWindows()