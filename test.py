import cv2
import numpy as np

def line_feature_extraction(image):
    "This function utilizes the Hessian matrix method to extract the line features"
    # Convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the componenets of the hessian matrix 
    dxx = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=3)
    dyy = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=3)
    dxy = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    # Compute eigenvalues of the hessian matrix
    lambda1 = 0.5 * (dxx + dyy + np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2))
    lambda2 = 0.5 * (dxx + dyy - np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2))
    # Use the minimum eigenvalue as line response
    hessian_response = np.minimum(lambda1, lambda2)
    hessian_response = cv2.normalize(hessian_response, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(hessian_response)

def line_elements_extraction(image,percentage):
    "This functions extracts the 'best' lines for further use in the line detection"
    # Sort hessian responces 
    sorted_response = np.sort(image.flatten())[::-1]
    total_pixels = len(sorted_response)
    threshold_index = int(total_pixels * (percentage / 100.0))
    # theshold the result
    threshold_value = sorted_response[threshold_index]
    _, feat = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return feat

def dilate_lines(image, kernel_size=3, iterations=5):
    # Create a kernel for dilation
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    
    # Apply dilation to bridge small gaps
    image = cv2.dilate(image, kernel, iterations=iterations)
    return image

def extend_line(line, frame_shape):
    """ Extend the line to the edges of the frame. """
    x1, y1, x2, y2 = line[0]
    
    # Calculate the slope and intercept of the line
    if x1 == x2:  # vertical line
        return [(x1, 0), (x1, frame_shape[0])]
    if y1 == y2:  # horizontal line
        return [(0, y1), (frame_shape[1], y1)]
    
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Calculate intersection points with the borders
    points = []
    
    # Check y-intercept (x = 0)
    y0 = int(intercept)  
    if 0 <= y0 < frame_shape[0]:
        points.append((0, y0))
        
    # Check x-intercept (y = 0)
    x0 = int(-intercept / slope)  
    if 0 <= x0 < frame_shape[1]:
        points.append((x0, 0))

    # Check right edge (x = frame_shape[1])
    y1 = slope * frame_shape[1] + intercept
    if 0 <= y1 < frame_shape[0]:
        points.append((frame_shape[1], int(y1)))

    # Check bottom edge (y = frame_shape[0])
    x2 = (frame_shape[0] - intercept) / slope  
    if 0 <= x2 < frame_shape[1]:
        points.append((int(x2), frame_shape[0]))
    return points

def find_intersections(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1 = lines[i][0]
            x2, y2 = lines[i][1]
            x3, y3 = lines[j][0]
            x4, y4 = lines[j][1]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                continue
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            
            intersections.append((int(px), int(py)))
    return intersections

def find_court_lines(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=150, maxLineGap=10)
    img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return  img_with_lines, lines

def line_detection_main(image):
    extended_lines = []
    "Main structure for line dectection"
    image = line_feature_extraction(image)
    image = line_elements_extraction(image,percentage=95)
    image, lines = find_court_lines(image)
    if lines is not None:
        for line in lines:
            extended_points = extend_line(line, frame.shape)
            extended_lines.append(extended_points)
            cv2.line(image, extended_points[0], extended_points[1], (255, 0, 0), 2)
    # intersections = find_intersections(extended_lines)
    return image


video_path = 'video.mp4'
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
    Processed_frame = line_detection_main(frame)
    cv2.imshow('frame',Processed_frame)
    cv2.imshow('og frame',frame)
    cv2.waitKey(25)
    frame_count += 1 
cap.release()
cv2.destroyAllWindows()