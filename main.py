import cv2
import numpy as np
import os

def detect_lines(frame, canny_threshold1=100, canny_threshold2=700):
    """
    Detects lines in an image and returns the image with lines drawn.
    
    Parameters:
    - frame: Input image (as a NumPy array).
    - canny_threshold1: Lower threshold for the Canny edge detector.
    - canny_threshold2: Upper threshold for the Canny edge detector.
    - hough_threshold: Threshold for the Hough Line Transform.
    
    Returns:
    - output_image: Image with detected lines drawn on it.
    - lines: List of detected lines in (rho, theta) format.
     """
    # Step 1: Create a mask for white pixels in the color image
    # Define lower and upper bounds for white color in the BGR color space
    lower_white = np.array([170, 170, 170], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    # Apply the color mask to isolate white pixels
    white_mask = cv2.inRange(frame, lower_white, upper_white)
    
    # Step 2: Apply the mask to get only white regions in the grayscale image
    masked_image = cv2.bitwise_and(frame, frame, mask=white_mask)


    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Edge detection using Canny
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2, apertureSize=3)
    
    # Step 3: Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(
    edges,
    rho=1,                # Distance resolution in pixels
    theta=np.pi / 360,    # Angular resolution in radians (increase for finer resolution)
    threshold=100,         # Lower threshold for detecting lines (try 50 if 100 was too high)
    minLineLength=200,     # Decrease minLineLength if lines are short in the bottom-right
    maxLineGap=300         # Increase maxLineGap to bridge broken segments
)
    
    # Copy the original image to draw lines on
    output_image = frame.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Get merged line coordinates
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print("No lines detected")

    gray_float = np.float32(gray)

    # Apply Harris Corner Detection
    corners_harris = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

    # Dilate the corner image to mark the corners
    corners_harris = cv2.dilate(corners_harris, None)

    # Threshold for an optimal value
    threshold = 0.01 * corners_harris.max()
    output_image[corners_harris > threshold] = [0, 0, 255]  # Mark corners in red


    
    return output_image, lines

video_path = 'video2.mp4'
cap = cv2.VideoCapture(video_path)
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_segment = int(fps * 2.5)  
total_segments = 8
total_frames = frames_per_segment * total_segments
boundaries = [([170, 170, 100], [255, 255, 255])]


while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break
    # Step 1: line detection
    frame, lines = detect_lines(frame)
    
    cv2.imshow('frame',frame)
    cv2.waitKey(25)
    frame_count += 1 
#cap.release()
#cv2.destroyAllWindows()