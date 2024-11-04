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
    
    Returns:
    - output_image: Image with detected lines drawn on it.
    - lines: List of detected lines in (rho, theta) format.
    """
    # Step 1: Create a mask for white pixels in the color image
    lower_white = np.array([170, 170, 170], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    white_mask = cv2.inRange(frame, lower_white, upper_white)
    
    # Step 2: Apply the mask to get only white regions in the grayscale image
    masked_image = cv2.bitwise_and(frame, frame, mask=white_mask)

    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection using Canny
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2, apertureSize=3)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 360, threshold=100, minLineLength=150, maxLineGap=350)
    
    # Create an empty image for the lines
    line_mask = np.zeros_like(gray)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Get merged line coordinates
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)  # Draw lines in white on the mask
    else:
        print("No lines detected")

    # Create a kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 10x10 rectangular kernel

    # Apply dilation only to the line mask
    dilated_line_mask = cv2.dilate(line_mask, kernel, iterations=1)

    # Create an output image to draw the dilated lines on
    output_image = frame.copy()

    # Create a color mask to apply the dilated lines to the output image
    dilated_lines_color = cv2.cvtColor(dilated_line_mask, cv2.COLOR_GRAY2BGR)
    
    # Combine the output image with the dilated lines
    output_image = cv2.addWeighted(output_image, 1, dilated_lines_color, 1, 0)

    return line_mask, lines

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