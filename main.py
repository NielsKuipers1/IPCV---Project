import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import os

def detect_corners(frame, boundaries, lines, max_distance=5):
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
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            # Check distance to both ends of the lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate distances to both endpoints
                distance_to_start = np.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
                distance_to_end = np.sqrt((cx - x2) ** 2 + (cy - y2) ** 2)
                
                if distance_to_start <= max_distance or distance_to_end <= max_distance:
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    intersections.append(contour)
                    break  # No need to check other lines for this contour

    return frame, intersections


def detect_lines(frame, boundaries, min_line_length=70):
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

    return frame, filtered_lines


def tennis_court_calibration_from_dimensions():
    # Standard tennis court measurements (in m)
    COURT_LENGTH = 23.77                    # baseline to baseline
    COURT_WIDTH = 10.97                     # doubles sideline to sideline
    SECOND_LINE = 1.37                      # distance between base line and 2nd line
    
    # Define 3D points of tennis court key points (in world coordinates)
    points3d = np.array([
        # Baseline corners
        [0, 0, 0],                          # Bottom left
        [COURT_WIDTH, 0, 0],                # Bottom right
        [0, COURT_LENGTH, 0],               # Top left
        [COURT_WIDTH, COURT_LENGTH, 0],     # Top right

        # 2nd line corners
        [SECOND_LINE, 0, 0],
        [COURT_WIDTH-SECOND_LINE, 0, 0],
        [SECOND_LINE, COURT_LENGTH, 0],
        [COURT_WIDTH-SECOND_LINE, COURT_LENGTH, 0]

    ], dtype=np.float32)

    return points3d


def get_court_points(image):
    """
    Allow user to manually select court points in the image
    Returns array of 2D points corresponding to court_points_3d
    """
    points2d = []
    
    # Create a window to display the image
    window_name = 'Select Court Points'
    img_display = image.copy()
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points2d.append([x, y])
            # Draw circle at selected point
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, img_display)
    
    cv2.imshow(window_name, img_display)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Click on the following points in order:")
    print("1. Bottom left baseline corner")
    print("2. Bottom right baseline corner")
    print("3. Top left baseline corner")
    print("4. Top right baseline corner")
    print("5. Bottom left 2nd corner")
    print("6. Bottom right 2nd corner")
    print("7. Top left 2nd corner")
    print("8. Top right 2nd corner")
    
    while len(points2d) < len(points3d):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return np.array(points2d, dtype=np.float32)


def get_banner_points(image, bannerCorners):
    """
    Allow user to manually select banner points in the image
    """
    points2d = []
    
    # Create a window to display the image
    window_name = 'Select banner points'
    img_display = image.copy()
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points2d.append([x, y])
            # Draw circle at selected point
            cv2.circle(img_display, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow(window_name, img_display)
    
    cv2.imshow(window_name, img_display)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Click on the following points in order:")
    print("1. Bottom left baseline corner")
    print("2. Bottom right baseline corner")
    print("3. Top left baseline corner")
    print("4. Top right baseline corner")
    
    while len(points2d) < bannerCorners:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return np.array(points2d, dtype=np.float32)


def addHorizontalBanner(frame, adImage, points2d, outputSize=(800, 600)):
    """
    Add horizontal banner to frame using perspective transform
    """
    # Get frame dimensions
    frameH, frameW = frame.shape[:2]
    
    # Resize ad banner to appropriate width
    bannerWidth = frameW
    aspectRatio = adImage.shape[1] / adImage.shape[0]
    bannerHeight = int(bannerWidth / aspectRatio)
    adResized = cv2.resize(adImage, (bannerWidth, bannerHeight))
    
    # Define source points (rectangle for the ad)
    src_points = np.array([
        [0, 0],                    # top left
        [bannerWidth-1, 0],        # top right
        [bannerWidth-1, bannerHeight-1],  # bottom right
        [0, bannerHeight-1]        # bottom left
    ], dtype=np.float32)
    
    # Reorder destination points to match orientation
    dst_points = np.array([
        points2d[2],  # bottom right
        points2d[3],  # bottom left
        points2d[1],  # top right
        points2d[0]   # top left   
    ], dtype=np.float32)
    
    # Calculate perspective transform
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Warp ad banner
    warpedBanner = cv2.warpPerspective(adResized, matrix, (frameW, frameH))
    
    # Create mask for the warped banner
    mask = np.zeros((frameH, frameW), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_points.astype(np.int32), 255)
    
    # Blend the banner with the original frame
    mask3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = frame * (1 - mask3d) + warpedBanner * mask3d
    
    return result.astype(np.uint8)

# Main video processing loop
def process_video(video_path, ad_path):
    cap = cv2.VideoCapture(video_path)
    adImage = cv2.imread(ad_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect lines and corners
        boundaries = [([170, 170, 100], [255, 255, 255])]
        output_frame, detected_lines = detect_lines(frame, boundaries)
        output_frame, intersections = detect_corners(frame, boundaries, detected_lines)
        
        # Get court points (you might want to modify this to automatically detect points)
        points2d = get_court_points(frame)
        
        # Add banner
        result_frame = addHorizontalBanner(frame, adImage, points2d)
        
        # Write frame
        out.write(result_frame)
        
        # Display result
        cv2.imshow('Video with Banner', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


#====================================================================
# # initiate video 
# video_path = 'video.mp4'
# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)

# # defenition of parameters
# frame_count = 0
# frames_per_segment = int(fps * 2.5)  
# total_segments = 8
# total_frames = frames_per_segment * total_segments
# boundaries = [([170, 170, 100], [255, 255, 255])] # define threshold to only use the white lines 

# while frame_count < total_frames:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # Step 1: corner and line detection (localize calibration points)
#     output_frame, detected_lines = detect_lines(frame, boundaries)
#     output_frame, intersections = detect_corners(frame,boundaries,detected_lines)
    
#     # Step 2 Intrinsic camera calibration using the known 3D positions of reference objects
    
#     # Step 3 External camera calibration: the pose of the camera relative to the 3D reference objects

#     # Step 4 Tracking of 2D points and/or lines in the movie.

#     # Step 5 Based on these tracked points and/or lines, camera pose tracking during the movie.

#     # Step 6 Projection of a virtual banner which has a rectangular shape in the real world and located near a court line

#     cv2.imshow("Detected Lines", output_frame)
#     cv2.waitKey(25)
#     frame_count += 1 
# cap.release()
# cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Load image
    image = cv2.imread('court.png')
    ad = cv2.imread('image.png')

    # Get image size
    imageSize = (image.shape[1], image.shape[0])
    adSize = (ad.shape[1], ad.shape[0])

    # Define points in court
    points3d = tennis_court_calibration_from_dimensions()

    # Get 2D points from user input
    points2dCourt = get_court_points(image)
    
    if len(points2dCourt) < len(points3d):
        raise ValueError("Not enough points selected")

    # Test addHorizontalBanner 
    bannerCorners = 4
    points2dBanner = get_banner_points(image, bannerCorners)
    test = addHorizontalBanner(image, ad, points2dBanner, outputSize=(800, 600))
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Test')
    plt.show()
