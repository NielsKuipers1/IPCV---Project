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
    # Standard tennis court measudrements (in m)
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


def calibrate_from_points(points3d, points2d, imageSize):
    """
    Calculate intrinsic camera parameters from corresponding 3D-2D points
    """
    # Initial guess for camera matrix
    focal_length = 1000  # Initial guess
    center = (imageSize[1]/2, imageSize[0]/2)
    cameraMatrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float32)
    
    # Initial guess for distortion coefficients
    distCoeffs = np.zeros((4,1))

    # Find camera parameters
    success, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        [points3d], [points2d], imageSize, cameraMatrix, distCoeffs,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    
    return cameraMatrix, distCoeffs, rvecs, tvecs 


def warpToTopDownView(image, points3d, points2d, cameraMatrix, distCoeffs, rvecs, tvecs, output_size=(800, 600)):
    """
    Warps the image to a top-down view using calculated camera parameters.
    """
    # Project the 3D points of the court to the image plane
    projectedPoints, _ = cv2.projectPoints(points3d, rvecs[0], tvecs[0], cameraMatrix, distCoeffs)
    projectedPoints = projectedPoints.reshape(-1, 2)

    alpha = 50

    # Define the destination points in the overhead view (top-down)
    dstPoints = np.array([
        [0, 0],                                      # Bottom-left
        [output_size[0], 0],                         # Bottom-right
        [0, output_size[1]],                         # Top-left
        [output_size[0], output_size[1]],            # Top-right
        [alpha, 0],                                  # Bottom-left 2nd
        [output_size[0]-alpha, 0],                   # Bottom-right 2nd
        [alpha, output_size[1]],                     # Top-left 2nd
        [output_size[0]-alpha, output_size[1]]       # Top-right 2nd
    ], dtype=np.float32)

    # Compute the homography from the detected points to the top-down view
    homographyMatrix, _ = cv2.findHomography(points2d, dstPoints)

    # Warp the image using the homography matrix to get the top-down view
    topDownView = cv2.warpPerspective(image, homographyMatrix, output_size)

    return topDownView, dstPoints


def warpBackToOriginalView(topDownView, points2d, dstPoints, outputSize, originalImageSize):
    """
    Warps the top-down view back to the original perspective.
    """
    # Calculate the homography matrix from top-down to original perspective
    homographyMatrix, _ = cv2.findHomography(dstPoints, points2d)
    
    # Warp the top-down image back to the original perspective
    originalPerspectiveView = cv2.warpPerspective(topDownView, homographyMatrix, originalImageSize)
    
    return originalPerspectiveView


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

    # Calibrate camera
    cameraMatrix, distCoeffs, rvecs, tvecs = calibrate_from_points(
        points3d, points2dCourt, imageSize)

    # Display results
    print("\nCamera Matrix:")
    print(cameraMatrix)
    print("\nDistortion Coefficients:")
    print(distCoeffs.ravel())
    print("\nRotation matrix:")
    print(rvecs)
    print("\nTranslation matrix:")
    print(tvecs)

    # Generate the top-down view of the court
    topDownView, dstPoints = warpToTopDownView(image, points3d, points2dCourt, cameraMatrix, distCoeffs, rvecs, tvecs)

    # Display the top-down view
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(topDownView, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Top-Down View of the Tennis Court')
    plt.show()

    adFlipped = cv2.flip(ad, 0)
    topImageResized = cv2.resize(adFlipped, (topDownView.shape[1], adFlipped.shape[0]))
    combinedImage = np.vstack((topDownView, topImageResized))

    # Plot again
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(combinedImage, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Top-Down View of the Tennis Court')
    plt.show()

    # back to original view
    OGView = warpBackToOriginalView(combinedImage, points2dCourt, dstPoints, (800,600), image.shape[1::-1])

    # Display the top-down view
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(OGView, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('OG View of the Tennis Court with ad')
    plt.show()
