import cv2
import numpy as np
from scipy.spatial import KDTree

def detect_corners(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Harris Corner Detection
    gray_float = np.float32(blurred)
    corners_harris = cv2.cornerHarris(gray_float, blockSize=5, ksize=5, k=0.04)



    # Dilate the corner image to mark the corners
    corners_harris = cv2.dilate(corners_harris, None)

    # Threshold for an optimal value
    threshold = 0.005 * corners_harris.max()
    
    # Get coordinates of corners above the threshold
    corner_coords = np.argwhere(corners_harris > threshold)
    
    # Create a list of points for clustering
    points = np.array(corner_coords)

    # Group points that are close together using KD-Tree
    grouped_points = group_close_points(points, distance_threshold=15)

    # Draw the centroids of each cluster
    for group in grouped_points:
        if len(group) > 0:
            centroid = np.mean(group, axis=0).astype(int)
            cv2.circle(frame, (centroid[1], centroid[0]), 5, (0, 0, 255), -1)  # Draw centroid in red

    
    return frame


def group_close_points(points, distance_threshold):
    """Group points that are close together based on a distance threshold using KD-Tree."""
    # Create a KD-Tree from the points
    tree = KDTree(points)
    visited = set()
    grouped = []

    for i, point in enumerate(points):
        if i in visited:
            continue
        
        # Find all points within the distance threshold
        indices = tree.query_ball_point(point, r=distance_threshold)

        # Create a cluster of points found
        cluster = points[indices]
        grouped.append(cluster)
        
        # Mark these points as visited
        visited.update(indices)

    return grouped

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
    frame = detect_corners(frame)
    
    cv2.imshow('frame',frame)
    cv2.waitKey(25)
    frame_count += 1 
#cap.release()
#cv2.destroyAllWindows()