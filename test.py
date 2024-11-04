import cv2
import numpy as np
from scipy.spatial import KDTree

def load_templates(template_paths):
    templates = {}
    for label, path in template_paths.items():
        # Load template and convert to grayscale immediately
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Failed to load template: {label} from path: {path}")
        else:
            print(f"Loaded template: {label} with shape: {template.shape}")
        templates[label] = template  # Store template regardless of success for consistent access
    return templates

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
    grouped_points = group_close_points(points, distance_threshold=25)
    flat_points = np.array([np.mean(group, axis=0) for group in grouped_points])
    isolated_points = filter_isolated_points(flat_points, distance_threshold=30)

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

def classify_corners(frame, corners, templates, threshold=5):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Create a FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for corner in corners:
        x, y = int(corner[1]), int(corner[0])  # Convert to (x, y) for drawing
        
        # Define the region of interest around the corner
        h, w = 100, 100  # Size of the region to match
        roi = frame[max(0, y-h):min(frame.shape[0], y+h), max(0, x-w):min(frame.shape[1], x+w)]  # Extract region of interest

        # Convert ROI to grayscale for SIFT
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect SIFT features in the ROI
        kp_roi, des_roi = sift.detectAndCompute(roi_gray, None)
        best_match = None
        best_match_count = 0

        # Match the ROI against each template
        for label, template in templates.items():
            if template is not None:  # Ensure template is loaded
                kp_template, des_template = sift.detectAndCompute(template, None)
                if des_template is not None and des_roi is not None:
                    # Perform matching
                    matches = flann.knnMatch(des_roi, des_template, k=2)

                    # Apply Lowe's ratio test
                    good_matches = [m for m, n in matches if m.distance < 0.9 * n.distance]

                    # Check for a good match
                    if len(good_matches) > best_match_count:
                        best_match_count = len(good_matches)
                        best_match = label

        # If the best match is above the threshold, draw it on the frame
        if best_match and best_match_count >= threshold:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw in green for classification
            cv2.putText(frame, best_match, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# Load your templates here
template_paths = {
    'Corner A': 'templates/bottomleftin.png',
    'Corner B': 'templates/bottomleftout.png',
    'Corner C': 'templates/bottomrightin.png',
    'Corner D': 'templates/bottomleftout.png',
    'Corner E': 'templates/topleftout.png',
    'Corner F': 'templates/toprightin.png',
    'Corner G': 'templates/topleftin.png',
    'Corner H': 'templates/toprightout.png',
}
templates = load_templates(template_paths)

# Video processing setup
video_path = 'video2.mp4'
cap = cv2.VideoCapture(video_path)
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_segment = int(fps * 2.5)
total_segments = 8
total_frames = frames_per_segment * total_segments

while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    corners = detect_corners(frame)  # Get detected corners
    frame = classify_corners(frame, corners, templates)  # Classify corners

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
