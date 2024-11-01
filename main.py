import cv2
import numpy as np
import os
import math

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

# Map ad to the rectangle
def mapIm2Parallelogram(frame, image, paraPoints):
    frameProcessed = 0
    return frameProcessed

# Set the dimensions of the frame
height = 720
width = 1280


overlay_image = cv2.imread('image.png')

# Create a blank frame (all zeros means a black frame)
frame = np.zeros((height, width, 3), dtype=np.uint8)
pCornerLeftUpper = (500,100)
pCornerLeftLower = (100,500)

beginAdProcent = 0.6
endAdProcent = 0.8
angleOfAd = 0 # angle that ad makes with the perpendicular line to the plotted line in degrees

adBeginPoint, adEndPoint = getBeginEndCoords(beginAdProcent, endAdProcent, pCornerLeftLower, pCornerLeftUpper)
lengthLowerSide, angleLowerSide = getLengthAndAngle(adBeginPoint,adEndPoint)

print(f"length: {lengthLowerSide}     angle: {math.degrees(angleLowerSide)}")

adBeginUpperPoint, adEndUpperPoint = getParallelogramPoints(adBeginPoint, adEndPoint, lengthLowerSide, angleOfAd, color=(255, 255, 255), thickness=2)

paraArray = np.array([[adBeginUpperPoint[0], adBeginUpperPoint[1]], [adEndUpperPoint[0], adEndUpperPoint[1]], [adBeginPoint[0], adBeginPoint[1]], [adEndPoint[0], adEndPoint[1]]], dtype=np.float32)
overlay_h, overlay_w = overlay_image.shape[:2]
imageArray = np.array([[0, 0], [overlay_w, 0], [overlay_w, overlay_h], [0, overlay_h]], dtype=np.float32)

cv2.line(frame, pCornerLeftUpper, pCornerLeftLower, color=(0, 255, 0), thickness=2)
cv2.circle(frame, adBeginPoint, radius=5, color=(0,0,255))
cv2.circle(frame, adEndPoint, radius=5, color=(0,0,255))
cv2.circle(frame, adBeginUpperPoint, radius=5, color=(0,0,255))
cv2.circle(frame, adEndUpperPoint, radius=5, color=(0,0,255))





homography_matrix, _ = cv2.findHomography(imageArray, paraArray)
# Warp the overlay image to fit the current frame's src_points
warped_image = cv2.warpPerspective(overlay_image, homography_matrix, (frame.shape[1], frame.shape[0]))
# Create a mask from the warped image
mask = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
# Invert mask to apply to the original frame
mask_inv = cv2.bitwise_not(mask)
# Black-out the area of the overlay on the original frame
frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
# Add the warped overlay image
combined_frame = cv2.add(frame_bg, warped_image)




cv2.imshow("Frame", combined_frame)
while True:
    if cv2.waitKey(1) == 27:  # 27 is the ASCII code for the Escape key
        break

# Destroy all windows
cv2.destroyAllWindows()