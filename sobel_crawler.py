import cv2
import numpy as np
from grayscale_utils import rescale_minmax, normalize_and_rescale
from angle_utils import angle_from_horizontal, angle_from_vertical
img = cv2.imread('8caea184401566c859a9f3d42a645fa9.jpg', cv2.IMREAD_GRAYSCALE)
#img = normalize_and_rescale(img)
vertical = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
horizontal = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

vertical = cv2.convertScaleAbs(vertical)
vertical_rescaled = rescale_minmax(vertical)
horizontal = cv2.convertScaleAbs(horizontal)

v_ret, v_thresh = cv2.threshold(vertical_rescaled, 127.5, 255, cv2.THRESH_TOZERO) 
lines_list =[]
lines = cv2.HoughLinesP(
            v_thresh, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=200, # Min number of votes for valid line
            minLineLength=5, # Min allowed length of line
            maxLineGap=20 # Max allowed gap between line for joining them
            )

new_image = np.zeros_like(img)
# Iterate over points
for points in lines:
    print(points)
      # Extracted points nested in the list
    x1,y1,x2,y2=points[0]
    # Draw the lines joing the points
    # On the original image
    cv2.line(new_image,(x1,y1),(x2,y2),(255),1)
    # Maintain a simples lookup list for points
    lines_list.append([(x1,y1),(x2,y2)])
#raise
#print(v_thresh.max())
#raise
#cv2.imshow('Verticals', vertical)
#cv2.imshow('Verticals RESCALED', vertical_rescaled)
cv2.imshow('Verticals THRESH', v_thresh)
cv2.imshow('Vertical Lines', new_image)
#cv2.imshow('Horizontals', horizontal)
#cv2.imshow('Combined', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()