import cv2
import numpy as np
from grayscale_utils import rescale_minmax, normalize_and_rescale
from line_utils import get_horizontal_angle, get_vertical_angle, higher_point, filter_angles
import line_utils
img = cv2.imread('e5bceb111df2dcf04bff925d569d0f99.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
print(img.dtype, img.shape)
_, img = cv2.threshold(img, img.mean(), 255, cv2.THRESH_TOZERO) 
print(img.dtype, img.shape)
#img = normalize_and_rescale(img)
vertical = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
horizontal = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

vertical = cv2.convertScaleAbs(vertical)
vertical_rescaled = rescale_minmax(vertical)
horizontal = cv2.convertScaleAbs(horizontal)

v_ret, v_thresh = cv2.threshold(vertical_rescaled, vertical_rescaled.mean(), 255, cv2.THRESH_TOZERO) 
lines_list =[]
lines = cv2.HoughLinesP(
            v_thresh, # Input edge image
            1, # Distance resolution in pixels
            np.pi/360, # Angle resolution in radians
            threshold=250, # Min number of votes for valid line
            minLineLength=5, # Min allowed length of line
            maxLineGap=2 # Max allowed gap between line for joining them
            )

def process_point(points:np.ndarray):
    """
    Returns the sorted 2 points in descending order
    """
    p1 = points[0:2]
    p2 = points[2:4]
    x1, y1 = p1
    x2, y2 = p2
    ind = higher_point(x1, y1, x2, y2)
    if ind == 0:
        return x1, y1, x2, y2
    else:
        return x2, y2, x1, y1

angles = np.empty(len(lines), dtype=np.float32)
# Iterate over points
for i, points in enumerate(lines):
    #(x1, y1), (x2, y2) = points[0]
    x1, y1, x2, y2 = process_point(points[0])
    angles[i] = np.abs(get_vertical_angle(x1, y1, x2, y2))

angles = np.array(angles)
indices = filter_angles(angles, thresh=5.0)
filtered_lines = lines[indices]
filtered_angles = angles[indices]
# Unpack filtered lines to reduce repeated 0-indice accesses
filtered_lines = filtered_lines[:, 0]

#vertical_lines = np.zeros_like(img)
#for point in filtered_lines:
#    x1, y1, x2, y2 = point
#    cv2.line(vertical_lines,(x1,y1),(x2,y2),(255),1)
#    #lines_list.append([(x1,y1),(x2,y2)])

# Extract the x1 into a list for vertical clustering
x_points = filtered_lines[:, 0]
group = line_utils.cluster()
group.predict(x_points, filtered_angles)

canvas = group.vis(filtered_lines, canvas=np.zeros((h, w, 3), dtype=np.uint8)+63)
cv2.imshow('Labelled Clusters', canvas)
overlayed = cv2.addWeighted(canvas, 0.8, cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), 0.2, 0)
cv2.imshow('Overlaid Clusters', overlayed)
v_overlayed = cv2.addWeighted(canvas, 0.8, cv2.cvtColor(v_thresh, cv2.COLOR_GRAY2RGB), 0.2, 0)
cv2.imshow('Overlaid Edges', v_overlayed)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(group.n_clusters)
#line_utils.cluster(x_points)
#print(x_points)
raise
#raise
#print(v_thresh.max())
#raise
#cv2.imshow('Verticals', vertical)
#cv2.imshow('Verticals RESCALED', vertical_rescaled)
cv2.imshow('Verticals THRESH', v_thresh)
cv2.imshow('Vertical Lines', vertical_lines)
#cv2.imshow('Horizontals', horizontal)
#cv2.imshow('Combined', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()