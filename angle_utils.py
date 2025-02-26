import numpy as np

def angle_from_horizontal(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def angle_from_vertical(x1, y1, x2, y2):
    return 90 - angle_from_horizontal(x1, y1, x2, y2)
