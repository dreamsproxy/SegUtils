import numpy as np

def get_horizontal_angle(x1, y1, x2, y2):
    return np.float32(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

def get_vertical_angle(x1, y1, x2, y2):
    return np.float32(90 - get_horizontal_angle(x1, y1, x2, y2))

def higher_point(x1, y1, x2, y2):
    """
    Returns the indice of the point, either 0 or 1
    """
    return 0 if y1 < y2 else 1

def filter_angles(angles:np.ndarray[np.float32], thresh:float=1.0):
    """
    Filters the angles given a threshold.
    Returns indices of angles within threshold
    """
    return list(np.where(angles <= thresh)[0])

class cluster:
    def __init__(self):
        from sklearn.cluster import DBSCAN
        from collections import OrderedDict
        self.dbscan = DBSCAN(eps=4.0, min_samples=1)
        self.labels = None

    def predict(self, axis_points, angles):
        #axis_points = axis_points.reshape(-1, 1)
        data = np.array([axis_points, angles]).T
        #print(data.shape)
        #raise
        """
        Groups the lines by it's x axes
        """
        # Apply DBSCAN clustering
        self.labels = self.dbscan.fit_predict(data)
        self.n_clusters = len(set(self.labels))

    def vis(self, lines=np.ndarray, canvas:np.ndarray = None):
        import cv2
        colors = []
        colors.append(np.linspace(0, 255, num=self.n_clusters, endpoint=True, dtype=np.uint8))
        colors.append(np.ones(shape=(self.n_clusters))*127)
        colors.append(np.linspace(255, 0, num=self.n_clusters, endpoint=True, dtype=np.uint8))
        colors = np.array(colors)
        colors = colors.reshape((self.n_clusters, 3))
        label_colors = {}
        for i, label in enumerate(sorted(list(set(self.labels)))):
            label_colors[label] = tuple([int(j) for j in colors[i]])
        
        for i, l in enumerate(self.labels):
            x1, y1, x2, y2 = lines[i]
            print(x1, y1, x2, y2)
            print(label_colors[l])
            cv2.line(canvas, (x1,y1), (x2,y2), label_colors[l], 1)
        
        return canvas