import cv2
import numpy as np

from smooth import smooth_contour

def draw_border(image):
    w, h = image.shape
    bordered = np.copy(image)
    bordered[[0, w - 1], :] = 255
    bordered[:, [0, h - 1]] = 255
    return bordered

def close_gaps(image, gaps_smooth):
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((gaps_smooth, gaps_smooth), np.uint8), iterations=1)
    return closed


def _make_contours(edges, labels, min_area, min_radius, gaps_smooth, remove, set_digits):
    c_line = np.ones(labels.shape, dtype=np.uint8)*255
    c_image = -np.ones(labels.shape, dtype=np.int32)

    edges = draw_border(edges)
    edges[edges > 0] = 255
    edges = close_gaps(edges, gaps_smooth)
    contour_list, hierarchy = cv2.findContours(edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        
    area = np.zeros(len(contour_list))
    for j, c in enumerate(contour_list):
        area[j] += cv2.contourArea(c)
        if (p := hierarchy[0, j, 3]) != -1:
            area[p] -= area[j]

    c_ids = -np.ones(labels.shape, dtype=np.int32)
    all_contours = np.ones(labels.shape, dtype=np.uint8) * 255
    for j, c in enumerate(contour_list):
        if area[j] >= min_area and remove:
            cv2.drawContours(c_ids, [c], contourIdx=-1, color=j, thickness=-1)
            cv2.drawContours(all_contours, [c], contourIdx=-1, color=0, thickness=1)

    c_map = [[] for _ in range(len(contour_list))]
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if (ind := c_ids[x][y]) != -1:
                c_map[ind].append((x, y))

    far_point = list()
    dist = cv2.distanceTransform(all_contours, cv2.DIST_L2, 3)
    for positions in c_map:
        fp = None
        for p in positions:
            if fp is None or dist[p] > dist[fp]:
                fp = p
        if fp is not None and dist[fp] < min_radius and remove:
            fp = None
        far_point.append(fp)

    for j, c in enumerate(contour_list):
        if far_point[j] is not None:
            cv2.drawContours(c_line, [c], contourIdx=-1, color=0, thickness=1)
            most_freq = int(np.bincount(labels[tuple(zip(*c_map[j]))]).argmax())
            cv2.drawContours(c_image, [c], contourIdx=-1, color=most_freq, thickness=-1)
            digit_top_left = (far_point[j][1] - int(min_radius * 0.8), far_point[j][0] + int(min_radius * 0.8))
            if set_digits:
                cv2.putText(c_line, "{}".format(most_freq + 1), digit_top_left,
                        cv2.FONT_HERSHEY_PLAIN, fontScale=min_radius/8., color=0, thickness=1)
            
    return c_line, c_image

def draw_contours(image, labels, min_area, min_radius, gaps_smooth):
    edges = cv2.Canny(image, 0, 0, L2gradient=True)
    edges, new_labels = _make_contours(edges, labels, min_area, min_radius, gaps_smooth, remove=True, set_digits=True)
    return edges, new_labels
