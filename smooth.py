import numpy as np
import cv2
from scipy.interpolate import splprep, splev
import random


def smooth_morphology(labels, morph_type, kernel=np.ones((3, 3)), morth_iterations=1, inplace=False):
    smoothed = labels if inplace else np.copy(labels)
    x = list(range(np.max(labels) + 1))
    random.Random(239 + kernel.sum() + morth_iterations).shuffle(x)
    for i in x:
        morph = cv2.morphologyEx((smoothed == i).astype(np.uint8), morph_type, kernel, iterations=morth_iterations)
        smoothed[morph==1] = i
    return smoothed

def smooth_image(labels, kernel_sizes, morth_iterations):
    smoothed = np.copy(labels)
    for ksize in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)) 
        smoothed = smooth_morphology(smoothed, cv2.MORPH_CLOSE, kernel, morth_iterations, True)
    return smoothed

# poorly works
def smooth_contour(c, smoothness):
    if len(c) <= 10:
        return c
    nc = c[:, 0, :]
    nc = np.concatenate([nc, nc[0][None]], axis=0).T

    tck, u = splprep(nc, s=smoothness, per=1) 
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.stack([x_new, y_new], axis=1)[:, None, :].astype(np.int32)
