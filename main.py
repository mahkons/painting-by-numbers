import argparse
import cv2
import numpy as np
import random

from preprocessing import denoise, sharpen
from cluster import cluster_image, recolour
from contours import draw_contours
from smooth import smooth_image

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output-image", type=str, default=None, required=False)
    parser.add_argument("--output-contours", type=str, default=None, required=False)
    return parser


if __name__ == "__main__":
    random.seed(239)
    np.random.seed(239)
    args = create_parser().parse_args()
    input_image = cv2.imread(args.image)

    print("Preprocessing")
    w, h, _ = input_image.shape
    resized_image = cv2.resize(input_image, (h * 1500 // max(h, w), w * 1500 // max(h, w)))
    denoised_image = denoise(resized_image, 3, 3)
    sharpened_image = sharpen(denoised_image, kernel_size=(5, 5), sharpness=0.0, iterations=3)

    print("Clustering")
    labels, centers = cluster_image(sharpened_image, 8)
    print("Smoothing")
    smoothed_labels = smooth_image(labels, 3 + np.arange(3), 1)

    print("Drawing contours")
    output_contours, output_labels = draw_contours(centers[smoothed_labels], smoothed_labels, min_area=100, min_radius=8, gaps_smooth=5)
    if np.min(output_labels) < 0:
        print("Warning: Not all pixels painted")

    # watershed for uncoloured parts. not cool?
    #  output_labels += 1
    #  cv2.watershed(sharpened_image, output_labels)
    #  output_labels -= 1

    output_image = recolour(sharpened_image, output_labels)
    output_contours = cv2.cvtColor(output_contours, cv2.COLOR_GRAY2BGR)

    # save and show
    if args.output_image is not None:
        cv2.imwrite(args.output_image, output_image)
    if args.output_contours is not None:
        cv2.imwrite(args.output_contours, output_contours)

    window_name = "just image"
    stacked_image = np.hstack([output_image, output_contours])

    w, h, _ = stacked_image.shape
    stacked_image = cv2.resize(stacked_image, (h * 1500 // max(w, h), w * 1500 // max(w, h)))

    cv2.imshow(window_name, stacked_image)
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0 and cv2.waitKey(delay=100):
        pass
    cv2.destroyWindow(window_name)
