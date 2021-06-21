import cv2
import numpy as np

def denoise(image, h, h_color):
    return cv2.fastNlMeansDenoisingColored(image, h=h, hColor=h_color)

def sharpen(image, kernel_size=(3, 3), sigma=1., sharpness=0.1, iterations=1):
    for i in range(iterations):
        blur = cv2.GaussianBlur(image, kernel_size, sigma) - image
        image = image - sharpness * blur
    return np.clip(image, 0, 255).astype(np.uint8)
