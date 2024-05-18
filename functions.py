from matplotlib import pyplot as plt
import cv2
import numpy as np
import re
from math import ceil


def drawing_boxes(LABELS, IMG):
    x,y,w,h = cv2.selectROI(LABELS, IMG, fromCenter=False, showCrosshair=True)
    return (x,y,w,h)


def extract_coordinates_gpt4(response_text):
    # This pattern matches coordinates for both listed and inline formats
    # pattern = r'- x0: (\d+)\s+- y0: (\d+)\s+- x1: (\d+)\s+- y1: (\d+)|Coordinates: \((\d+), (\d+), (\d+), (\d+)\)'
    # matches = re.findall(pattern, response_text)
    
    # # Extracting the coordinates and converting them to integers
    # coordinates = []
    # for match in matches:
    #     coords = match if match[0] != '' else match[4:]
    #     coordinates.append(tuple(map(int, coords)))

    # This pattern matches coordinates that are comma-separated
    #return coordinates

    pattern = r'(\d+),\s*(\d+),\s*(\d+),\s*(\d+)'
    matches = re.findall(pattern, response_text)

  # Convert the string matches to lists of integers
    coordinates_list = [list(map(int, match)) for match in matches]
    
    return coordinates_list




    
def calculate_image_tokens(width: int, height: int):
    if width > 2048 or height > 2048:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width, height = 2048, int(2048 / aspect_ratio)
        else:
            width, height = int(2048 * aspect_ratio), 2048
            
    if width >= height and height > 768:
        width, height = int((768 / height) * width), 768
    elif height > width and width > 768:
        width, height = 768, int((768 / width) * height)

    tiles_width = ceil(width / 512)
    tiles_height = ceil(height / 512)
    total_tokens = 85 + 170 * (tiles_width * tiles_height)
    
    return total_tokens

def calculate_cost(width, height):
    num_tokens = calculate_image_tokens(width, height)
    cost_per_token = 10.00 / 1_000_000  # 10 dollar per 1 million tokens
    cost = num_tokens * cost_per_token
    return cost




def calculate_iou(boxA, boxB):
    # Calculate the (x, y) coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA + 1)
    interHeight = max(0, yB - yA + 1)
    interArea = interWidth * interHeight

    # Compute the area of both the prediction and ground truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

