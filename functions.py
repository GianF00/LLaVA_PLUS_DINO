from math import ceil
from matplotlib import pyplot as plt
import cv2
import numpy as np
import sklearn.metrics
import time
import replicate

def help_iou_more_coordinates(lis1,lis2):
    iou_list = []
    l1 = []
    l2 = []

    
    for i in range(0, len(lis1), 4):
        gt = lis1[i:i+4]
        pred = lis2[i:i+4]
        l1.append(gt)
        l2.append(pred)
    #    iou = intersection_over_union(gt, pred)
    #    iou_list.append(iou)

    #return iou_list
    return l1,l2

def iou(box1, box2):
    # Assuming box format is (x1, y1, x2, y2)
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou_value = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou_value

def intersection_over_union(gt, pred):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(gt[0], pred[0])
    yA = max(gt[1], pred[1])
    xB = min(gt[2], pred[2])
    yB = min(gt[3], pred[3])
    # if there is no overlap between predicted and ground-truth box
    if xB < xA or yB < yA:
        return 0.0
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    boxBArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold):
    GT, P = help_iou_more_coordinates(gt_boxes,pred_boxes)
    print(f"GT {GT}, P {P}")
    result = get_single_image_results(GT, P, iou_threshold)
    print(result['true_pos'], "\n")
    print(result['false_pos'], "\n")
    print(result['false_neg'])


    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    matched_gt_boxes = set()
    
    for pred_box in P:
        matched = False
        for gt_index, gt_box in enumerate(GT):
            if gt_index in matched_gt_boxes:
                continue  # Skip already matched ground truth boxes
            print("pred_boxes", pred_box, "\n", "gt box", gt_box)
            if iou(pred_box, gt_box) >= iou_threshold:
                tp += 1
                matched_gt_boxes.add(gt_index)
                matched = True
                break
        if not matched:
            fp += 1
    print(f"Matched GT Boxes: {matched_gt_boxes}\n")
    print("len ", len(GT) )
    fn = len(GT) - len(matched_gt_boxes)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"GT: {GT}")
    print(f"P: {P}")
    return precision, recall, tp, fp, fn, result


def drawing_boxes(LABELS, IMG):
    x,y,w,h = cv2.selectROI(LABELS, IMG, fromCenter=False, showCrosshair=True)
    return (x,y,w,h)


def calculate_predi_time(Version, input_data, price):
    client = replicate.Client()

    prediction = client.predictions.create(
        version=Version,
        input=input_data
    )
    print("Prediction created with details:", prediction)
    
    api_key = 'r8_X7BImFgGCsTYYNYiRsHNlipu0zf3pG12fITxR'  # Ensure this is securely handled

    prediction_id = prediction.id
    print(f"Prediction ID: {prediction_id}\n")
    print("type: ", type(prediction))

    while True:
        prediction_result = client.predictions.get(prediction_id)
        if prediction_result.status == 'succeeded':
            # Process the successful prediction
            break
        elif prediction_result.status == 'failed':
            # Handle the failure
            break
        else:
            # Wait some time before checking again
            time.sleep(5)

    # Check the status or result of your prediction
    if prediction_result.status == 'succeeded':
        result = prediction_result.metrics
        print(result, "\n")
        predic_time = prediction_result.metrics['predict_time']
        print(f"Predict Time: {predic_time} seconds")
    elif prediction_result.status == 'failed':
        error_message = prediction_result.error
        print(error_message)
        predic_time = None  # Explicitly setting to None in case of failure
    else:
        # If it's still running, you might want to check again later
        print("Prediction is still processing.")
        predic_time = None  # Explicitly setting to None if still processing

    if predic_time is not None:
        total_cost = predic_time * price
    else:
        total_cost = None  # Set to None or an appropriate value in case of failure or processing

    return total_cost, predic_time

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



#Länk: https://gist.github.com/sadjadasghari/dc066e3fb2e70162fbb838d4acb93ffc 
def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou

def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int) """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            IOU = iou(pred_box, gt_box)
            if IOU > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(IOU)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
