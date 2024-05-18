from math import ceil
from matplotlib import pyplot as plt
import cv2
import numpy as np
import sklearn.metrics
import time
import replicate



def compute_precision_recall(yTrue, predScores, thresholds):
    precisions = []
    recalls = []
    # loop over each threshold from 0.6 to 0.9
    for threshold in thresholds:
        # yPred is dog if prediction score greater than threshold
        # else cat if prediction score less than threshold
        yPred = [
            #"cat" if score >= threshold else "dog"
            "dog" if score >= threshold else "cat"
            for score in predScores
        ]
  
        # compute precision and recall for each threshold
        precision = sklearn.metrics.precision_score(y_true=yTrue,
            y_pred=yPred, pos_label="dog")
        recall = sklearn.metrics.recall_score(y_true=yTrue,
            y_pred=yPred, pos_label="dog")
  
        # append precision and recall for each threshold to
        # precisions and recalls list
        precisions.append(np.round(precision, 3))
        recalls.append(np.round(recall, 3))
    # return them to calling function
    return precisions, recalls


def pr_compute(groutrut_values, predic_values):
    # define thresholds from 0.2 to 0.50 with step size of 0.05
    thresholds = np.arange(start=0.6, stop=0.9, step=0.05)
    # call the compute_precision_recall function
    precisions, recalls = compute_precision_recall(
        yTrue=groutrut_values, predScores=predic_values,
        thresholds=thresholds,
    )
 
    # return the precisions and recalls
    return (precisions, recalls,thresholds)


def plot_pr_curve(precisions, recalls):
    # plots the precision recall values for each threshold
    # and save the graph to disk
    plt.plot(recalls, precisions, linewidth=4, color="red")
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    #plt.savefig(path)
    plt.show()


def drawing_boxes(LABELS, IMG):
    x,y,w,h = cv2.selectROI(LABELS, IMG, fromCenter=False, showCrosshair=True)
    return (x,y,w,h)


def calculate_predi_time(Version, input_data,price):
    client = replicate.Client()

    prediction = client.predictions.create(
        version=Version,
        input=input_data
    )
    print("Prediction created with details:", prediction)
    # Replace 'your_api_key_here' with your actual Replicate API key
    api_key = 'r8_X7BImFgGCsTYYNYiRsHNlipu0zf3pG12fITxR'

    prediction_id = prediction.id
    print(f"Prediction ID: {prediction_id}\n")
    print("type: ", type(prediction))
    # time.sleep(7) 
    # prediction_result = client.predictions.get(prediction_id)
    # print(prediction_result)


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
    else:
        # If it's still running, you might want to check again later
        print("Prediction is still processing.")

    total_cost = predic_time * price
    return total_cost,predic_time


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
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou