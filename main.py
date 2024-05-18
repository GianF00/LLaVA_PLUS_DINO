import time 
from matplotlib import pyplot as plt
import numpy
import supervision as sv
import cv2 
import numpy as np
import psutil
import nltk
import json
import os
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score
#nltk.download('punkt')  # Ensure the tokenizer is downloaded
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tabulate import tabulate
from functions import calculate_iou, drawing_boxes, calculate_IOU, calculate_predi_time
import replicate
import matplotlib.patches as patches
from PIL import Image


image = "ordning3.jpg"
start = time.time()
f = open('YOLO_data.txt', 'a')

input_image = open(f"{image}", "rb")
input_data={
    "nms": 0.05,
    "conf": 0.37,
    "tsize": 640,
    "model_name": "yolox-x",
    "input_image": input_image,
    "return_json": True
}
output = replicate.run(
    "daanelson/yolox:ae0d70cebf6afb2ac4f5e4375eb599c178238b312c8325a9a114827ba869e3e9",
    input=input_data
)
print(output)

end = time.time()

# Remove the extra quotes and parse the JSON string
json_str = output['json_str'].strip('"')
detections = json.loads(json_str.replace('\'', '"'))  # Replace single quotes with double quotes to form valid JSON

###=========== draw the ground truth =====================##############
image_path = image
image = cv2.imread(image_path)
numOfTimes = 10
title = "objects"
coordinates = []
for _ in range(numOfTimes):
    x_val, y_val, w_val, h_val = drawing_boxes(title,image)
    coordinates.append([x_val,y_val, (x_val + w_val), (y_val+h_val)])
    #print("Num of iter: ", x)

##Add two list together:
# result = coordinates[0] + coordinates[1]
# print("\n",result)

result_gt = []
sz = len(coordinates)
for x in range(sz):
    #merged_result = []

    # Add each element of list_float to itself and append to merged_result
    # for num in list_float:
    #     merged_result.append(num + num)

    result_gt.extend(coordinates[x])
    #result_gt.append(coordinates[x])

    # Append the merged_result to mrgd_list_float
    # mrgd_lis_float.append(merged_result)

print("\nresult ground truth bounding box",result_gt)


#selectROI USED TO DRAW THE BOUNDING BOX WITH THE HELP OF CV2 AND OBTAIN THE COORDINATES 
#x,y,w,h = cv2.selectROI("cat", img, fromCenter=False, showCrosshair=True)

for coord in coordinates:
    temp = cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 2)

cv2.imshow('Detections', image)
cv2.imwrite("yolotest3_bboxes.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
###=========== end draw the ground truth =====================##############


###=========== drawing the predic ground truth =====================##############
image = cv2.imread("yolotest3_bboxes.jpg")
pred_bboxes = []
if image is not None:
    # Loop through the detections and draw each bounding box
    for det_key, det in detections.items():
        x0 = int(det['x0'])
        y0 = int(det['y0'])
        x1 = int(det['x1'])
        y1 = int(det['y1'])
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        label = f"{det['cls']}: {det['score']:.2f}"
        #cv2.putText(image, (x0, y0 - 10), label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.putText(image, f'{label}', (x0, y0  - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        pred_bboxes.append([x0, y0, x1, y1])

   # Display the result
    cv2.imshow('Detections', image)
    cv2.imwrite("yolotest3_ordning3_2.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Image could not be loaded.")
###=========== drawing the predic ground truth =====================##############
####============== CALCULATING THE PREDICTION TIME  =========####
Total_cost, Predic_time = calculate_predi_time("ae0d70cebf6afb2ac4f5e4375eb599c178238b312c8325a9a114827ba869e3e9",input_data,0.000225) 
####============== END OF CALCULATING THE PREDICTION TIME  =========####

###============= CALCULATING THE PRECISION, RECALL, TP, FP, FN =====####
#pr_boxes = [pred_bboxes[i:i+4] for i in range(0, len(pred_bboxes), 4)]
print("Predicted bboxes",pred_bboxes, "\n")
true_positives = 0
false_positives = 0
## Byt iou tröskelvärde mellan 20, 50 och 60%
iou_threshold = 0.50
matches = []
# matched_ground_truth = []  # Lista för att hålla reda på vilka GT-boxar som matchats
gt_matched = set()  # För att hålla reda på vilka GT-boxar som matchats
p_bboxes = []
# Calculate True Positives and False Positives
for p_idx, p_box in enumerate(pred_bboxes):
    match_found = False
    for gt_idx, gt_box in enumerate(coordinates):
        iou = calculate_iou(p_box, gt_box)
        if iou >= iou_threshold:
            true_positives += 1
            gt_matched.add(gt_idx)
            match_found = True
            p_bboxes.append(p_box)
            break
    if not match_found:
        false_positives += 1

# Calculate False Negatives
false_negatives = len(coordinates) - len(gt_matched)
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
print("p_bboxes: ", p_bboxes, "\n")
for coord1 in p_bboxes:
    cv2.rectangle(image, (coord1[0], coord1[1]), (coord1[2], coord1[3]), (255, 0,0), 2)

print(f"True Positives (TP): {true_positives}")
print(f"False Positives (FP): {false_positives}")
print(f"False Negatives (FN): {false_negatives}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
cv2.imshow("bounding boxes", image)
cv2.imwrite('YOLO2.jpg',image)      
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"Matched ground truth: {gt_matched}")
###============= END CALCULATING THE PRECISION, RECALL, TP, FP, FN =====####

####========= DRAWING THE STOLP DIAGRAM ===========####
metrics = ['True Positives', 'False Positives', 'False Negatives', 'Precision', 'Recall']
values = [true_positives, false_positives, false_negatives, precision, recall]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['green', 'red', 'blue', 'purple', 'orange'])

# Adding the title and labels
plt.title('Object Detection Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values / Scores')
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

# Show the plot
plt.tight_layout()
plt.show()
####========= END OF DRAWING THE STOLP DIAGRAM ===========####

mydata = [
    ["Execution time of DINO", f"{end - start} second"], 
    # ["Execution time of DINO", f"{time_exec_dino} second"],
    # ["METEOR Score", f"{score1}"],
    # ["BLEU score", f"{bleu_score}"],
    #["IOU", f"{iou_results}\n"],
    #["Num of found items", f"{len(labels)}, {labels}"],
    ["Predict time usage of Nvidia T4 GPU hardware (YOLOV8x)", f"{Predic_time} seconds"],
    ["Cost of the usage Nvidia T4 GPU hardware (YOLOV8x)", f"{Predic_time} * 0.000225/s = {Total_cost} dollar"],
    ["Precision", f"{precision}"],
    ["Recall", f"{recall}"],
    ["DINO predicted coordinates of objects: ", f"{coordinates}"],
    ["Precision", f"{precision}"],
    ["Recall", f"{recall}"],
    ["True positive", f"{true_positives}"],
    ["False positive", f"{false_positives}"],
    ["False negative", f"{false_negatives}"],
    ["Matched ground truth:", f"{gt_matched}"],
    ["iou threshold:", f"{iou_threshold}"]
]
 
# create header
head = [f"{image_path}","YOLOv8x"]
print(tabulate(mydata, headers=head, tablefmt="grid"))
f.write(tabulate(mydata, headers=head, tablefmt="grid"))