from matplotlib import pyplot as plt
import replicate
import time 
import numpy
import supervision as sv
import cv2 
import numpy as np
import psutil
import nltk
import json
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score
#nltk.download('punkt')  # Ensure the tokenizer is downloaded
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tabulate import tabulate
from functions import calculate_IOU, drawing_boxes, calculate_predi_time,calculate_iou

#Länk: https://replicate.com/yorickvp/llava-13b/api/learn-more 
# LÄnk till API https://github.com/oobabooga/text-generation-webui/commit/38ab214a416b2dbb6bcba4d318bfc847fbb4da36
#start = time.time()

image = "ordning3.jpg"

#image = "ordning5.jpg"
start = time.time()
input_image = open(f"{image}", "rb")
f = open('LLaVA_data.txt', 'a')
# Ordning3
query = "describe the utensils, glass, the white plate and white bowl in the dish rack, the dish rack, the shelf above the dish rack and the cups inside of the white shelf"

f.write("Query: " + query + "\n")

# Reference text:
# imagen ordning 3
reference = "The image shows a kitchen scen where there is a dish rack. Inside the dish rack there are a white plate, a glass, a white bowl, two forks a spoon and a knife.\
                \nOn top of the dish rack there is a white shelf with cups. The wall of the kitchen is made of white majolica."  # Correct answer

# output = replicate.run(
#     "yorickvp/llava-13b:a0fdc44e4f2e1f20f2bb4e27846899953ac8e66c5886c5878fa1d6b73ce009e5",
#     input={
#         "image" : input_image,
#         "top_p": 1,
#         "prompt": query,
#         "max_tokens": 1024,
#         "temperature": 0.1
#     }
# )

###================ LLaVA ====================#####
input_data1={
    "image": input_image,
    "top_p": 1,
    "prompt": query,
    "history": [],
    "max_tokens": 1024,
    "temperature": 0.1
}

output = replicate.run(
    "yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",
    input=input_data1
)

res = "".join(output)
print(res)

# User or model-generated answer
candidate = res
print("\nCandidate: ", candidate)

####=================== END of LLAVA =========================####

############================== Extraction of the objects from the picture =============================##########################
 #candidate
with open('items.json', 'r') as file:
    data = json.load(file)
    item_list = data['items']

# Normalize the description to lower case to ensure case-insensitive matching
description_lower = candidate.lower()

# Find which items from the JSON list are mentioned in the description
found_items = [item for item in item_list if item in description_lower]

# Clean items to ensure proper formatting for API query
found_items_cleaned = [item.strip() for item in found_items]  # Remove extra spaces
found_items_query = ','.join(found_items_cleaned)
print("\nFound Items:", found_items_query)
## Number 
print("Number of Items Found:", len(found_items))
print("Total Number of Items:", len(item_list))
print("Found {} out of {} items.".format(len(found_items), len(item_list)))

####################============END OF Extraction of the objects==========================####################
###########============= DRAWING THE GROUND TRUHT BOUNDING BOXES ================############
img = cv2.imread(image)
numOfTimes = 10
print("\nN of times to draw", numOfTimes)
Labels = "objects"
coordinates = []
for _ in range(numOfTimes):
    x_val, y_val, w_val, h_val = drawing_boxes(Labels,img)
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
    # Append the merged_result to mrgd_list_float
    # mrgd_lis_float.append(merged_result)

print("\nresult ground truth bounding box",result_gt)
print("\nCoordinates: ", coordinates)
for coord in coordinates:
    temp = cv2.rectangle(img, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 2)


#print(f"{x= }	{y= }	{w= }	{h= }")
#cv2.waitKey(0)

#rectangle DRAW THE BOUNDING BOXES BASED ON THE COORDINATES 
# img_with_gruthrubbox = cv2.rectangle(img, (x,y),(x + w,y + h), (0,255,0),2)
# cv2.imshow("with the box", img_with_gruthrubbox)
# cv2.waitKey(0)
#iou_results = help_iou_more_coordinates(result_gt, predicted_bboxes)

####============== END OF THE DRAWING THE GROUND TRUHT BOUNDING BOXES =====###
#####=============DINO================#####


# image = "ordning3.jpg"
# input_image = open(f"{image}", "rb")
input_data2={
    "image": input_image,
    "query": found_items_query,
    "box_threshold": 0.38,
    "text_threshold": 0.25,
    "show_visualisation": True,   
}
start_dino = time.time()
output = replicate.run(
    "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",    
    input=input_data2
)
#resultatet blir en JSON format med en länk som visar input bilden med resultatet
print(output)
end = time.time()
# # CPU usage after the API call
# cpu_after_dino = psutil.cpu_percent(interval=None)
###===============END OF DINO============####

# if "cup" or "mugg" in found_items:
#     numOfTimes = len(found_items)+1
#     print("N of times to run: ", numOfTimes,"\n")

###=========== EXTRACTING THE BBOXES OF DINO TO CALCULATE THE IOU ============#
predicted_bboxes = []
# List to hold detailed info including bounding boxes, confidence, and class for 'dog' or 'cat' detections
bounding_boxes = []
confid = []
labels = []
for el in output['detections']:
    # label = el['label']
    # if label == 'dog' or label == 'cat':
    #     bbox = el['bbox']
    #     conf = el['confidence']
        
    #     # Add just the bbox to the predicted_bboxes list
    #     predicted_bboxes.extend(bbox)
    #     confid.append(conf)
    #     # Add detailed info to the bounding_boxes list
    #     bounding_boxes.append({
    #         'class': label,
    #         'confidence': conf,
    #         'bbox': bbox
    #     })
    
    bbox = el['bbox']
    conf = el['confidence']
    label = el['label']
    # Add just the bbox to the predicted_bboxes list
    predicted_bboxes.extend(bbox)
    confid.append(conf)
    labels.append(label)
    # Add detailed info to the bounding_boxes list
    bounding_boxes.append({
        'class': label,
        'confidence': conf,
        'bbox': bbox
    })


print("result bounding_boxes: ", bounding_boxes)
print("\nresult predicted_bbox: ", predicted_bboxes)
print("\nresult confid: ", confid)
print("\nlabels: ", labels)

for bbox in bounding_boxes:
    class_label = bbox['class']
    confidence = bbox['confidence']
    bbox = bbox['bbox']

    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    cv2.putText(img, f'{class_label}: {confidence:.2f}', (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("Ground truth bounding boxes", temp)
cv2.imwrite('llaVA_ordning3_37_3.jpg',temp)      
cv2.waitKey(0)
cv2.destroyAllWindows()
###=========== EDN OF EXTRACTING THE BBOXES OF DINO TO CALCULATE THE IOU ============#



###============= CALCULATING THE PRECISION, RECALL, TP, FP, FN =====####
pr_boxes = [predicted_bboxes[i:i+4] for i in range(0, len(predicted_bboxes), 4)]
print("Predicted bboxes",pr_boxes, "\n")
true_positives = 0
false_positives = 0
iou_threshold = 0.60
matches = []
gt_matched = set()  # För att hålla reda på vilka GT-boxar som matchats
# Calculate True Positives and False Positives


for p_idx, p_box in enumerate(pr_boxes):
    match_found = False
    for gt_idx, gt_box in enumerate(coordinates):
        iou = calculate_iou(p_box, gt_box)
        if iou >= iou_threshold:
            if gt_idx not in gt_matched:
                true_positives += 1
                gt_matched.add(gt_idx)
                match_found = True

                # Draw bounding box on the image if IoU is greater than 0.80
                cv2.rectangle(img, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (255, 0, 0), 2)
                cv2.putText(img, f'{labels[p_idx]}: {confid[p_idx]:.2f}', (p_box[0], p_box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                break

    if not match_found:
        false_positives += 1

# Calculate False Negatives
false_negatives = len(coordinates) - len(gt_matched)

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
cv2.imshow("Ground truth bounding boxes", temp)
cv2.imwrite('LLAVA_37_3.jpg',temp)      
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"True Positives (TP): {true_positives}")
print(f"False Positives (FP): {false_positives}")
print(f"False Negatives (FN): {false_negatives}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
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
####========= DRAWING THE DIAGRAM ===========####



###============= CALCULATING THE PREDICTION TIME FOR LLaVA AND Grounding DINO =============######
Total_cost,Predic_time = calculate_predi_time("0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63", input_data1,0.000725)
Total_cost2, Predic_time2 = calculate_predi_time("efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",input_data2,0.000225) 
###============= END OF CALCULATING THE PREDICTION TIME FOR LLaVA AND Grounding DINO =============######


# time_exec_llava = end_llava - start_llava
# time_exec_dino = end_dino - start_dino
print("\n")
# print(f"CPU usage change: {cpu_usage_llava}%")
# print(f"CPU usage change: {cpu_usage_dino}%")
# print(f"Memory usage change: {memory_usage_dino} bytes")
# print(f"Memory usage change: {memory_usage_llava} bytes")
# print("time of execution: ", time_exec_llava,"\n")
# print("time of execution: ", time_exec_dino)

# Tokenize the reference and candidate
tokenized_reference = word_tokenize(reference)
tokenized_candidate = word_tokenize(candidate)


#print("List of iou values for every bbox in the picture: ", iou_results, "\n")

f.write(f"image: {image}, below are the measures for this image using the model LLaVA")
f.write("\nReference text provided by the user:\n" + reference +"\n")
f.write("\n")
f.write("Candidate text provided by the modell:\n" + candidate + "\n")
# f.write("CPU usage llava: " + repr(cpu_usage_llava) + "\n")
# f.write("Memory usage llava: " + repr(memory_usage_llava) + "\n")
# f.write("CPU usage dino: " + repr(cpu_usage_dino) + "\n")
# f.write("Memory usage dino: " + repr(memory_usage_dino) + "\n")
# f.write("Execution time LLaVA: " + repr(time_exec_llava)+ "\n")
# f.write("Execution time DINO: " + repr(time_exec_dino)+ "\n")
#f.write("METEOR Score: "+ repr(score)+ "\n")
#f.write("BLEU score: "+repr(bleu_score))
#f.write("IOU: " + repr(iou_results))

mydata = [
    # ["CPU usage llava", f"{cpu_usage_llava}%"],
    # ["CPU usage dino", f"{cpu_usage_dino}%"], 
    # ["Memory usage llava", f"{memory_usage_llava} B"], 
    # ["Memory usage dino", f"{memory_usage_dino} B"], 
    # ["Execution time LLaVA", f"{time_exec_llava} second"], 
    ["Execution time LLaVA + DINO", f"{end - start} second"], 
    # ["METEOR Score", f"{score}"],
    # ["BLEU score", f"{bleu_score}"],
    #["IOU", f"{iou_results}\n"],
    ["Num of found items ", f"{len(labels)}, {labels}"],
    ["Predict time usage of the Nvidia A40 (Large) GPU hardware (LLaVA)", f"{Predic_time} seconds"],
    ["Cost of the usage Nvidia A40 (Large) GPU hardware (LLaVA)", f"{Predic_time} * 0.000725/s = {Total_cost} dollar"],
    ["Predict time usage of Nvidia T4 GPU hardware (DINO)", f"{Predic_time2} seconds"],
    ["Cost of the usage Nvidia T4 GPU hardware (DINO)", f"{Predic_time2} * 0.000225/s = {Total_cost2} dollar"],
    # ["Total CPU usage:", f"{cpu_usage_dino + cpu_usage_llava}"],
    # ["Total memory usage:", f"{memory_usage_dino + memory_usage_llava}"],
    ["Precision", f"{precision}"],
    ["Recall", f"{recall}"],
    ["True positive", f"{true_positives}"],
    ["False positive", f"{false_positives}"],
    ["False negative", f"{false_negatives}"],
    ["Matched ground truth:", f"{gt_matched}"],
    ["DINO predicted coordinates of objects: ", f"{coordinates}"],
    ["IoU threshold ", f"{iou_threshold}"],

]
 
# create header
head = [f"{image}","LLaVA v1.6 13B och Grounding DINO"]
print(tabulate(mydata, headers=head, tablefmt="grid"))
f.write(tabulate(mydata, headers=head, tablefmt="grid"))