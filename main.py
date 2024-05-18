import replicate
import time 
import numpy
import supervision as sv
import cv2 
import numpy as np
import psutil
import nltk
import json
import re
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score
#nltk.download('punkt')  # Ensure the tokenizer is downloaded
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tabulate import tabulate
from functions import calculate_iou, drawing_boxes,calculate_predi_time
import matplotlib.pyplot as plt
#Länk: https://replicate.com/yorickvp/llava-13b/api/learn-more 
# LÄnk till API https://github.com/oobabooga/text-generation-webui/commit/38ab214a416b2dbb6bcba4d318bfc847fbb4da36
#start = time.time()
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html 
#https://guides.himmelfarb.gwu.edu/studydesign101/helpful-formulas
#image = "bord10.jpg"
#image = "ordning3.jpg"
client = replicate.Client()

image = "ordning3.jpg"
#image = "bord3.jpg"
start = time.time()
input_image = open(f"{image}", "rb")
f = open('LLaVA_data.txt', 'a')
#query = "Calculate the coordinates in this format x0,y0,x1,y1 of the pot lid, the blue bowl, the scissors, the dish rack, the shelf above the sink, the spatula and of the cheese slicer."
# ordning3
query = "give me the bounding boxes of the white plate, the white bowl, the glass, the fork, the spoon, the knife, the dish rack, the shelf above the dish rack, the white coffe mugg in the shelf and the blue coffe mugg in the shelf"

f.write("Query: " + query + "\n")


# Reference text:
reference = "The image shows a kitchen scen where there is a dish drainer. Inside the dish drainer there are a white plate, a glass, a white bowl, two forks a spoon and a knife.\
                \nOn top of the dish drainer there is a white shelf with a white coffe mug and a blue coffe mug. The wall of the kitchen is made of white majolica."  # Correct answer

# input_data={
#     "image" : input_image,
#     "top_p": 1,
#     "prompt": query,
#     "max_tokens": 1024,
#     "temperature": 0.1
# }
# output = replicate.run(
#     "yorickvp/llava-13b:a0fdc44e4f2e1f20f2bb4e27846899953ac8e66c5886c5878fa1d6b73ce009e5",
#     input=input_data
# )

input_data={
    "image": input_image,
    "top_p": 1,
    "prompt": query,
    "history": [],
    "max_tokens": 1024,
    "temperature": 0.1
}
output = replicate.run(
    "yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",
    input=input_data
)
res = "".join(output)
print(res)
# User or model-generated answer
candidate = res
print("\nCandidate: ", candidate)
end = time.time()

############================== Extraction of the objects=============================##########################

description1 = candidate
with open('items.json', 'r') as file:
    data = json.load(file)
    item_list = data['items']

# Normalize the description to lower case to ensure case-insensitive matching
description_lower = description1.lower()

# Find which items from the JSON list are mentioned in the description
found_items = [item for item in item_list if item in description_lower]

# Clean items to ensure proper formatting for API query
found_items_cleaned = [item.strip() for item in found_items]  # Remove extra spaces
found_items_query = ','.join(found_items_cleaned)

print("\nFound Items:", str(found_items_query))
## Number 
print("Number of Items Found:", len(found_items), "\n")
print("Total Number of Items:", len(item_list), "\n")
print("Found {} out of {} items.".format(len(found_items), len(item_list)), "\n")
####################============END OF Extraction of the objects==========================####################


###===============Getting the bounding boxes from the text================##########
pattern = re.compile(r'\[\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\s*\]')

# Find all matches in the text
matches = pattern.findall(candidate)

# Convert the matches to a list of tuples with float numbers
pred_coordinates = [list(map(float, match)) for match in matches]

# Print the extracted coordinates
for coord in pred_coordinates:    
    print(coord)
###===============End of getting the bounding boxes================##########


###########============= DRAWING THE GROUND TRUHT BOUNDING BOXES ================############
img = cv2.imread(image)
numOfTimes = 10
labels = "objects"
coordinates = []
for _ in range(numOfTimes):
    x_val, y_val, w_val, h_val = drawing_boxes(labels,img)
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

cv2.imshow("Ground truth bounding boxes", temp)
cv2.imwrite('gt_image_ordning3.jpg',temp)      
cv2.waitKey(0)
#print(f"{x= }	{y= }	{w= }	{h= }")
#cv2.waitKey(0)

#rectangle DRAW THE BOUNDING BOXES BASED ON THE COORDINATES 
# img_with_gruthrubbox = cv2.rectangle(img, (x,y),(x + w,y + h), (0,255,0),2)
# cv2.imshow("with the box", img_with_gruthrubbox)
# cv2.waitKey(0)
cv2.destroyAllWindows()

####============== END OF THE DRAWING THE GROUND TRUHT BOUNDING BOXES 

###============drawing the predicted bboxes ================#####

img = cv2.imread('gt_image_ordning3.jpg')
image_height, image_width = img.shape[0], img.shape[1]

pr = []
for el in pred_coordinates:    
    x_min_scaled = int(el[0] * image_width)
    y_min_scaled = int(el[1] * image_height)
    x_max_scaled = int(el[2] * image_width)
    y_max_scaled = int(el[3] * image_height)

    # Justera koordinater för att undvika att gå utanför bildens gränser
    x_min_scaled = max(0, min(x_min_scaled, image_width - 1))
    y_min_scaled = max(0, min(y_min_scaled, image_height - 1))
    x_max_scaled = max(0, min(x_max_scaled, image_width - 1))
    y_max_scaled = max(0, min(y_max_scaled, image_height - 1))

    
    pr.append([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled])

    # Rita den förutsagda bounding boxen
    cv2.rectangle(img, (x_min_scaled, y_min_scaled), (x_max_scaled, y_max_scaled), (0, 0, 255), 2)

cv2.imshow('Image with Bounding Boxes', img)
cv2.imwrite("LLaVA_ordning3_ny6_1.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

####======== end of dwaing the predicted bboxes =============####

####======== CALCULATING THE PRECISION, RECALL, TP, FP, FN=============####
pr_boxes = pr
print("Predicted bboxes",pr_boxes, "\n")

## Calculating the Precision and Recall using different iou threshold: 20, 50, 60%
iou_threshold = 0.2
true_positives = 0
false_positives = 0
gt_matched = set()

# Calculate TP and FP
for i, pred_box in enumerate(pr_boxes):
    match_found = False
    for j, gt_box in enumerate(coordinates):
        iou = calculate_iou(pred_box, gt_box)
        if iou >= iou_threshold:
            if j not in gt_matched:
                true_positives += 1
                gt_matched.add(j)
                match_found = True
                break
    if not match_found:
        false_positives += 1

# Calculate FN
false_negatives = len(coordinates) - len(gt_matched)

# Calculate Precision and Recall
precision = true_positives / (true_positives + false_positives) 
recall = true_positives / (true_positives + false_negatives)

print(f"True Positives (TP): {true_positives}")
print(f"False Positives (FP): {false_positives}")
print(f"False Negatives (FN): {false_negatives}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
cv2.imshow("Ground truth bounding boxes", temp)
cv2.imwrite('LLaVA1.jpg',temp)      
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"Matched ground truth: {gt_matched}")
####======== END CALCULATING THE PRECISION, RECALL, TP, FP, FN=============####

####========= DRAWING THE DIAGRAM ===========####
metrics = ['True Positives', 'False Positives', 'False Negatives', 'Precision', 'Recall']
values = [true_positives, false_positives, false_negatives, precision, recall]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['green', 'red', 'blue', 'purple', 'orange'])

# Adding the title and labels
plt.title('Object Detection Metrics ')
plt.xlabel('Metrics')
plt.ylabel('Values / Scores')
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

# Show the plot
plt.tight_layout()
plt.show()
####========= DRAWING THE DIAGRAM ===========####


#####=============DINO================#####
# output = replicate.run(
#     "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",    
#     input={
#         "image": input_image,
#         "query": found_items_query,
#         "box_threshold": 0.35,
#         "text_threshold": 0.25,
#         "show_visualisation": True,   
#     }
# )
#resultatet blir en JSON format med en länk som visar input bilden med resultatet
#print(output)
###===============END OF DINO============####

####====== CALCULATING THE COST OF THE GPU with REPLICATE TIME =====#####
Total_cost,Predic_time = calculate_predi_time("0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",input_data,0.000725)

###======= END OF CALCULATING THE COST OF THE GPU with REPLICATE TIME =======### 

####=========== OLIKA MÄTNINGAR ==========#####
time_exec = end - start
print("time of execution: ", time_exec )

# Tokenize the reference and candidate
tokenized_reference = word_tokenize(reference)
tokenized_candidate = word_tokenize(candidate)

# Calculate METEOR score
score = meteor_score([tokenized_reference], tokenized_candidate)
print(f"METEOR Score: {score:.3f}")

chencherry = SmoothingFunction()

# Now, let's calculate the BLEU score with smoothing
bleu_score = sentence_bleu([reference], candidate, 
                           smoothing_function=chencherry.method4)  # reference tokens must be a list of lists

print(f"BLEU Score: {bleu_score}\n")

f.write(f"image: {image}, below are the measures for this image using the model LLaVA")
f.write("\nReference text provided by the user:\n" + reference +"\n")
f.write("\n")
f.write("Candidate text provided by the modell:\n" + candidate + "\n")
f.write("Execution time: " + repr(time_exec)+ "\n")
# f.write("METEOR Score: "+ repr(score)+ "\n")
# f.write("BLEU score: "+repr(bleu_score))


mydata = [
    ["Execution time local", f"{time_exec} seconds"], 
    # # ["METEOR Score", f"{score}"],
    # ["BLEU score", f"{bleu_score}"],
    ["Predicted coordinates from LLaVA", f"{coordinates}"],
    #["IOU", f"{iou_results}"],
    ["Num of found items", f"{len(coordinates)}, {found_items}"],
    ["Precision", f"{precision}"],
    ["Recall", f"{recall}"],
    ["True positive", f"{true_positives}"],
    ["False positive", f"{false_positives}"],
    ["False negative", f"{false_negatives}"],
    ["Matched ground truth:", f"{gt_matched}"],
    ["Predict time usage of the Nvidia A40 (Large) GPU hardware (LLaVA)", f"{Predic_time} seconds"],
    ["Cost of the usage Nvidia A40 (Large) GPU hardware (LLaVA)", f"{Predic_time} * 0.000725/s = {Total_cost} dollar"],
    ["iou threshold: ", f"{iou_threshold}"]

]
 
# create header
head = [f"{image}","LLaVA"]
print(tabulate(mydata, headers=head, tablefmt="grid"))
f.write(tabulate(mydata, headers=head, tablefmt="grid"))
f.close()


