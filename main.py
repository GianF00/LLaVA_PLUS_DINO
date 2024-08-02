from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import replicate
import time 
import supervision as sv
import cv2 
import numpy as np
import json
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score
#nltk.download('punkt')  # Ensure the tokenizer is downloaded
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tabulate import tabulate
from functions import calculate_iou, calculate_precision_recall, drawing_boxes, calculate_predi_time, help_iou_more_coordinates,intersection_over_union



#image = "ordning3.jpg"
#image = "gt_bordny9_ny1.jpg"
#image = "gt_bordny8_yolo.jpg"
#image = "gt_bordny8_ny.jpg"
#image = "gt_ordning11.jpg"
#image = "gt_ordning12.jpg"
#image = "gt_ordning14.jpg"
image = "gt_ordning15.jpg"
img = cv2.imread(image)
start = time.time()
input_image = open(f"{image}", "rb")
f = open('LLaVA_data.txt', 'a')
query = (
    "Describe in detail the main objects present in the picture."
    "Provide the result in a JSON format where each object has 'name' key and the JSON data has the object name 'labels'. Ensure that each object is reported only once."
)
f.write("Query: " + query + "\n")



####====================== Model LLaVA =====================######
input_data1={
    "image": input_image,
    "top_p": 1,
    "prompt": query,
    "history": [],
    "max_tokens": 1024,
    "temperature": 0.60
}
#ändra på temperaturen för att visa att du förstår parameter 
temper = input_data1["temperature"]

output = replicate.run(
    "yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",
    input=input_data1
)

res = "".join(output)
print(res)
####====================== End of Model LLaVA =====================######

###=====================Extracting the objects Version 2============###
new_res = res.replace('json', '').replace("```", "")
print(f"\n {new_res}")
with open("sample1.json", "w") as file:
    file.write(new_res)

with open("sample1.json", "r") as in_file:
    json_object = json.load(in_file)

print(json_object)
f.write(repr(json_object))
names = ', '.join(item['name'] for item in json_object['labels'])
print(f"\n names: {names}")

# f.write("\n" + repr(json_object) + "\n")
###==================== End of extracting the objects Version 2 ======###

###########============= DRAWING THE GROUND TRUHT BOUNDING BOXES ================############
# img = cv2.imread(image)
# numOfTimes = 5
# print("\nN of times to draw", numOfTimes)
# Labels = "objects"
# coordinates = []
# for _ in range(numOfTimes):
#     x_val, y_val, w_val, h_val = drawing_boxes(Labels,img)
#     coordinates.append([x_val,y_val, (x_val + w_val), (y_val+h_val)])
#     #print("Num of iter: ", x)

# ##Add two list together:
# # result = coordinates[0] + coordinates[1]
# # print("\n",result)

# result_gt = []
# sz = len(coordinates)
# for x in range(sz):
#     #merged_result = []

#     # Add each element of list_float to itself and append to merged_result
#     # for num in list_float:
#     #     merged_result.append(num + num)
#     result_gt.extend(coordinates[x])
#     # Append the merged_result to mrgd_list_float
#     # mrgd_lis_float.append(merged_result)

# print("\nresult ground truth bounding box",result_gt)
# print("\nCoordinates: ", coordinates)
# for coord in coordinates:
#     temp = cv2.rectangle(img, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 2)

# cv2.imshow("Ground truth bounding boxes", temp)
# cv2.imwrite('gt_image_llavaDino3.jpg',temp)      
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#bordny9
#gt = [[195, 328, 354, 483], [88, 385, 125, 520], [129, 82, 238, 169],[42, 407, 414, 601],  [0, 0, 437, 187]]
#gt = [195, 328, 354, 483, 88, 385, 125, 520, 42, 407, 414, 601, 129, 82, 238, 169, 0, 0, 437, 187]

#bordny8
#gt = [[197, 323, 362, 485], [78, 392, 116, 535], [43, 415, 441, 623], [0, 2, 457, 175], [90, 78, 195, 156], [217, 83, 323, 161]]
#orden:tallrik, kniv, dish rack, hylla, vänster mug och höger mug    

#gt = [197, 323, 362, 485, 78, 392, 116, 535, 43, 415, 441, 623, 0, 2, 457, 175, 90, 78, 195, 156, 217, 83, 323, 161]

#rgt = changingFormatOfList(gt)

#   BORDNY11.JPG      orden: cuchiaio, forchetta, tabla de picar, dish rack, bicchiere nero, shelf 
#gt = [[328, 395, 355, 465], [295, 394, 323, 463], [77, 319, 267, 543], [49, 427, 423, 631], [262, 127, 337, 206], [1, 2, 444, 220]]
#   BORDNY12.JPG        orden: forchetta,cuchiaio, forbice, dish rack, bicchiere nero, grater, shelf
#gt = [[259, 397, 285, 463], [289, 397, 314, 460], [127, 424, 196, 526], [28, 426, 371, 619], [56, 154, 124, 225], [202, 71, 304, 226], [0, 0, 376, 235]]
#   BORDNY13.JPG    orden: forbice, coperchio, dish rack, bicchiere nero, tazza bianca, shelf
#gt = [[262, 399, 324, 466], [92, 369, 251, 522], [39, 432, 382, 624], [85, 162, 153, 231], [223, 161, 320, 228], [0, 2, 385, 238]]
#   BORDNY14.JPG orden: forchetta, forbici, dish rack, grater, shelf
#gt = [[294, 377, 318, 453], [116, 420, 186, 536], [11, 415, 380, 611], [141, 39, 244, 195], [0, 1, 385, 209]]
#   BORDNY15        orden: grater, fork, dish rack, black mug, whit mug, shelf
gt = [[225, 402, 311, 551], [91, 455, 125, 561], [31, 430, 406, 632], [105, 124, 171, 196], [252, 127, 353, 199], [0, 0, 415, 214]]

####============== END OF THE DRAWING THE GROUND TRUHT BOUNDING BOXES =====###
####=============DINO================#####
#input_image = open(f"{image}", "rb")
input_data2={
    "image": input_image,
    "query": names,
    "box_threshold": 0.37,
    "text_threshold": 0.12,
    "show_visualisation": True,   
}
value_of_bThresh = input_data2["box_threshold"]
str_bThresh = f"DINO's box threshold {value_of_bThresh}"
value = 37
PredimgToBeStored = f"LLaVA_pred6_bordnyyy15_bt{value}.jpg"
BlueimgToBeStored = f"LLaVA_blue6_bordnyyy15_bt{value}.jpg"
start_dino = time.time()
output = replicate.run(
    "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",    
    input=input_data2
)
#resultatet blir en JSON format med en länk som visar input bilden med resultatet
print(output)
end = time.time()
# ###===============END OF DINO============####

# ###=========== EXTRACTING THE BBOXES OF DINO TO CALCULATE THE IOU ============#
predicted_bboxes = []
bounding_boxes = []
predi_boundboxes = []
confid = []
labels = []
for el in output['detections']:    
    bbox = el['bbox']
    conf = el['confidence']
    label = el['label']
    # Add just the bbox to the predicted_bboxes list
    predicted_bboxes.extend(bbox)
    predi_boundboxes.append(bbox)
    confid.append(conf)
    labels.append(label)
    # Add detailed info to the bounding_boxes list
    bounding_boxes.append({
        'class': label,
        'confidence': conf,
        'bbox': bbox
    })

print("\n")
print("Bounding_boxes: ", bounding_boxes)
print("\nPredicted_bbox (extend): ", predicted_bboxes)
print("\nPredicted_bbox (append): ", predi_boundboxes)
print("\nResult confid: ", confid)
print("\nlabels: ", labels)

for bbox in bounding_boxes:
    class_label = bbox['class']
    confidence = bbox['confidence']
    bbox = bbox['bbox']

    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    cv2.putText(img, f'{class_label}: {confidence:.2f}', (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

unique_labels = set(labels)
#unique_count = len(unique_labels)

cv2.imshow("Predicted bounding boxes", img)
#bt = box threshold
cv2.imwrite(PredimgToBeStored,img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ###=========== EDN OF EXTRACTING THE BBOXES OF DINO TO CALCULATE THE IOU ============#



###=============== stolpdiagram för objekten som detekteras =================#####
# Plot the bar chart with confidence and labels
#gt_bordny8_ny.jpg
# color_map = {
#         'knife': 'green',
#         'plate': 'green',
#         'cup': 'green',
#         'dish rack': 'green',
#         'shelf':'green',
#         'cupboard':'green',
#         'white plate':'green',
#         'white dish rack': 'green',
#         'black knife':'green',
#         'white cup':'green'                                 
# }
#gt_ordning11.jpg
# color_map = {
#         'cup': 'green',
#         'dish rack': 'green',
#         'shelf':'green',
#         'cupboard':'green',
#         'white dish rack': 'green',
#         'spoon':'green',
#         'fork':'green',
#         'cutting board':'green',
#         'black cup':'green',
#         'cabinet':'green'                                
# }
#gt_ordning12.jpg
# color_map = {
#         'cup': 'green',
#         'dish rack': 'green',
#         'shelf':'green',
#         'cupboard':'green',
#         'white dish rack': 'green',
#         'spoon':'green',
#         'fork':'green',
#         'scissors':'green',
#         'black cup':'green',
#         'grater':'green',
#         'cabinet':'green'                                 
# }
#gt_ordning14.jpg
# color_map = {
#     'dish rack': 'green',
#     'shelf':'green',
#     'cupboard':'green',
#     'white dish rack': 'green',
#     'fork':'green',
#     'cabinet':'green',
#     'scissors':'green',
#     'grater':'green'                                
# }
#gt_ordning15.jpg
color_map = {
        'dish rack': 'green',
        'shelf':'green',
        'cupboard':'green',
        'white dish rack': 'green',
        'fork':'green',
        'cabinet':'green',
        'black cup':'green',
        'grater':'green',
        'white cup':'green',
        'cup':'green',
        'white mug':'green'                                   
}

default_color = 'gray'
colors = [color_map.get(label, default_color) for label in labels]
plt.figure(figsize=(10, 6))
plt.bar(range(len(labels)), confid, color=colors, tick_label=labels)
plt.xlabel('Objects')
plt.ylabel('Confidence')

plt.axhline(y=value_of_bThresh, color='red', linestyle='--', label=str_bThresh)
plt.text(0, value_of_bThresh, f'{value_of_bThresh}', color='red', va='center', ha='left', backgroundcolor='white')
plt.title('Confidence levels of different objects generated by LLaVA + Grounding DINO')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
ground_truth_patch = mpatches.Patch(color='green', label='Ground Truth Objects')
non_ground_truth_patch = mpatches.Patch(color='gray', label='Non Ground Truth Objects')
plt.legend(handles=[ground_truth_patch, non_ground_truth_patch])
# Save the plot to a file
plot_path = fr"C:\Users\gianf\OneDrive\Skrivbord\YOLOv8\llava_confidenceeee6_bar_chart{value}.png"
plt.savefig(plot_path)
###=============== end stolpdiagram för objekten som detekteras =================#####




# ###============= CALCULATING THE PRECISION, RECALL, TP, FP, FN =====####
pr_boxes = [predicted_bboxes[i:i+4] for i in range(0, len(predicted_bboxes), 4)]
#print("Predicted bboxes: ",pr_boxes, "\n", "GT: ", gt,"\n")
true_positives = 0
false_positives = 0
iou_threshold = 0.75
matches = []
gt_matched = set()  # För att hålla reda på vilka GT-boxar som matchats
# GT, P = help_iou_more_coordinates(gt,predicted_bboxes)
# print(f"GT {GT}, P {P}")
# Calculate True Positives and False Positives

for p_idx, p_box in enumerate(predi_boundboxes):
    match_found = False
    for gt_idx, gt_box in enumerate(gt):
        iou = calculate_iou(p_box, gt_box)
        print(f"Comparing Ground Truth Box: {gt_box} with Predicted Box: {p_box}, IoU: {round(iou, 3)}")  # Add this line to visualize all comparisons
        if iou >= iou_threshold:
            if gt_idx not in gt_matched:
                true_positives += 1
                gt_matched.add(gt_idx)
                print(f"Match Found! Ground Truth Box: {gt_box}, Predicted Box: {p_box}, IoU: {round(iou, 3)}")
                match_found = True

                cv2.rectangle(img, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (255, 0, 0), 2)
                cv2.putText(img, f'{labels[p_idx]}: {confid[p_idx]:.2f}', (p_box[0], p_box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                break

    if not match_found:
        print(f"No match found for Predicted Box: {p_box}")
        false_positives += 1

##diskutera denna fp
#formeln: fn = antal gt minus true positve
false_negatives = len(gt) - true_positives
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
print("Precision: ", precision)
print("Recall: ", recall)
print(f"True Positives (TP): {true_positives}")
print(f"False Positives (FP): {false_positives}")
print(f"False Negatives (FN): {false_negatives}")
print(f"Matched ground truth: {gt_matched}")
  
cv2.imshow("Blue bounding boxes", img)
cv2.imwrite(BlueimgToBeStored,img)
cv2.waitKey(0)
cv2.destroyAllWindows()

###============= END CALCULATING THE PRECISION, RECALL, TP, FP, FN =====####


###============= CALCULATING THE PREDICTION TIME FOR LLaVA AND Grounding DINO =============######
Total_cost,Predic_time = calculate_predi_time("0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63", input_data1,0.000725)
Total_cost2, Predic_time2 = calculate_predi_time("efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",input_data2,0.000225) 
###============= END OF CALCULATING THE PREDICTION TIME FOR LLaVA AND Grounding DINO =============######

#print("List of iou values for every bbox in the picture: ", iou_results, "\n")

f.write(f"image: {image}, below are the measures for this image using the model LLaVA")
#f.write("\nReference text provided by the user:\n" + reference +"\n")
f.write("\n")
f.write("res text provided by the modell:\n" + res + "\n")
f.write("\n" + repr(json_object) + "\n")

mydata = [

    ["Execution time LLaVA + DINO", f"{end - start} second"], 
    ["Num of found items ", f"{len(labels)}, {labels}"],
    ["Predict time usage of the Nvidia A40 (Large) GPU hardware (LLaVA)", f"{Predic_time} seconds"],
    ["Cost of the usage Nvidia A40 (Large) GPU hardware (LLaVA)", f"{Predic_time} * 0.000725/s = {Total_cost} dollar"],
    ["Predict time usage of Nvidia T4 GPU hardware (DINO)", f"{Predic_time2} seconds"],
    ["Cost of the usage Nvidia T4 GPU hardware (DINO)", f"{Predic_time2} * 0.000225/s = {Total_cost2} dollar"], 
    ["Precision", f"{precision}"],
    ["Recall", f"{recall}"],
    ["True positive", f"{true_positives}"],
    ["False positive", f"{false_positives}"],
    ["False negative", f"{false_negatives}"],
    ["Matched ground truth:", f"{gt_matched}"],
    ["DINO predicted coordinates of objects: ", f"{predi_boundboxes}"],
    ["IoU threshold ", f"{iou_threshold}"],
    ["gt coordinates: ", f"{gt}"],
    ["Result confid:", f"{confid}"],
    ["labels: ", f"{labels}"],
    ["box_threshold: ", f"{value_of_bThresh}"],
    ["Temperature: ", f"{temper}"]

]
 
# create header
head = [f"{image}","LLaVA v1.6 13B och Grounding DINO"]
print(tabulate(mydata, headers=head, tablefmt="grid"))
f.write(tabulate(mydata, headers=head, tablefmt="grid"))
