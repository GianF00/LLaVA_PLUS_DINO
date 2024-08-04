import matplotlib.pyplot as plt

# Defining the data
data = [
    # gt_ordning14.jpg
    {"recall": 1.0, "precision": 0.14705882352941177, "label": "GPT4 + Grounding DINO 12% (gt_ordning14.jpg)"},
    {"recall": 1.0, "precision": 0.3125, "label": "GPT4 + Grounding DINO 23% (gt_ordning14.jpg)"},
    {"recall": 0.8, "precision": 1.0, "label": "GPT4 + Grounding DINO 37% (gt_ordning14.jpg)"},

    # gt_ordning15.jpg
    {"recall": 0.8333333333333334, "precision": 0.15151515151515152, "label": "GPT4 + Grounding DINO 12% (gt_ordning15.jpg)"},
    {"recall": 0.8333333333333334, "precision": 0.45454545454545453, "label": "GPT4 + Grounding DINO 23% (gt_ordning15.jpg)"},
    {"recall": 0.6666666666666666, "precision": 1.0, "label": "GPT4 + Grounding DINO 37% (gt_ordning15.jpg)"},

    # YOLOv8x on gt_ordning14.jpg
    {"recall": 0.2, "precision": 0.5, "label": "YOLOv8x 12% (gt_ordning14.jpg)"},
    {"recall": 0.2, "precision": 0.5, "label": "YOLOv8x 23% (gt_ordning14.jpg)"},
    {"recall": 0.2, "precision": 0.5, "label": "YOLOv8x 37% (gt_ordning14.jpg)"},

    # YOLOv8x on gt_ordning15.jpg
    {"recall": 0.16666666666666666, "precision": 0.5, "label": "YOLOv8x 12% (gt_ordning15.jpg)"},
    {"recall": 0.16666666666666666, "precision": 0.5, "label": "YOLOv8x 23% (gt_ordning15.jpg)"},
    {"recall": 0.16666666666666666, "precision": 0.5, "label": "YOLOv8x 37% (gt_ordning15.jpg)"},

    # YOLOv8x on gt_bordny8_yolo.jpg
    {"recall": 0.5, "precision": 1.0, "label": "YOLOv8x 12% (gt_bordny8_yolo.jpg)"},
    {"recall": 0.5, "precision": 1.0, "label": "YOLOv8x 23% (gt_bordny8_yolo.jpg)"},
    {"recall": 0.5, "precision": 1.0, "label": "YOLOv8x 37% (gt_bordny8_yolo.jpg)"},

    # gt_bordny8_ny.jpg
    {"recall": 1.0, "precision": 0.3157894736842105, "label": "GPT4 + Grounding DINO 12% (gt_bordny8_ny.jpg)"},
    {"recall": 0.8333333333333334, "precision": 0.7142857142857143, "label": "GPT4 + Grounding DINO 23% (gt_bordny8_ny.jpg)"},
    {"recall": 0.5, "precision": 1.0, "label": "GPT4 + Grounding DINO 37% (gt_bordny8_ny.jpg)"},

    # gt_bordny8_ny.jpg with LLaVA
    {"recall": 0.8333333333333334, "precision": 0.125, "label": "LLaVA v1.6 12% (gt_bordny8_ny.jpg)"},
    {"recall": 0.8333333333333334, "precision": 0.625, "label": "LLaVA v1.6 23% (gt_bordny8_ny.jpg)"},
    {"recall": 0.8333333333333334, "precision": 0.5555555555555556, "label": "LLaVA v1.6 37% (gt_bordny8_ny.jpg)"},

    # gt_ordning14.jpg with LLaVA
    {"recall": 0.8, "precision": 0.25, "label": "LLaVA v1.6 12% (gt_ordning14.jpg)"},
    {"recall": 0.8, "precision": 0.6666666666666666, "label": "LLaVA v1.6 23% (gt_ordning14.jpg)"},
    {"recall": 0.6, "precision": 1.0, "label": "LLaVA v1.6 37% (gt_ordning14.jpg)"},

    # gt_ordning15.jpg with LLaVA
    {"recall": 1.0, "precision": 0.15789473684210525, "label": "LLaVA v1.6 12% (gt_ordning15.jpg)"},
    {"recall": 1.0, "precision": 0.4286, "label": "LLaVA v1.6 23% (gt_ordning15.jpg)"},
    {"recall": 0.6666666666666666, "precision": 1.0, "label": "LLaVA v1.6 37% (gt_ordning15.jpg)"},
]

# Create a new coordinate system plot
plt.figure(figsize=(12, 8))
plt.title('Precision vs Recall')

# Iterate over the data and plot each point
for entry in data:
    plt.scatter(entry["recall"], entry["precision"], label=entry["label"])

# Label the axes
plt.xlabel('Recall')
plt.ylabel('Precision')

# Add a legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()
