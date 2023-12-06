import numpy as np
import os
import json
from os.path import join
from .config import *

"""
    Delete random classes from dataset
    so that we can explore Metric Learning with rare classes in dataset
    
    Create "classes.json" with information abount rare and freq classes
"""

removal_classes = np.random.choice(
    np.arange(CLASSES_CNT),
    size=removal_size,
    replace=False
)

classes = {}

for folder in os.listdir(train_path):
    class_id = int(folder)
    if int(folder) in removal_classes:
        pathes = os.listdir(join(train_path, folder))
        remain_images = np.random.choice(
            pathes,
            size=remain_images_among_rares,
            replace=False
        )
        print("Remove", folder)
        print("    remain images: ", remain_images)
        for path in pathes:
            if path not in remain_images:
                os.remove(join(train_path, folder, path))
        # to JSON
        classes[folder] = {"id": class_id,
                           "type": "rare"}
    else:
        classes[folder] = {"id": class_id,
                           "type": "freq"}

with open("classes.json", "x") as f:
    f.write(json.dumps(classes))

