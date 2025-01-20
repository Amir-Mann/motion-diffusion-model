

import re
import os
import random


random.seed(452) # Days of the hostages are imprisoned at time of writing this.


# Paths to the input and output files
base_path = "/home/amir.mann/MDM"
all_file = os.path.join(base_path, 'dataset/HumanML3D/all.txt')
test_file = os.path.join(base_path, 'dataset/HumanML3D/test.txt')
train_val_file = os.path.join(base_path, 'dataset/HumanML3D/train_val.txt')
train_file = os.path.join(base_path, 'dataset/HumanML3D/train.txt')
val_file = os.path.join(base_path, 'dataset/HumanML3D/val.txt')
SPLIT_SIZE = 0.94


# Read the content of all.txt and test.txt
with open(all_file, 'r') as f:
    all_lines = set(f.read().splitlines())

with open(train_val_file, 'r') as f:
    train_val_lines = f.read().splitlines()

pattern = re.compile(r'(^|\s)(sit|seat|chair)')
train_lines = []
val_lines = []
last_mark = 0
for i, line in enumerate(sorted(train_val_lines)):
    with open(f"/home/amir.mann/MDM/dataset/HumanML3D/texts/{line}.txt", "r") as f:
        caption = f.read()
        if pattern.search(caption):
            val_lines.append(line)
        else:
            train_lines.append(line)


# Print the number of items in train and val
print(f"Train items: {len(train_lines)}")
print(f"Val items: {len(val_lines)}")


with open(train_file, 'w') as f:
    f.write('\n'.join(train_lines))

with open(val_file, 'w') as f:
    f.write('\n'.join(val_lines))

print("Files have been created successfully.")
