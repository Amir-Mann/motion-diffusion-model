

import random


random.seed(452) # Days of the hostages are imprisoned at time of writing this.


# Paths to the input and output files
all_file = 'dataset/HumanML3D/all.txt'
test_file = 'dataset/HumanML3D/test.txt'
train_val_file = 'dataset/HumanML3D/train_val.txt'
train_file = 'dataset/HumanML3D/train.txt'
val_file = 'dataset/HumanML3D/val.txt'
SPLIT_SIZE = 0.94


# Read the content of all.txt and test.txt
with open(all_file, 'r') as f:
    all_lines = set(f.read().splitlines())

with open(test_file, 'r') as f:
    test_lines = set(f.read().splitlines())

# Filter out the test entries from all.txt
train_val_lines_no_mirror = [line for line in all_lines if line not in test_lines if "M" not in line]
for line in train_val_lines_no_mirror:
    assert line in all_lines
    assert "M" + line in all_lines
    assert line not in test_lines
    assert "M" + line not in test_lines

# Split the filtered lines into train (95%) and val (5%) sets
random.shuffle(train_val_lines_no_mirror)
train_count = int(SPLIT_SIZE * len(train_val_lines_no_mirror))
train_lines_no_mirror = train_val_lines_no_mirror[:train_count]
val_lines_no_mirror = train_val_lines_no_mirror[train_count:]

# Ensure matching entries of the form xxxxxx and Mxxxxxx are in the same set
train_val_lines = sorted(train_val_lines_no_mirror) + ["M" + line for line in sorted(train_val_lines_no_mirror)]
train_lines = sorted(train_lines_no_mirror) + ["M" + line for line in sorted(train_lines_no_mirror)]
val_lines = sorted(val_lines_no_mirror) + ["M" + line for line in sorted(val_lines_no_mirror)]

# Reverse assertion: everything in all_lines is either in test, train, or val, but not in more than one
train_lines_set = set(train_lines)
val_lines_set = set(val_lines)
for line in all_lines:
    assert (line in test_lines) + (line in train_lines_set) + (line in val_lines_set) == 1, f"{line} is in more than one list"

# Assertion for Mxxxxxx correspondence: if x in test, Mx is in test, and vice versa
for line in test_lines:
    assert "M" + line in test_lines if "M" not in line else line[1:] in test_lines
for line in train_lines_set:
    assert "M" + line in train_lines_set if "M" not in line else line[1:] in train_lines_set
for line in val_lines_set:
    assert "M" + line in val_lines_set if "M" not in line else line[1:] in val_lines_set

# Print the number of items in train and val
print(f"Train items: {len(train_lines)}")
print(f"Val items: {len(val_lines)}")


# Write the train_val, train, and val files
with open(train_val_file, 'w') as f:
    f.write('\n'.join(train_val_lines))

with open(train_file, 'w') as f:
    f.write('\n'.join(train_lines))

with open(val_file, 'w') as f:
    f.write('\n'.join(val_lines))

print("Files have been created successfully.")
