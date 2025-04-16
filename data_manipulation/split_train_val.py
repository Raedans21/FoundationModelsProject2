import random

with open("../data/train.jsonl", "r") as f:
    lines = f.readlines()

# Shuffle file lines before splitting
random.seed(10)
random.shuffle(lines)

# 20% validation, 80% training set split
val_ratio = 0.2
split_index = int(len(lines) * (1- val_ratio))
train_set = lines[:split_index]
val_set = lines[split_index:]

with open("../data/split_data/train.jsonl", "w") as f:
    f.writelines(train_set)

with open("../data/split_data/val.jsonl", "w") as f:
    f.writelines(val_set)
