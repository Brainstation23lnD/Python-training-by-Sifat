import os
import shutil
import random


with_mask_dir = "../data/with_mask"
without_mask_dir = "../data/without_mask"

base_dir = "../data"
os.makedirs(base_dir, exist_ok=True)

folders = ["train", "val", "test"]
classes = ['with_mask', 'without_mask']

for folder in folders:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, folder, cls), exist_ok=True)

def split_data(_source, _training, _validation, _test, split_ratio=(0.7, 0.15, 0.15)):
    # Only include regular files; ignore directories/hidden/system files
    files = [f for f in os.listdir(_source) if os.path.isfile(os.path.join(_source, f))]
    random.shuffle(files)

    train_size = int(len(files) * split_ratio[0])
    val_size = int(len(files) * split_ratio[1])

    train_files = files[:train_size]
    val_files = files[train_size:(train_size + val_size)]
    test_files = files[(train_size + val_size):]

    for f in train_files:
        shutil.copy(os.path.join(_source, f), _training)

    for f in val_files:
        shutil.copy(os.path.join(_source, f), _validation)

    for f in test_files:
        shutil.copy(os.path.join(_source, f), _test)


split_data(
    with_mask_dir,
    os.path.join(base_dir, "train/with_mask"),
    os.path.join(base_dir, "val/with_mask"),
    os.path.join(base_dir, "test/with_mask")
)

split_data(
    without_mask_dir,
    os.path.join(base_dir, "train/without_mask"),
    os.path.join(base_dir, "val/without_mask"),
    os.path.join(base_dir, "test/without_mask")
)





