import os
import shutil
import random
from sklearn.model_selection import train_test_split

# ================= CONFIG =================
SEED = 42
DATA_ROOT = "data"
SOURCE_DIR = os.path.join(DATA_ROOT, "cell_images")
SPLIT_DIRS = ["train", "val", "test"]

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

CLASSES = ["Parasitized", "Uninfected"]
# =========================================

random.seed(SEED)

def make_dirs():
    for split in SPLIT_DIRS:
        for cls in CLASSES:
            path = os.path.join(DATA_ROOT, split, cls)
            os.makedirs(path, exist_ok=True)

def split_class(class_name):
    class_path = os.path.join(SOURCE_DIR, class_name)
    images = os.listdir(class_path)

    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=(1 - SPLIT_RATIOS["train"]),
        random_state=SEED,
        shuffle=True
    )

    val_ratio_adjusted = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"])

    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=(1 - val_ratio_adjusted),
        random_state=SEED,
        shuffle=True
    )

    return train_imgs, val_imgs, test_imgs

def copy_files(file_list, split, class_name):
    for fname in file_list:
        src = os.path.join(SOURCE_DIR, class_name, fname)
        dst = os.path.join(DATA_ROOT, split, class_name, fname)
        shutil.copy2(src, dst)

def main():
    make_dirs()

    for cls in CLASSES:
        train_imgs, val_imgs, test_imgs = split_class(cls)

        copy_files(train_imgs, "train", cls)
        copy_files(val_imgs, "val", cls)
        copy_files(test_imgs, "test", cls)

        print(f"{cls}: "
              f"train={len(train_imgs)}, "
              f"val={len(val_imgs)}, "
              f"test={len(test_imgs)}")

    print("\n✅ Dataset split complete.")
    print("Original data untouched.")
    print("Splits are reproducible with SEED =", SEED)

if __name__ == "__main__":
    main()
