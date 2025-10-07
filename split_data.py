# src/split_data.py
import os, shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"            # where your original TB and NORMAL folders are
OUT_DIR = "data/processed"      # will be created/filled with train/val/test
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
CLASS_NAMES = None  # set to None to infer from RAW_DIR subfolders

def prepare_splits(raw_dir=RAW_DIR, out_dir=OUT_DIR,
                   test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE):
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data folder not found: {raw_dir}")

    # infer classes
    classes = CLASS_NAMES or sorted([d for d in os.listdir(raw_dir)
                                     if os.path.isdir(os.path.join(raw_dir, d)) and not d.startswith('.')])
    if not classes:
        raise RuntimeError(f"No class subfolders found in {raw_dir}. Expected e.g. 'TB' and 'NORMAL'.")

    print("Classes detected:", classes)

    # collect files and labels
    files = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_dir):
            raise RuntimeError(f"Class folder missing: {cls_dir}")
        cls_files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  {cls}: {len(cls_files)} images")
        files += cls_files
        labels += [cls] * len(cls_files)

    # first split test
    f_trainval, f_test, l_trainval, l_test = train_test_split(
        files, labels, test_size=test_size, random_state=random_state, stratify=labels)

    # split train/val from trainval
    rel_val = val_size / (1.0 - test_size)
    f_train, f_val, l_train, l_val = train_test_split(
        f_trainval, l_trainval, test_size=rel_val, random_state=random_state, stratify=l_trainval)

    splits = {
        'train': (f_train, l_train),
        'val': (f_val, l_val),
        'test': (f_test, l_test)
    }

    # create out dirs and copy files
    for split_name, (f_list, l_list) in splits.items():
        for cls in classes:
            out_cls_dir = os.path.join(out_dir, split_name, cls)
            os.makedirs(out_cls_dir, exist_ok=True)
        for src, cls in zip(f_list, l_list):
            dst = os.path.join(out_dir, split_name, cls, os.path.basename(src))
            shutil.copy2(src, dst)

    print("Splits created at:", out_dir)
    for split_name in ['train', 'val', 'test']:
        total = 0
        print(split_name, "counts:")
        for cls in classes:
            c = len(os.listdir(os.path.join(out_dir, split_name, cls)))
            total += c
            print(f"  {cls}: {c}")
        print("  total:", total)

if __name__ == "__main__":
    prepare_splits()
