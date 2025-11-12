import numpy as np
from skimage.transform import resize

def resize_array(arr, target_h=800, target_w=64, order=1):
    """
    Resize a 3D or 4D array to (target_h, target_w).
    Supports:
        (N, H, W)  or  (N, H, W, 1)
    """
    arr_resized = []
    for i in range(arr.shape[0]):
        img = arr[i]
        if img.ndim == 3:  # (H, W, 1)
            img_r = resize(img, (target_h, target_w, 1), order=order, preserve_range=True, anti_aliasing=True)
        else:  # (H, W)
            img_r = resize(img, (target_h, target_w), order=order, preserve_range=True, anti_aliasing=True)
        arr_resized.append(img_r)
    return np.array(arr_resized, dtype=np.float32)

def merge_and_resize_folds(folds_list, target_h=800, target_w=64, shuffle=False, seed=42):
    """
    Merge multiple fold sets (e.g., Antarctica, Greenland, SHARAD)
    and resize all to (target_h, target_w).
    """
    rng = np.random.default_rng(seed)
    n_folds = min(len(fl) for fl in folds_list)
    merged_folds = []

    for i in range(n_folds):
        parts = [fl[i] for fl in folds_list]

        # --- Resize + Merge each split ---
        def merge_split(split_key):
            arrays = []
            for p in parts:
                if split_key not in p: continue
                arr = p[split_key]
                arr_r = resize_array(arr, target_h, target_w, order=1 if 'image' in split_key else 0)
                arrays.append(arr_r)
            merged = np.concatenate(arrays, axis=0)
            if shuffle:
                idx = rng.permutation(len(merged))
                merged = merged[idx]
            return merged

        tr_img = merge_split('train_images')
        tr_lab = merge_split('train_labels')
        va_img = merge_split('val_images')
        va_lab = merge_split('val_labels')
        te_img = merge_split('test_images')
        te_lab = merge_split('test_labels')

        merged_folds.append({
            'fold': i + 1,
            'train_images': tr_img,
            'train_labels': tr_lab,
            'val_images':   va_img,
            'val_labels':   va_lab,
            'test_images':  te_img,
            'test_labels':  te_lab,
        })

        print(f"[Merged Fold {i+1}] train {tr_img.shape}, val {va_img.shape}, test {te_img.shape}")

    return merged_folds
