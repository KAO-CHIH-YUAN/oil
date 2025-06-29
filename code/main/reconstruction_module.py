import torch
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm
from torchvision.ops import nms

def stitch_masks(patch_predictions, original_size, imgsz):
    """
    將多個 patch 的分割遮罩平滑地拼接到一張大圖上。
    此版本已修復變數解包 (unpack) 的錯誤。
    """
    stitched_canvas = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
    count_map = np.zeros_like(stitched_canvas, dtype=np.float32)
    
    # ⭐⭐⭐ 核心修正：使用明確的變數名稱 (mask, x, y) 來解包 ⭐⭐⭐
    for mask, x, y in patch_predictions:
        if mask is not None:
            mask_np = mask.cpu().numpy().astype(np.float32)
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze(0)
            
            # 使用從迴圈中正確解包的 x 和 y
            h, w = mask_np.shape
            stitched_canvas[y:y+h, x:x+w] += mask_np
            count_map[y:y+h, x:x+w] += 1
            
    count_map[count_map == 0] = 1
    final_mask = stitched_canvas / count_map
    return (final_mask > 0.5).astype(np.uint8)

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-9)

def draw_final_boxes(image, boxes, scores, labels):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        score = scores[i]
        label = labels[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        text = f"{label} {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 255), -1)
        cv2.putText(image, text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return image

def run_reconstruction_evaluation(model, test_image_dir, original_data_root, results_path, imgsz, vis_params):
    print("\n--- 開始執行 Patch 重組評估 ---")
    original_images_dir = original_data_root / 'images' / 'test'
    original_gt_dir = original_data_root / 'labels' / 'test'
    if not original_images_dir.is_dir() or not original_gt_dir.is_dir(): return {}

    patches_by_original = defaultdict(list)
    for patch_path in test_image_dir.glob('*.jpg'):
        match = re.match(r"(.+)_patch_x(\d+)_y(\d+)", patch_path.stem)
        if match:
            original_name, x, y = match.groups()
            patches_by_original[original_name].append((patch_path, int(x), int(y)))
    
    if not patches_by_original: return {}

    all_ious = []
    overlay_vis_dir = results_path / "reconstruction_overlays"; overlay_vis_dir.mkdir(exist_ok=True)
    label_vis_dir = results_path / "reconstruction_visuals"; label_vis_dir.mkdir(exist_ok=True)

    for original_name, patches in tqdm(patches_by_original.items(), desc="Reconstructing and Visualizing"):
        original_img_path = next(original_images_dir.glob(f"{original_name}.*"), None)
        gt_mask_path = original_gt_dir / f"{original_name}.png"
        if not original_img_path or not gt_mask_path.exists(): continue

        original_image = cv2.imread(str(original_img_path))
        gt_mask_cv = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        if original_image is None or gt_mask_cv is None: continue

        gt_mask_binary = (gt_mask_cv > 0).astype(np.uint8)
        h, w, _ = original_image.shape
        
        all_patch_masks = []
        all_global_boxes = []
        all_scores = []
        
        for patch_path, x, y in patches:
            results = model.predict(source=str(patch_path), verbose=False, imgsz=imgsz, conf=vis_params['min_conf'])
            all_patch_masks.append((results[0].masks.data[0] if results and results[0].masks else None, x, y))
            if results and results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    all_global_boxes.append([x1 + x, y1 + y, x2 + x, y2 + y])
                    all_scores.append(box.conf[0].cpu().item())

        reconstructed_mask = stitch_masks(all_patch_masks, (w, h), imgsz)
        iou = calculate_iou(reconstructed_mask, gt_mask_binary)
        all_ious.append(iou)
        
        final_boxes, final_scores, final_labels = [], [], []
        if all_global_boxes:
            boxes_tensor = torch.tensor(all_global_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
            indices = nms(boxes_tensor, scores_tensor, vis_params['nms_iou'])
            final_boxes = boxes_tensor[indices].numpy()
            final_scores = scores_tensor[indices].numpy()
            final_labels = [model.names[0]] * len(indices)

        label_vis_image = np.zeros((h, w, 3), dtype=np.uint8)
        label_vis_image[gt_mask_binary == 1] = [0, 255, 0]
        label_vis_image[reconstructed_mask == 1] = [0, 0, 255]
        label_vis_image[np.logical_and(gt_mask_binary, reconstructed_mask) == 1] = [0, 255, 255]
        cv2.imwrite(str(label_vis_dir / f"{original_name}_iou_{iou:.4f}.png"), label_vis_image)

        overlay = original_image.copy(); alpha = vis_params['alpha']
        true_positives = np.logical_and(reconstructed_mask, gt_mask_binary)
        false_positives = np.logical_and(reconstructed_mask, np.logical_not(gt_mask_binary))
        false_negatives = np.logical_and(np.logical_not(reconstructed_mask), gt_mask_binary)
        overlay[true_positives] = (0, 255, 255); overlay[false_positives] = (0, 0, 255); overlay[false_negatives] = (255, 0, 0)
        final_overlay_image = cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0)
        final_overlay_image_with_boxes = draw_final_boxes(final_overlay_image, final_boxes, final_scores, final_labels)
        cv2.imwrite(str(overlay_vis_dir / f"{original_name}_overlay_iou_{iou:.4f}.png"), final_overlay_image_with_boxes)

    mean_iou = np.mean(all_ious) if all_ious else 0.0
    print(f"--- 重組評估完成 ---"); print(f"    平均整圖 IoU (Mean Full-Image IoU): {mean_iou:.4f}")
    return {"reconstruction_mean_iou": f"{mean_iou:.4f}"}