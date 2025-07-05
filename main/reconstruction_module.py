import torch
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm
from torchvision.ops import nms

def stitch_masks(patch_predictions_with_scores, original_size):
    """
    【加權平均版】將來自多個圖塊(patch)的分割遮罩，透過「加權平均」的方式平滑地拼接到一張原始大圖上。
    權重為每個遮罩對應的 bounding box 信心度分數。
    """
    stitched_canvas = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
    weight_map = np.zeros_like(stitched_canvas, dtype=np.float32)
    
    for mask, score, x, y in patch_predictions_with_scores:
        if mask is not None:
            mask_np = mask.cpu().numpy().astype(np.float32)
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze(0)
            
            h, w = mask_np.shape
            stitched_canvas[y:y+h, x:x+w] += mask_np * score
            weight_map[y:y+h, x:x+w] += score
            
    safe_weight_map = weight_map.copy()
    safe_weight_map[safe_weight_map == 0] = 1
    
    final_mask_prob = stitched_canvas / safe_weight_map
    
    return (final_mask_prob > 0.5).astype(np.uint8)

def calculate_iou(mask1, mask2):
    """計算兩個二值化遮罩的 IoU。"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-9)

def draw_final_boxes(image, boxes, scores, labels):
    """在影像上繪製最終經過 NMS 處理的邊界框。"""
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
    """
    ⭐【最終修正版：混合模式】
    - 對於 Bounding Box：使用 NMS 產生清晰、不重疊的結果。
    - 對於 Segmentation：使用所有高信心度的偵測結果進行加權平均，以得到最穩健的分割圖。
    """
    print("\n--- 開始執行 Patch 重組評估 (最終版：混合模式) ---")
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

    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
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
        
        # 步驟 1: 收集當前大圖所有 patch 上的所有高信心度偵測物件
        all_detections_on_image = []
        for patch_path, x, y in patches:
            results = model.predict(source=str(patch_path), verbose=False, imgsz=imgsz, conf=vis_params['min_conf'])
            if results and results[0].masks:
                for i in range(len(results[0].masks.data)):
                    all_detections_on_image.append({
                        "mask": results[0].masks.data[i],
                        "score": results[0].boxes.conf[i].cpu().item(),
                        "global_box": [
                            results[0].boxes.xyxy[i][0].cpu().item() + x,
                            results[0].boxes.xyxy[i][1].cpu().item() + y,
                            results[0].boxes.xyxy[i][2].cpu().item() + x,
                            results[0].boxes.xyxy[i][3].cpu().item() + y
                        ],
                        "origin_x": x,
                        "origin_y": y
                    })

        if not all_detections_on_image:
            reconstructed_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # ⭐⭐⭐ 核心修改：分離 BBox 和 Segmentation 的處理 ⭐⭐⭐

            # --- 2a. 處理 Bounding Box：對所有偵測物件執行 NMS，以得到最終要繪製的清晰邊界框 ---
            boxes_tensor = torch.tensor([d['global_box'] for d in all_detections_on_image], dtype=torch.float32)
            scores_tensor = torch.tensor([d['score'] for d in all_detections_on_image], dtype=torch.float32)
            winning_indices = nms(boxes_tensor, scores_tensor, vis_params['nms_iou'])

            final_boxes = boxes_tensor[winning_indices].numpy()
            final_scores = scores_tensor[winning_indices].numpy()
            final_labels = [model.names[0]] * len(winning_indices)

            # --- 2b. 處理 Segmentation：使用「所有」高信心度的偵測遮罩進行加權平均，以得到最穩健的分割圖 ---
            # 準備 stitch_masks 的輸入，這次使用 all_detections_on_image 而不是 NMS 的勝利者
            all_mask_inputs = []
            for detection in all_detections_on_image:
                all_mask_inputs.append(
                    (detection['mask'], detection['score'], detection['origin_x'], detection['origin_y'])
                )
            reconstructed_mask = stitch_masks(all_mask_inputs, (w, h))

        # --- 步驟 3: 計算指標 (基於最終的 reconstructed_mask) ---
        total_tp += np.sum(np.logical_and(reconstructed_mask, gt_mask_binary))
        total_tn += np.sum(np.logical_and(np.logical_not(reconstructed_mask), np.logical_not(gt_mask_binary)))
        total_fp += np.sum(np.logical_and(reconstructed_mask, np.logical_not(gt_mask_binary)))
        total_fn += np.sum(np.logical_and(np.logical_not(reconstructed_mask), gt_mask_binary))

        # --- 步驟 4: 產生並儲存視覺化結果 (使用 NMS 後的 final_boxes) ---
        per_image_iou = calculate_iou(reconstructed_mask, gt_mask_binary)
        label_vis_image = np.zeros((h, w, 3), dtype=np.uint8)
        label_vis_image[gt_mask_binary == 1] = [0, 255, 0]
        label_vis_image[reconstructed_mask == 1] = [0, 0, 255]
        label_vis_image[np.logical_and(gt_mask_binary, reconstructed_mask) == 1] = [0, 255, 255]
        cv2.imwrite(str(label_vis_dir / f"{original_name}_iou_{per_image_iou:.4f}.png"), label_vis_image)
        
        overlay = original_image.copy(); alpha = vis_params['alpha']
        overlay[np.logical_and(reconstructed_mask, gt_mask_binary)] = (0, 255, 255)
        overlay[np.logical_and(reconstructed_mask, np.logical_not(gt_mask_binary))] = (0, 0, 255)
        overlay[np.logical_and(np.logical_not(reconstructed_mask), gt_mask_binary)] = (255, 0, 0)
        final_overlay_image = cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0)
        final_overlay_image_with_boxes = draw_final_boxes(final_overlay_image, final_boxes, final_scores, final_labels)
        cv2.imwrite(str(overlay_vis_dir / f"{original_name}_overlay_iou_{per_image_iou:.4f}.png"), final_overlay_image_with_boxes)

    # --- 最終步驟: 在所有圖片處理完後，計算並回傳整體指標 ---
    iou_denominator = total_tp + total_fp + total_fn
    final_iou = total_tp / iou_denominator if iou_denominator > 0 else 0
    acc_denominator = total_tp + total_tn + total_fp + total_fn
    final_accuracy = (total_tp + total_tn) / acc_denominator if acc_denominator > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_denominator = precision + recall
    final_f1_score = 2 * (precision * recall) / f1_denominator if f1_denominator > 0 else 0

    print(f"--- 重組評估完成 ---")
    print(f"    整體像素準確率 (Overall Accuracy): {final_accuracy:.4f}")
    print(f"    整體像素 F1 分數 (Overall F1-Score): {final_f1_score:.4f}")
    print(f"    整體像素 IoU (Overall IoU): {final_iou:.4f}")
    
    return {
        "reconstruction_accuracy": f"{final_accuracy:.4f}",
        "reconstruction_f1_score": f"{final_f1_score:.4f}",
        "reconstruction_mean_iou": f"{final_iou:.4f}"
    }