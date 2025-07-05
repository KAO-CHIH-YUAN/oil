from ultralytics import YOLO
from pathlib import Path
import yaml
import numpy as np
import cv2
from tqdm import tqdm
import torch
import shutil

from reconstruction_module import run_reconstruction_evaluation

def generate_test_predictions(model, exp_config, imgsz, results_path):
    """
    對測試集產生預測圖並儲存。
    """
    print("\n--- 產生測試集預測圖 ---")
    try:
        dataset_cfg = exp_config.get('dataset', {})
        if not dataset_cfg: return
        
        project_root = Path.cwd()
        base_path = project_root / dataset_cfg['path'].lstrip('../')
        test_img_dir = base_path / dataset_cfg['test']

        if not test_img_dir.is_dir():
            print(f"  [警告] 找不到測試圖片資料夾: {test_img_dir}，無法產生預測圖。")
            return
            
        predictions_dir = results_path / "test_predictions"
        # ⭐⭐⭐ 修正 #2: 移除刪除資料夾的指令 ⭐⭐⭐
        # 舊指令: 
        # if predictions_dir.exists():
        #     shutil.rmtree(predictions_dir) 
        # 移除後，程式不會再刪除任何檔案，重複執行時 YOLO 會自動建立 predict2, predict3...
        
        print(f"  - 預測結果圖將儲存於: {predictions_dir.parent}")
        # 使用 exist_ok=True，讓 YOLO 在資料夾已存在時不會報錯
        model.predict(
            source=str(test_img_dir), imgsz=imgsz, save=True,
            project=str(results_path), name=predictions_dir.name,
            conf=exp_config.get('eval_conf', 0.25),
            exist_ok=True
        )
    except Exception as e:
        print(f"  [錯誤] 產生預測圖時出錯: {e}"); import traceback; traceback.print_exc()

def calculate_pixel_level_metrics(model, exp_config, imgsz):
    """
    計算 patch 層級的像素指標。
    """
    print("\n--- 計算像素級別指標 (Accuracy, IoU) ---")
    try:
        dataset_cfg = exp_config.get('dataset', {});
        if not dataset_cfg: return {}
        project_root = Path.cwd(); base_path = project_root / dataset_cfg['path'].lstrip('../'); test_img_dir = base_path / dataset_cfg['test']
        test_label_dir = test_img_dir.parent.parent / 'labels' / test_img_dir.name
        if not test_label_dir.is_dir(): return {}
        image_files = list(test_img_dir.glob('*.jpg')) + list(test_img_dir.glob('*.png'))
        if not image_files: return {}
        total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
        for img_path in tqdm(image_files, desc="Calculating Pixel Metrics at Original Scale"):
            results = model.predict(source=str(img_path), verbose=False, imgsz=imgsz)
            gt_label_path = test_label_dir / (img_path.stem + '.png');
            if not gt_label_path.exists(): continue
            gt_mask_cv = cv2.imread(str(gt_label_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask_cv is None: continue
            gt_mask_binary = (gt_mask_cv > 0).astype(np.uint8); h_orig, w_orig = gt_mask_binary.shape
            if results and results[0].masks: pred_mask_small = torch.any(results[0].masks.data, dim=0).cpu().numpy().astype(np.uint8)
            else:
                pred_shape_h, pred_shape_w = (imgsz, imgsz) if isinstance(imgsz, int) else (imgsz[0], imgsz[1])
                if results and results[0].masks is not None: pred_shape_h, pred_shape_w = results[0].masks.data.shape[-2:]
                pred_mask_small = np.zeros((pred_shape_h, pred_shape_w), dtype=np.uint8)
            pred_mask_resized_to_original = cv2.resize(pred_mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            total_tp += np.sum(np.logical_and(pred_mask_resized_to_original, gt_mask_binary)); total_tn += np.sum(np.logical_and(np.logical_not(pred_mask_resized_to_original), np.logical_not(gt_mask_binary)))
            total_fp += np.sum(np.logical_and(pred_mask_resized_to_original, np.logical_not(gt_mask_binary))); total_fn += np.sum(np.logical_and(np.logical_not(pred_mask_resized_to_original), gt_mask_binary))
        denominator = total_tp + total_tn + total_fp + total_fn; pixel_accuracy = (total_tp + total_tn) / denominator if denominator > 0 else 0
        iou_denominator = total_tp + total_fp + total_fn; pixel_iou = total_tp / iou_denominator if iou_denominator > 0 else 0
        print(f"  - 像素級別準確率 (在原始尺寸上計算): {pixel_accuracy:.4f}"); print(f"  - 像素級別 IoU (在原始尺寸上計算): {pixel_iou:.4f}")
        return {'Accuracy(pixel)': f"{pixel_accuracy:.4f}", 'IoU(pixel)': f"{pixel_iou:.4f}"}
    except Exception as e:
        print(f"  [錯誤] 計算像素級指標時出錯: {e}"); import traceback; traceback.print_exc()
        return {}

def evaluate_and_visualize(exp_config, data_yaml_path, model_path, results_path):
    """
    評估流程的主函式。
    """
    exp_name = exp_config.get('test_name') or exp_config['experiment_name']
    print(f"\n--- 開始評估: {exp_name} ---")
    
    try:
        model = YOLO(model_path)
        imgsz = exp_config.get('imgsz', 640)
        
        eval_charts_dir = results_path / "standard_evaluation_charts"
        print(f"  - 正在執行實例級評估 (model.val)，結果圖將儲存於: {eval_charts_dir}")
        metrics = model.val(data=str(data_yaml_path), split='test', project=str(results_path), name=eval_charts_dir.name, exist_ok=True, imgsz=imgsz, conf=exp_config.get('eval_conf', 0.25), iou=exp_config.get('eval_iou', 0.6))
        
        eval_results = {}
        if hasattr(metrics, 'box') and metrics.box.map is not None:
            p,r = metrics.box.mp, metrics.box.mr; eval_results.update({'Precision(B)':f"{p:.4f}",'Recall(B)':f"{r:.4f}",'mAP50(B)':f"{metrics.box.map50:.4f}",'mAP50-95(B)':f"{metrics.box.map:.4f}",'F1-score(B)':f"{2*p*r/(p+r+1e-9):.4f}"})
        if hasattr(metrics, 'seg') and metrics.seg.map is not None:
            p_seg,r_seg=metrics.seg.mp,metrics.seg.mr; eval_results.update({'Precision(M)':f"{p_seg:.4f}",'Recall(M)':f"{r_seg:.4f}",'mAP50(M)':f"{metrics.seg.map50:.4f}",'mAP50-95(M)':f"{metrics.seg.map:.4f}",'F1-score(M)':f"{2*p_seg*r_seg/(p_seg+r_seg+1e-9):.4f}"})

        pixel_metrics = calculate_pixel_level_metrics(model, exp_config, imgsz)
        eval_results.update(pixel_metrics)
        generate_test_predictions(model, exp_config, imgsz, results_path)

        recon_config = exp_config.get('reconstruction')
        if recon_config and recon_config.get('enabled'):
            dataset_cfg = exp_config.get('dataset', {})
            project_root = Path.cwd()
            base_path = project_root / dataset_cfg.get('path', '').lstrip('./')
            patch_test_dir = base_path / dataset_cfg.get('test', '')
            
            original_data_root = Path(recon_config['original_data_root']).resolve()
            vis_params = {
                'min_conf': recon_config.get('overlay_min_conf', 0.25),
                'nms_iou': recon_config.get('overlay_nms_iou', 0.5),
                'alpha': recon_config.get('overlay_alpha', 0.2)
            }
            
            if patch_test_dir.is_dir() and original_data_root.is_dir():
                recon_metrics = run_reconstruction_evaluation(
                    model, patch_test_dir, original_data_root, results_path, imgsz, vis_params
                )
                eval_results.update(recon_metrics)
            else:
                 print(f"\n[警告] 重組評估跳過，因為找不到所需的路徑。")

        print(f"評估完成 ---")
        return eval_results
    except Exception as e:
        print(f"評估過程中發生嚴重錯誤: {e}"); import traceback; traceback.print_exc()
        return {"error": str(e)}