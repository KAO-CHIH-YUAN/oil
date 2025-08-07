# ==============================================================================
# ===              完整 evaluation_module.py 檔案 (附完整註解)               ===
# ==============================================================================

# 匯入必要的函式庫
from ultralytics import YOLO  # YOLOv8 的核心函式庫
from pathlib import Path      # 用於處理檔案和目錄路徑，確保跨作業系統相容性
import yaml                   # 用於讀寫 YAML 設定檔
import numpy as np            # 用於數值計算，特別是影像陣列處理
import cv2                    # OpenCV 函式庫，用於影像讀取、寫入和處理
from tqdm import tqdm         # 提供一個漂亮的進度條，用於長時間運行的迴圈
import torch                  # PyTorch 函式庫，YOLOv8 基於此建構
import shutil                 # 用於高階的檔案操作，例如刪除整個資料夾

# 從您的專案中匯入 reconstruction_module，用於後續的影像重組評估
from reconstruction_module import run_reconstruction_evaluation

def generate_categorized_predictions(model, exp_config, imgsz, results_path):
    """
    對測試集產生預測，並根據預測結果與真實標籤的比較，
    將預測後的影像分類儲存到對應的資料夾 (TP, FP, FN, TN) 中。

    Args:
        model (YOLO): 已經載入權重的 YOLO 模型物件。
        exp_config (dict): 包含當前實驗設定的字典。
        imgsz (int): 用於預測的影像尺寸。
        results_path (Path): 本次實驗結果的根目錄路徑。
    """
    print("\n--- 產生分類化的測試集預測圖 (TP, FP, FN, TN) ---")
    try:
        # --- 步驟 1: 準備並驗證所有需要的路徑 ---
        dataset_cfg = exp_config.get('dataset', {})
        if not dataset_cfg:
            print("  [錯誤] 找不到 dataset 設定，無法執行分類預測。")
            return

        project_root = Path.cwd() # 獲取當前工作目錄
        # 組合出資料集的絕對路徑
        base_path = project_root / dataset_cfg.get('path', '').lstrip('./')
        # 測試影像資料夾路徑
        test_img_dir = base_path / dataset_cfg.get('test', 'images/test')
        # 根據影像路徑推斷出對應的標籤資料夾路徑
        test_label_dir = test_img_dir.parent.parent / 'labels' / test_img_dir.name
        
        # 檢查影像和標籤資料夾是否存在，若不存在則無法繼續
        if not test_img_dir.is_dir() or not test_label_dir.is_dir():
            print(f"  [警告] 找不到測試圖片資料夾 ({test_img_dir}) 或標籤資料夾 ({test_label_dir})，跳過此任務。")
            return

        # --- 步驟 2: 建立用來存放分類結果的輸出資料夾 ---
        output_base_dir = results_path / "categorized_predictions"
        # 定義四種分類的資料夾路徑
        tp_dir = output_base_dir / "1_True_Positive (偵測到且正確)"
        fp_dir = output_base_dir / "2_False_Positive (偵測到但錯誤)"
        fn_dir = output_base_dir / "3_False_Negative (未偵測到但應偵測)"
        tn_dir = output_base_dir / "4_True_Negative (正確忽略)"

        # 為了確保每次執行都是全新的結果，如果舊的資料夾存在，就先刪除它
        if output_base_dir.exists():
            shutil.rmtree(output_base_dir)
        # 建立所有分類資料夾
        tp_dir.mkdir(parents=True)
        fp_dir.mkdir(parents=True)
        fn_dir.mkdir(parents=True)
        tn_dir.mkdir(parents=True)

        # 獲取測試資料夾中所有 .jpg 和 .png 格式的影像檔案
        image_files = list(test_img_dir.glob('*.jpg')) + list(test_img_dir.glob('*.png'))

        # --- 步驟 3: 遍歷所有測試影像，進行預測和分類 ---
        for img_path in tqdm(image_files, desc="分類預測結果"):
            # 檢查真實標籤 (Ground Truth) 是否存在且非空檔案
            gt_txt_path = test_label_dir / (img_path.stem + '.txt')
            # 如果 .txt 標籤檔存在且檔案大小大於0，代表這張圖有真實的油汙目標
            has_ground_truth_oil = gt_txt_path.exists() and gt_txt_path.stat().st_size > 0

            # 使用 YOLO 模型對單張影像執行預測
            results = model.predict(source=str(img_path), verbose=False, imgsz=imgsz, conf=exp_config.get('eval_conf', 0.25))
            
            # 檢查模型的預測結果中是否包含任何偵測框
            prediction_made = results and results[0].boxes and len(results[0].boxes.data) > 0

            # --- 步驟 4: 根據「是否有預測」和「是否有真實目標」的組合，進行四種情況的分類 ---
            if prediction_made and has_ground_truth_oil:
                # 情況一: 真陽性 (True Positive, TP)
                # 模型做出了預測，且影像中確實有油汙。這是最好的情況。
                annotated_image = results[0].plot() # 獲取疊加上預測框和遮罩的影像
                cv2.imwrite(str(tp_dir / img_path.name), annotated_image) # 儲存到 TP 資料夾
            
            elif prediction_made and not has_ground_truth_oil:
                # 情況二: 偽陽性 (False Positive, FP)
                # 模型做出了預測，但影像中其實沒有油汙。這是「誤報」。
                annotated_image = results[0].plot()
                cv2.imwrite(str(fp_dir / img_path.name), annotated_image) # 儲存到 FP 資料夾

            elif not prediction_made and has_ground_truth_oil:
                # 情況三: 偽陰性 (False Negative, FN)
                # 模型沒有做出預測，但影像中其實有油汙。這是「漏報」。
                shutil.copy(img_path, fn_dir / img_path.name) # 將原始影像複製到 FN 資料夾
            
            elif not prediction_made and not has_ground_truth_oil:
                # 情況四: 真陰性 (True Negative, TN)
                # 模型沒有做出預測，且影像中確實也沒有油汙。這是模型正確的「忽略」行為。
                shutil.copy(img_path, tn_dir / img_path.name) # 將原始影像複製到 TN 資料夾

        print(f"  分類化的預測圖已成功儲存至: {output_base_dir}")

    except Exception as e:
        print(f"  [錯誤] 產生分類化預測圖時發生錯誤: {e}")
        import traceback; traceback.print_exc()

def calculate_pixel_level_metrics(model, exp_config, imgsz):
    """
    計算 patch 層級的像素指標，例如像素準確率 (Pixel Accuracy) 和 IoU。
    這個函式會直接比較預測遮罩和真實遮罩中的每一個像素。
    """
    print("\n--- 計算像素級別指標 (Accuracy, IoU) ---")
    try:
        # 同樣地，準備好所有需要的路徑
        dataset_cfg = exp_config.get('dataset', {});
        if not dataset_cfg: return {}
        project_root = Path.cwd(); base_path = project_root / dataset_cfg['path'].lstrip('../'); test_img_dir = base_path / dataset_cfg['test']
        test_label_dir = test_img_dir.parent.parent / 'labels' / test_img_dir.name
        if not test_label_dir.is_dir(): return {}
        image_files = list(test_img_dir.glob('*.jpg')) + list(test_img_dir.glob('*.png'))
        if not image_files: return {}
        
        # 初始化 TP, TN, FP, FN 計數器 (這裡的 TP/TN 是指像素級別)
        total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

        # 遍歷所有測試圖片
        for img_path in tqdm(image_files, desc="Calculating Pixel Metrics at Original Scale"):
            results = model.predict(source=str(img_path), verbose=False, imgsz=imgsz)
            
            # 讀取真實的 .png 遮罩檔案
            gt_label_path = test_label_dir / (img_path.stem + '.png');
            if not gt_label_path.exists(): continue
            gt_mask_cv = cv2.imread(str(gt_label_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask_cv is None: continue
            
            # 將真實遮罩轉換為二值化 (0 或 1)
            gt_mask_binary = (gt_mask_cv > 0).astype(np.uint8); h_orig, w_orig = gt_mask_binary.shape
            
            # 處理模型的預測遮罩
            if results and results[0].masks:
                # 如果有預測遮罩，將所有遮罩合併為一個
                pred_mask_small = torch.any(results[0].masks.data, dim=0).cpu().numpy().astype(np.uint8)
            else:
                # 如果沒有任何預測，則產生一個全黑的遮罩
                pred_shape_h, pred_shape_w = (imgsz, imgsz) if isinstance(imgsz, int) else (imgsz[0], imgsz[1])
                if results and results[0].masks is not None: pred_shape_h, pred_shape_w = results[0].masks.data.shape[-2:]
                pred_mask_small = np.zeros((pred_shape_h, pred_shape_w), dtype=np.uint8)
            
            # 將預測遮罩放大回原始影像尺寸，以便與真實遮罩比較
            pred_mask_resized_to_original = cv2.resize(pred_mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
            # 在像素層級計算 TP, TN, FP, FN
            total_tp += np.sum(np.logical_and(pred_mask_resized_to_original, gt_mask_binary))
            total_tn += np.sum(np.logical_and(np.logical_not(pred_mask_resized_to_original), np.logical_not(gt_mask_binary)))
            total_fp += np.sum(np.logical_and(pred_mask_resized_to_original, np.logical_not(gt_mask_binary)))
            total_fn += np.sum(np.logical_and(np.logical_not(pred_mask_resized_to_original), gt_mask_binary))
        
        # 根據累計的像素計數，計算最終的準確率和 IoU
        denominator = total_tp + total_tn + total_fp + total_fn; pixel_accuracy = (total_tp + total_tn) / denominator if denominator > 0 else 0
        iou_denominator = total_tp + total_fp + total_fn; pixel_iou = total_tp / iou_denominator if iou_denominator > 0 else 0
        
        print(f"  - 像素級別準確率 (在原始尺寸上計算): {pixel_accuracy:.4f}")
        print(f"  - 像素級別 IoU (在原始尺寸上計算): {pixel_iou:.4f}")
        return {'Accuracy(pixel)': f"{pixel_accuracy:.4f}", 'IoU(pixel)': f"{pixel_iou:.4f}"}
    except Exception as e:
        print(f"  [錯誤] 計算像素級指標時出錯: {e}"); import traceback; traceback.print_exc()
        return {}

def evaluate_and_visualize(exp_config, data_yaml_path, model_path, results_path):
    """
    評估流程的主函式，它會協調所有評估相關的任務。

    這是評估流程的總入口點。
    """
    exp_name = exp_config.get('test_name') or exp_config['experiment_name']
    print(f"\n--- 開始評估: {exp_name} ---")
    
    try:
        # 載入模型和設定
        model = YOLO(model_path)
        imgsz = exp_config.get('imgsz', 640)
        
        # --- 任務 1: 執行 YOLO 內建的標準評估 (model.val) ---
        # 這會產生 mAP, Precision, Recall 等標準指標，並儲存 PR 曲線等圖表
        eval_charts_dir = results_path / "standard_evaluation_charts"
        print(f"  - 正在執行實例級評估 (model.val)，結果圖將儲存於: {eval_charts_dir}")
        metrics = model.val(data=str(data_yaml_path), split='test', project=str(results_path), name=eval_charts_dir.name, exist_ok=True, imgsz=imgsz, conf=exp_config.get('eval_conf', 0.25), iou=exp_config.get('eval_iou', 0.6))
        
        # 將標準評估結果整理到一個字典中，以便後續記錄到 Excel
        eval_results = {}
        if hasattr(metrics, 'box') and metrics.box.map is not None:
            p,r = metrics.box.mp, metrics.box.mr; eval_results.update({'Precision(B)':f"{p:.4f}",'Recall(B)':f"{r:.4f}",'mAP50(B)':f"{metrics.box.map50:.4f}",'mAP50-95(B)':f"{metrics.box.map:.4f}",'F1-score(B)':f"{2*p*r/(p+r+1e-9):.4f}"})
        if hasattr(metrics, 'seg') and metrics.seg.map is not None:
            p_seg,r_seg=metrics.seg.mp,metrics.seg.mr; eval_results.update({'Precision(M)':f"{p_seg:.4f}",'Recall(M)':f"{r_seg:.4f}",'mAP50(M)':f"{metrics.seg.map50:.4f}",'mAP50-95(M)':f"{metrics.seg.map:.4f}",'F1-score(M)':f"{2*p_seg*r_seg/(p_seg+r_seg+1e-9):.4f}"})

        # --- 任務 2: 計算自訂的像素級別指標 ---
        pixel_metrics = calculate_pixel_level_metrics(model, exp_config, imgsz)
        eval_results.update(pixel_metrics)

        # --- 任務 3: 產生並儲存分類化的預測結果 (TP/FP/FN/TN) ---
        # 這是我們新增的核心功能，用於詳細的錯誤分析
        generate_categorized_predictions(model, exp_config, imgsz, results_path)

        # --- 任務 4: (可選) 執行影像重組評估 ---
        recon_config = exp_config.get('reconstruction')
        if recon_config and recon_config.get('enabled'):
            dataset_cfg = exp_config.get('dataset', {})
            project_root = Path.cwd()
            base_path = project_root / dataset_cfg.get('path', '').lstrip('./')
            patch_test_dir = base_path / dataset_cfg.get('test', '')
            
            original_data_root = Path(recon_config['original_data_root']).resolve()
            vis_params = {
                'min_conf': recon_config.get('overlay_min_conf', 0.25),
                'nms_iou': recon_config.get('overlay_nms_iou', 0.6),
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