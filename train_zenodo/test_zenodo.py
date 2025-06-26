import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO
from tqdm import tqdm

# 定義顏色 (BGR 格式)
BLUE = (255, 0, 0)      # 重疊區 (GT=Yes, Pred=Yes)
RED = (0, 0, 255)      # 僅預測區輪廓 (GT=No, Pred=Yes)
WHITE = (255, 255, 255) # 僅 GT 輪廓 (GT=Yes, Pred=No)
YELLOW = (0, 255, 255) # BBox 顏色
TEXT_COLOR = (0, 0, 0) # 黑色 (文字, 放在黃色背景上較清晰)

def create_visualizations_with_transparency_and_outlines(
    model_path: str,
    image_folder: str,
    label_folder: str,
    output_folder: str,
    img_width: int = 2048,
    img_height: int = 2048,
    inference_size: int = 1024,
    alpha: float = 0.3, # <--- 預設遮罩透明度 (0.0 完全透明, 1.0 完全不透明)
    conf_threshold: float = 0.4,
    outline_thickness: int = 2, # <--- 輪廓線粗細
    bbox_thickness: int = 4   # <--- BBox 線粗細 (加大)
):
    """
    產生詳細視覺化結果，包含半透明遮罩、BBox、信賴度及不同顏色輪廓。

    - 重疊區 (GT=Yes, Pred=Yes): 藍色半透明遮罩
    - 僅預測區 (GT=No, Pred=Yes): 紅色輪廓
    - 僅 GT 區 (GT=Yes, Pred=No): 白色輪廓
    - BBox: 黃色粗框

    Args:
        model_path (str): 模型路徑。
        image_folder (str): 圖片來源路徑。
        label_folder (str): 標註來源路徑。
        output_folder (str): 輸出路徑。
        img_width (int, optional): 圖片寬度 (目前程式碼會自動抓取，此參數暫無作用). Defaults to 2048.
        img_height (int, optional): 圖片高度 (目前程式碼會自動抓取，此參數暫無作用). Defaults to 2048.
        inference_size (int, optional): 推論尺寸. Defaults to 1024.
        alpha (float, optional): 遮罩透明度. Defaults to 0.4.
        conf_threshold (float, optional): BBox 信賴度閾值. Defaults to 0.25.
        outline_thickness (int, optional): 輪廓線粗細. Defaults to 2.
        bbox_thickness (int, optional): BBox 線粗細. Defaults to 3.
    """
    print("-" * 50)
    print(f"開始處理: {os.path.basename(model_path)}")
    print(f"  - 輸出至: {output_folder}")
    print("-" * 50)

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"錯誤：無法載入模型 {model_path}。 {e}")
        return

    os.makedirs(output_folder, exist_ok=True)
    image_files = glob.glob(os.path.join(image_folder, '*.png'))
    image_files.extend(glob.glob(os.path.join(image_folder, '*.jpg')))

    if not image_files:
        print(f"警告：在 '{image_folder}' 中找不到任何圖片檔案。")
        return

    print(f"找到 {len(image_files)} 張圖片，開始處理...")

    for img_path in tqdm(image_files, desc=f"視覺化 {os.path.basename(model_path)}"):
        try:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_folder, base_name + '.txt')
            img = cv2.imread(img_path)

            if img is None:
                print(f"\n警告：無法讀取圖片 {os.path.basename(img_path)}，已跳過。")
                continue

            h, w, _ = img.shape
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            pred_mask = np.zeros((h, w), dtype=np.uint8)

            # --- 繪製 GT 遮罩 ---
            if os.path.exists(label_path):
                with open(label_path, 'r') as f: lines = f.readlines()
                for line in lines:
                    try:
                        parts = line.strip().split()
                        coords = [float(p) for p in parts[1:]]
                        polygon = [[int(coords[i] * w), int(coords[i+1] * h)]
                                   for i in range(0, len(coords), 2)]
                        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(gt_mask, [pts], color=255)
                    except Exception as e_label:
                         print(f"\n警告：處理標註檔 {label_path} 時發生錯誤: {e_label}")

            # --- 進行預測 ---
            results = model.predict(img_path, imgsz=inference_size, conf=conf_threshold, verbose=False)
            r = results[0] if results and results[0] else None

            # --- 繪製 Pred 遮罩 ---
            if r and r.masks:
                for mask_polygon in r.masks.xy:
                    pts = np.array(mask_polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(pred_mask, [pts], color=255)

            # --- 計算區域 ---
            intersection_mask = cv2.bitwise_and(gt_mask, pred_mask)
            prediction_only_mask = cv2.bitwise_and(pred_mask, cv2.bitwise_not(gt_mask))
            gt_only_mask = cv2.bitwise_and(gt_mask, cv2.bitwise_not(pred_mask))

            # --- 上色與混合 ---
            output_img = img.copy() # 從原始圖片開始
            overlay = output_img.copy()

            # 將重疊區域填上藍色
            overlay[intersection_mask == 255] = BLUE

            # 將藍色遮罩以 alpha 透明度疊加到輸出圖片上
            cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0, output_img)

            # --- 繪製輪廓 ---
            # 尋找並繪製 "僅預測區" 的紅色輪廓
            pred_contours, _ = cv2.findContours(prediction_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_img, pred_contours, -1, RED, outline_thickness)

            # 尋找並繪製 "僅 GT 區" 的白色輪廓
            gt_contours, _ = cv2.findContours(gt_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_img, gt_contours, -1, WHITE, outline_thickness)

            # --- 繪製 BBox 和信賴度 ---
            if r and r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = model.names[cls_id]
                    label = f"{cls_name} {conf:.2f}"

                    # 繪製黃色、較粗的 BBox
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), YELLOW, bbox_thickness)

                    # 繪製標籤 (黃色背景，黑色文字)
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    text_y = y1 - 10 if y1 - 10 > 10 else y1 + label_height + 10
                    text_x = x1
                    cv2.rectangle(output_img, (text_x, text_y - label_height - baseline), (text_x + label_width, text_y + baseline), YELLOW, -1)
                    cv2.putText(output_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

            # --- 儲存結果 ---
            output_path = os.path.join(output_folder, os.path.basename(img_path))
            cv2.imwrite(output_path, output_img)

        except Exception as e:
            print(f"\n處理 {os.path.basename(img_path)} 時發生嚴重錯誤: {e}")

    print(f"'{os.path.basename(model_path)}' 的視覺化處理完成！\n")


# --- 如何呼叫這個函式 ---
if __name__ == '__main__':
    print("開始執行批次視覺化任務 (半透明 + 輪廓 + 黃色 BBox)...")

    # --- 定義您的測試集路徑 ---
    # !!! 請確保路徑中的反斜線 \ 被正確處理，建議使用正斜線 / 或雙反斜線 \\ !!!
    TEST_IMAGES = 'All_split_dataset_zenodo_onlyVV/images/test'
    TEST_LABELS = 'All_split_dataset_zenodo_onlyVV/labels/test'

    models_to_process = [
        # {
        #     'model_path': 'runs/segment/yolo11n-seg-zenodo2/weights/best.pt',
        #     'output_folder': 'testing_result/zenodo_yolo11n_500epoch/test_visualize_exp1_new'
        # },
        # {
        #     'model_path': 'runs/segment/yolo11m-seg-zenodo/weights/best.pt',
        #     'output_folder': 'testing_result/zenodo_yolo11m/test_visualize_exp2_new'
        # },
        {
            'model_path': 'runs/segment/yolo11n-seg-datasetv2-zenodo-1000epoch-onlyVV/weights/best.pt',
            'output_folder': 'testing_result/yolo11n-seg-datasetv2-zenodo-1000epoch-onlyVV/test_visualize_exp' 
        },
    ]

    for config in models_to_process:
        create_visualizations_with_transparency_and_outlines( # <--- 呼叫新的函式
            model_path=config['model_path'],
            image_folder=TEST_IMAGES,
            label_folder=TEST_LABELS,
            output_folder=config['output_folder']
        )

    print("所有視覺化任務執行完畢。")