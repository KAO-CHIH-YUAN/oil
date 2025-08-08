# python create_patched_dataset_with_log.py
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import pandas as pd
from datetime import datetime

# PIL 可能會對大圖有警告，設定此項可避免
Image.MAX_IMAGE_PIXELS = None

# --- 來源與輸出設定 ---
SOURCE_BASE_DIR = "/home/yuan/OIL_PROJECT/dataset/dataset_zenodo"
OUTPUT_BASE_DIR = "/home/yuan/OIL_PROJECT/dataset/dataset_zenodo/zenodo_original_all_classes_patch_png"
CATEGORIES = ["zenodo_original_all_classes"] # SAR_2 zenodo
SPLITS = ["train", "val", "test"]

# --- Patching 參數設定 ---
PATCH_SIZE = 640
OVERLAP = 128
RANDOM_SEED = 42

# --- 背景樣本保留比例設定 ---
BACKGROUND_KEEP_RATIO = 0.2

# --- 功能開關 ---
SEPARATE_OUTPUT = False # 是否將正負樣本分開儲存
 
# --- PNG 轉 TXT 參數設定 ---
TXT_GENERATION_PARAMS = {
    "class_id": 0,
    "target_pixel_value": 255,
    "epsilon_factor": 0.002,
    "min_contour_area": 1.0
}


def slice_image_with_padding_generator(image_pil, mask_pil, patch_size, overlap):
    """
    生成器函式：對輸入的影像和遮罩進行切割，一次產出一個圖塊 (patch)。
    - 如果原始影像小於圖塊尺寸，會自動進行置中填充。
    - 否則，會根據指定的 overlap 進行滑動窗口切割。
    """
    image_width, image_height = image_pil.size
    if image_width < patch_size or image_height < patch_size:
        pad_color_image = (0, 0, 0) if image_pil.mode == 'RGB' else 0
        pad_value_mask = 0
        padded_img = Image.new(image_pil.mode, (patch_size, patch_size), pad_color_image)
        padded_seg = Image.new(mask_pil.mode, (patch_size, patch_size), pad_value_mask)
        paste_x = (patch_size - image_width) // 2
        paste_y = (patch_size - image_height) // 2
        padded_img.paste(image_pil, (paste_x, paste_y))
        padded_seg.paste(mask_pil, (paste_x, paste_y))
        yield padded_img, padded_seg, 0, 0
    else:
        stride = patch_size - overlap
        for y in range(0, image_height, stride):
            for x in range(0, image_width, stride):
                actual_x, actual_y = x, y
                if x + patch_size > image_width: actual_x = image_width - patch_size
                if y + patch_size > image_height: actual_y = image_height - patch_size
                patch_img = image_pil.crop((actual_x, actual_y, actual_x + patch_size, actual_y + patch_size))
                patch_seg = mask_pil.crop((actual_x, actual_y, actual_x + patch_size, actual_y + patch_size))
                yield patch_img, patch_seg, actual_x, actual_y
                if actual_x == image_width - patch_size: break
            if actual_y == image_height - patch_size: break

def check_target_pixels_exist(mask_segment_pil, target_value):
    """
    檢查一個遮罩圖塊中，是否存在指定的目標像素值。
    """
    segment_array = np.array(mask_segment_pil)
    return np.any(segment_array == target_value)

def convert_mask_png_to_yolo_txt(png_path, output_txt_path, params):
    """
    將二值的 PNG 遮罩圖檔轉換為 YOLOv8 Segmentation 所需的 .txt 標籤格式。
    """
    try:
        mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: return "error"
        height, width = mask.shape
        if height == 0 or width == 0: return "error"
        
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask[mask == params["target_pixel_value"]] = 255
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_format_lines = []
        for contour in contours:
            if cv2.contourArea(contour) < params["min_contour_area"]:
                continue
            
            epsilon = params["epsilon_factor"] * cv2.arcLength(contour, True)
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx_polygon) >= 3:
                normalized_points = []
                for point_wrapper in approx_polygon:
                    point = point_wrapper[0]
                    norm_x = max(0.0, min(1.0, point[0] / width))
                    norm_y = max(0.0, min(1.0, point[1] / height))
                    normalized_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
                
                if normalized_points:
                    yolo_format_lines.append(f"{params['class_id']} {' '.join(normalized_points)}")
        
        with open(output_txt_path, 'w') as f:
            if yolo_format_lines:
                f.write("\n".join(yolo_format_lines) + "\n")
                return "contours_written"
            else:
                return "empty_file_written"
    except Exception as e:
        # 在原版 `yuan.py` 中，這裡使用了 Path，但未導入，可能導致錯誤。
        # 為穩健起見，改用 os.path.basename
        print(f"  [ERROR] 處理檔案 '{os.path.basename(png_path)}' 時發生錯誤: {e}")
        return "error"


def main():
    """主執行函數"""
    random.seed(RANDOM_SEED)
    
    # 建立一個唯一的輸出資料夾名稱
    output_dir_name = f"Patched_P{PATCH_SIZE}_O{OVERLAP}_BG{int(BACKGROUND_KEEP_RATIO*100)}p"
    if SEPARATE_OUTPUT:
        output_dir_name += "_Separated"
    main_output_path = os.path.join(OUTPUT_BASE_DIR, output_dir_name)
    
    print(f"所有 patch 後的資料將儲存到: {main_output_path}")
    print(f"使用固定的隨機數種子: {RANDOM_SEED}")
    if SEPARATE_OUTPUT:
        print("[資訊] SEPARATE_OUTPUT 已啟用，正負樣本將被分開儲存。")

    # 初始化日誌記錄所需變數
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_run_stats = []

    for category in CATEGORIES:
        for split in SPLITS:
            print(f"\n--- 開始處理: Category '{category}', Split '{split}' ---")
            
            source_img_dir = os.path.join(SOURCE_BASE_DIR, category, "images", split)
            source_mask_dir = os.path.join(SOURCE_BASE_DIR, category, "labels", split)
            
            if not os.path.isdir(source_img_dir) or not os.path.isdir(source_mask_dir):
                print(f"  [警告] 來源路徑不存在，跳過: {source_img_dir} 或 {source_mask_dir}")
                continue

            # 根據 SEPARATE_OUTPUT 的設定，動態決定輸出路徑
            if SEPARATE_OUTPUT:
                output_img_dir_pos = os.path.join(main_output_path, category, "images", f"{split}_pos")
                output_label_dir_pos = os.path.join(main_output_path, category, "labels", f"{split}_pos")
                os.makedirs(output_img_dir_pos, exist_ok=True); os.makedirs(output_label_dir_pos, exist_ok=True)

                output_img_dir_neg = os.path.join(main_output_path, category, "images", f"{split}_neg")
                output_label_dir_neg = os.path.join(main_output_path, category, "labels", f"{split}_neg")
                os.makedirs(output_img_dir_neg, exist_ok=True); os.makedirs(output_label_dir_neg, exist_ok=True)
            else:
                output_img_dir = os.path.join(main_output_path, category, "images", split)
                output_label_dir = os.path.join(main_output_path, category, "labels", split)
                os.makedirs(output_img_dir, exist_ok=True); os.makedirs(output_label_dir, exist_ok=True)

            effective_bg_keep_ratio = BACKGROUND_KEEP_RATIO
            # 強制 test 資料集保留所有背景樣本，以利公正評估
            if split == 'test': 
                effective_bg_keep_ratio = 1.0

            # ==================================================================
            # === 修改處：新增 sorted() 函數 ===
            # 確保圖片檔案按名稱排序，可提升在不同作業系統上執行的結果一致性。
            image_files = sorted([f for f in os.listdir(source_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            # ==================================================================
            
            source_image_count = len(image_files)
            total_patches_with_oil = 0
            total_patches_background_kept = 0
            
            for image_filename in tqdm(image_files, desc=f"Processing {category}/{split}", unit="image"):
                base_filename = os.path.splitext(image_filename)[0]
                image_path = os.path.join(source_img_dir, image_filename)
                mask_path = os.path.join(source_mask_dir, base_filename + ".png")
                if not os.path.exists(mask_path): 
                    continue
                    
                try:
                    with Image.open(image_path) as img_pil_raw, Image.open(mask_path) as mask_pil_raw:
                        img_pil = img_pil_raw.convert("RGB")
                        mask_pil = mask_pil_raw.convert("L")
                        generated_patches = slice_image_with_padding_generator(img_pil, mask_pil, PATCH_SIZE, OVERLAP)
                        for patch_img, patch_mask, x, y in generated_patches:
                            patch_base_name = f"{base_filename}_patch_x{x}_y{y}"
                            
                            has_target = check_target_pixels_exist(patch_mask, TXT_GENERATION_PARAMS["target_pixel_value"])
                            
                            should_save = has_target or (random.random() < effective_bg_keep_ratio)
                            if not should_save: 
                                continue

                            if SEPARATE_OUTPUT:
                                current_img_dir, current_label_dir = (output_img_dir_pos, output_label_dir_pos) if has_target else (output_img_dir_neg, output_label_dir_neg)
                            else:
                                current_img_dir, current_label_dir = output_img_dir, output_label_dir
                            
                            output_img_png_path = os.path.join(current_img_dir, f"{patch_base_name}.png") 
                            output_png_path = os.path.join(current_label_dir, f"{patch_base_name}.png")
                            output_txt_path = os.path.join(current_label_dir, f"{patch_base_name}.txt")
                            
                            patch_img.save(output_img_png_path) # <--- 修改此行，儲存為 PNG
                            patch_mask.save(output_png_path)
                            
                            if has_target:
                                convert_mask_png_to_yolo_txt(output_png_path, output_txt_path, TXT_GENERATION_PARAMS)
                                total_patches_with_oil += 1
                            else:
                                with open(output_txt_path, 'w') as f: 
                                    pass # 為背景樣本創建一個空的 .txt 標籤檔
                                total_patches_background_kept += 1
                except Exception as e:
                    print(f"\n  [ERROR] 處理檔案 '{image_filename}' 時發生嚴重錯誤: {e}")

            print(f"  處理完成。")
            print(f"  原始圖片數量: {source_image_count}")
            print(f"  儲存了 {total_patches_with_oil} 個包含油汙的 patches。")
            print(f"  保留了 {total_patches_background_kept} 個不含油汙的 patches。")
            
            # 收集該 split 的統計數據到日誌中
            stat_record = {
                "RunTimestamp": run_timestamp,
                "Category": category,
                "Split": split,
                "SourceImageCount": source_image_count,
                "PatchesWithObject": total_patches_with_oil,
                "BackgroundPatchesKept": total_patches_background_kept,
                "TotalPatchesGenerated": total_patches_with_oil + total_patches_background_kept,
                "PatchSize": PATCH_SIZE,
                "Overlap": OVERLAP,
                "BG_Keep_Ratio_Setting": BACKGROUND_KEEP_RATIO,
                "Effective_BG_Keep_Ratio": effective_bg_keep_ratio,
                "RandomSeed": RANDOM_SEED,
                "SEPARATE_OUTPUT_Enabled": SEPARATE_OUTPUT
            }
            current_run_stats.append(stat_record)
    
    # 執行結束後，將日誌寫入 Excel 檔案
    if not current_run_stats:
        print("\n沒有處理任何資料，不產生 Log 檔案。")
        return

    excel_log_path = os.path.join(OUTPUT_BASE_DIR, "patch_generation_log.xlsx")
    new_stats_df = pd.DataFrame(current_run_stats)

    try:
        if os.path.exists(excel_log_path):
            print(f"\n偵測到現有 Log 檔案: {excel_log_path}，將附加新紀錄。")
            existing_df = pd.read_excel(excel_log_path)
            combined_df = pd.concat([existing_df, new_stats_df], ignore_index=True)
        else:
            print(f"\n正在創建新的 Log 檔案: {excel_log_path}")
            combined_df = new_stats_df

        combined_df.to_excel(excel_log_path, index=False)
        print("Excel Log 檔案已成功更新。")
    except Exception as e:
        print(f"[ERROR] 無法寫入 Excel Log 檔案: {e}")
        
    print("\n所有任務已完成！")


if __name__ == "__main__":
    main()