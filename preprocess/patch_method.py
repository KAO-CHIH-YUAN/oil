# python create_patched_dataset_with_log.py
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import pandas as pd  # <--- 新增 import
from datetime import datetime  # <--- 新增 import

# PIL 可能會對大圖有警告，設定此項可避免
Image.MAX_IMAGE_PIXELS = None

# --- 來源與輸出設定 ---
SOURCE_BASE_DIR = "/home/yuan/OIL_PROJECT/dataset/dataset_zenodo"
OUTPUT_BASE_DIR = "/home/yuan/OIL_PROJECT/dataset/dataset_zenodo/zenodo_patch"
CATEGORIES = ["zenodo"] # SAR_2 
SPLITS = ["train", "val", "test"]

# --- Patching 參數設定 ---
PATCH_SIZE = 640
OVERLAP = 128

# --- 背景樣本保留比例設定 (僅對 train/val 有效) ---
BACKGROUND_KEEP_RATIO = 0.2

# --- PNG 轉 TXT 參數設定 ---
TXT_GENERATION_PARAMS = {
    "class_id": 0,
    "target_pixel_value": 255,
    "epsilon_factor": 0.002,
    "min_contour_area": 1.0
}

# (slice_image_with_padding_generator, check_target_pixels_exist, convert_mask_png_to_yolo_txt 函數保持不變)
def slice_image_with_padding_generator(image_pil, mask_pil, patch_size, overlap):
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
    segment_array = np.array(mask_segment_pil)
    return np.any(segment_array == target_value)

def convert_mask_png_to_yolo_txt(png_path, output_txt_path, params):
    try:
        mask = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: return "error"
        height, width = mask.shape
        if height == 0 or width == 0: return "error"
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask[mask == params["target_pixel_value"]] = 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yolo_format_lines = []
        for contour in contours:
            if cv2.contourArea(contour) < params["min_contour_area"]: continue
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
        print(f"  [ERROR] 處理檔案 '{png_path}' 時發生錯誤: {e}")
        return "error"


# ==============================================================================
# --- 3. 主處理流程 (整合 Excel Log) ---
# ==============================================================================

def main():
    """主執行函數"""
    output_dir_name = f"Patched_Data_P{PATCH_SIZE}_O{OVERLAP}_BGKeep{int(BACKGROUND_KEEP_RATIO*100)}p"
    main_output_path = os.path.join(OUTPUT_BASE_DIR, output_dir_name)
    print(f"所有 patch 後的資料將儲存到: {main_output_path}\n")

    # <--- 新增部分：準備 Excel Log ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_run_stats = [] # 用來儲存這次執行的所有統計數據

    for category in CATEGORIES:
        for split in SPLITS:
            print(f"--- 開始處理: Category '{category}', Split '{split}' ---")
            
            source_img_dir = os.path.join(SOURCE_BASE_DIR, category, "images", split)
            source_mask_dir = os.path.join(SOURCE_BASE_DIR, category, "labels", split)
            output_img_dir = os.path.join(main_output_path, category, "images", split)
            output_label_dir = os.path.join(main_output_path, category, "labels", split)
            
            if not os.path.isdir(source_img_dir) or not os.path.isdir(source_mask_dir):
                print(f"  [警告] 來源路徑不存在。跳過此組合。")
                print("-" * (20 + len(category) + len(split)))
                continue
                
            os.makedirs(output_img_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)
            
            effective_bg_keep_ratio = BACKGROUND_KEEP_RATIO
            if split == 'test':
                effective_bg_keep_ratio = 1.0
                print(f"  [Info] 檢測到 'test' split，將強制保留 100% 的背景圖塊以便後續重組。")

            image_files = [f for f in os.listdir(source_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            source_image_count = len(image_files) # <--- 取得原始圖片數量
            
            total_patches_with_oil = 0
            total_patches_background_kept = 0
            
            for image_filename in tqdm(image_files, desc=f"Processing {category}/{split}", unit="image"):
                # ... (內部處理迴圈維持不變) ...
                base_filename = os.path.splitext(image_filename)[0]
                image_path = os.path.join(source_img_dir, image_filename)
                mask_path = os.path.join(source_mask_dir, base_filename + ".png")
                if not os.path.exists(mask_path): continue
                try:
                    with Image.open(image_path) as img_pil_raw, Image.open(mask_path) as mask_pil_raw:
                        img_pil = img_pil_raw.convert("RGB")
                        mask_pil = mask_pil_raw.convert("L")
                        generated_patches = slice_image_with_padding_generator(img_pil, mask_pil, PATCH_SIZE, OVERLAP)
                        for patch_img, patch_mask, x, y in generated_patches:
                            patch_base_name = f"{base_filename}_patch_x{x}_y{y}"
                            should_save, is_background = False, False
                            if check_target_pixels_exist(patch_mask, TXT_GENERATION_PARAMS["target_pixel_value"]):
                                should_save, is_background = True, False
                            else:
                                if random.random() < effective_bg_keep_ratio:
                                    should_save, is_background = True, True
                            if should_save:
                                output_jpg_path = os.path.join(output_img_dir, f"{patch_base_name}.jpg")
                                output_png_path = os.path.join(output_label_dir, f"{patch_base_name}.png")
                                output_txt_path = os.path.join(output_label_dir, f"{patch_base_name}.txt")
                                patch_img.save(output_jpg_path)
                                patch_mask.save(output_png_path)
                                if not is_background:
                                    convert_mask_png_to_yolo_txt(output_png_path, output_txt_path, TXT_GENERATION_PARAMS)
                                    total_patches_with_oil += 1
                                else:
                                    with open(output_txt_path, 'w') as f: pass
                                    total_patches_background_kept += 1
                except Exception as e:
                    print(f"  [ERROR] 處理檔案 '{image_filename}' 時發生錯誤: {e}")
            
            print(f"  處理完成。")
            print(f"  原始圖片數量: {source_image_count}")
            print(f"  儲存了 {total_patches_with_oil} 個包含油污的 patches。")
            if split != 'test':
                print(f"  根據 {BACKGROUND_KEEP_RATIO*100:.0f}% 的比例，隨機保留了 {total_patches_background_kept} 個不含油污的 patches。")
            else:
                 print(f"  為 'test' split 保留了全部 {total_patches_background_kept} 個不含油污的 patches。")
            print(f"  '{category}/{split}' 總共產生 {total_patches_with_oil + total_patches_background_kept} 個樣本。")
            
            # <--- 新增部分：將這次的統計數據存入 list ---
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
                "Effective_BG_Keep_Ratio": effective_bg_keep_ratio
            }
            current_run_stats.append(stat_record)
            print("-" * (20 + len(category) + len(split)) + "\n")

    # <--- 新增部分：在所有處理完成後，寫入 Excel 檔案 ---
    if not current_run_stats:
        print("沒有處理任何資料，不產生 Log 檔案。")
        return

    excel_log_path = os.path.join(OUTPUT_BASE_DIR, "patch_generation_log.xlsx")
    new_stats_df = pd.DataFrame(current_run_stats)

    try:
        if os.path.exists(excel_log_path):
            print(f"偵測到現有 Log 檔案: {excel_log_path}，將附加新紀錄。")
            existing_df = pd.read_excel(excel_log_path)
            combined_df = pd.concat([existing_df, new_stats_df], ignore_index=True)
        else:
            print(f"正在創建新的 Log 檔案: {excel_log_path}")
            combined_df = new_stats_df

        combined_df.to_excel(excel_log_path, index=False)
        print("Excel Log 檔案已成功更新。")
    except Exception as e:
        print(f"[ERROR] 無法寫入 Excel Log 檔案: {e}")
        print("請檢查是否有權限寫入該目錄，或檔案是否被其他程式開啟。")
        
    print("\n所有任務已完成！")


if __name__ == "__main__":
    main()