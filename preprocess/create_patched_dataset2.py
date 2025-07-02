# python create_patched_dataset2.py
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

# PIL 可能會對大圖有警告，設定此項可避免
Image.MAX_IMAGE_PIXELS = None

# --- 來源與輸出設定 (請根據您的環境修改) ---
# 來源：包含 train/val/test 的原始、未切割的資料夾
SOURCE_BASE_DIR = "data/No_patch"
# 輸出：將在哪裡創建新的 patch 後資料夾
OUTPUT_BASE_DIR = "data"
CATEGORIES = ["SAR"] # 可新增 "IR" 等
SPLITS = ["train", "val", "test"]

# --- Patching 參數設定 ---
PATCH_SIZE = 256
OVERLAP = 128

# --- 背景樣本保留比例設定 ---
# 1.0 = 保留所有不含目標的 patches
# 0.2 = 隨機保留 20% 不含目標的 patches
BACKGROUND_KEEP_RATIO = 0.2

# --- TXT 生成參數 ---
TXT_GENERATION_PARAMS = {
    "class_id": 0,
    "target_pixel_value": 255,
    "epsilon_factor": 0.002,
    "min_contour_area": 1.0
}

# ==============================================================================
# --- 核心功能函數 ---
# ==============================================================================

def slice_image_with_metadata_generator(image_pil, mask_pil, patch_size, overlap):
    """
    生成器：一次 yield 一個 patch 及其在原圖中的 (x, y) 左上角座標。
    """
    image_width, image_height = image_pil.size
    stride = patch_size - overlap

    for y in range(0, image_height, stride):
        for x in range(0, image_width, stride):
            # 邊界處理：確保 patch 不會超出右側和底部
            actual_x = x
            actual_y = y
            if x + patch_size > image_width:
                actual_x = image_width - patch_size
            if y + patch_size > image_height:
                actual_y = image_height - patch_size
            
            box = (actual_x, actual_y, actual_x + patch_size, actual_y + patch_size)
            patch_img = image_pil.crop(box)
            patch_seg = mask_pil.crop(box)
            
            yield patch_img, patch_seg, actual_x, actual_y

            if actual_x == image_width - patch_size:
                break
        if actual_y == image_height - patch_size:
            break

def check_target_pixels_exist(mask_segment_pil, target_value):
    return np.any(np.array(mask_segment_pil) == target_value)

def convert_mask_png_to_yolo_txt(png_path, output_txt_path, params):
    try:
        mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: return "error"
        height, width = mask.shape
        if height == 0 or width == 0: return "error"
        
        _, binary_mask = cv2.threshold(mask, params["target_pixel_value"] - 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_format_lines = []
        for contour in contours:
            if cv2.contourArea(contour) < params["min_contour_area"]: continue
            epsilon = params["epsilon_factor"] * cv2.arcLength(contour, True)
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx_polygon) >= 3:
                normalized_points = [f"{p[0][0] / width:.6f} {p[0][1] / height:.6f}" for p in approx_polygon]
                yolo_format_lines.append(f"{params['class_id']} {' '.join(normalized_points)}")
                
        with open(output_txt_path, 'w') as f:
            if yolo_format_lines:
                f.write("\n".join(yolo_format_lines) + "\n")
        return "success"
    except Exception as e:
        print(f"  [ERROR] Convert to TXT failed: {e}")
        return "error"

# ==============================================================================
# --- 主處理流程 ---
# ==============================================================================
def main():
    output_dir_name = f"Patched_Data_P{PATCH_SIZE}_O{OVERLAP}_BGKeep{int(BACKGROUND_KEEP_RATIO*100)}p"
    main_output_path = os.path.join(OUTPUT_BASE_DIR, output_dir_name)
    print(f"所有 patch 後的資料將儲存到: {main_output_path}\n")

    for category in CATEGORIES:
        for split in SPLITS:
            print(f"--- 開始處理: Category '{category}', Split '{split}' ---")
            source_img_dir = os.path.join(SOURCE_BASE_DIR, category, "images", split)
            source_mask_dir = os.path.join(SOURCE_BASE_DIR, category, "labels", split)
            output_img_dir = os.path.join(main_output_path, category, "images", split)
            output_label_dir = os.path.join(main_output_path, category, "labels", split)

            if not os.path.isdir(source_img_dir) or not os.path.isdir(source_mask_dir):
                print(f"  [警告] 來源路徑 '{source_img_dir}' or '{source_mask_dir}' 不存在，跳過。")
                continue
                
            os.makedirs(output_img_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)
            
            image_files = [f for f in os.listdir(source_img_dir) if f.lower().endswith(('.jpg', '.png'))]
            
            for image_filename in tqdm(image_files, desc=f"Processing {category}/{split}"):
                base_filename = os.path.splitext(image_filename)[0]
                image_path = os.path.join(source_img_dir, image_filename)
                mask_path = os.path.join(source_mask_dir, base_filename + ".png")

                if not os.path.exists(mask_path): continue

                try:
                    with Image.open(image_path).convert("RGB") as img_pil, Image.open(mask_path).convert("L") as mask_pil:
                        patch_generator = slice_image_with_metadata_generator(img_pil, mask_pil, PATCH_SIZE, OVERLAP)
                        
                        for patch_img, patch_mask, x, y in patch_generator:
                            patch_base_name = f"{base_filename}__patch_x{x}_y{y}"
                            
                            has_target = check_target_pixels_exist(patch_mask, TXT_GENERATION_PARAMS["target_pixel_value"])
                            
                            if has_target or (random.random() < BACKGROUND_KEEP_RATIO):
                                output_jpg_path = os.path.join(output_img_dir, f"{patch_base_name}.jpg")
                                output_txt_path = os.path.join(output_label_dir, f"{patch_base_name}.txt")
                                
                                patch_img.save(output_jpg_path)
                                
                                if has_target:
                                    temp_png_path = Path(output_label_dir) / f"{patch_base_name}_temp.png"
                                    patch_mask.save(temp_png_path)
                                    convert_mask_png_to_yolo_txt(temp_png_path, output_txt_path, TXT_GENERATION_PARAMS)
                                    os.remove(temp_png_path)
                                else:
                                    open(output_txt_path, 'w').close()
                except Exception as e:
                    print(f"\n[ERROR] 處理檔案 '{image_filename}' 時發生嚴重錯誤: {e}")
    print("\n所有任務已完成！")

if __name__ == "__main__":
    main()