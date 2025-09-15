import tifffile
import os
import numpy as np
from PIL import Image
import sys
import cv2

def normalize_float_to_uint8(data):
    """將浮點數陣列正規化到 0-255 的 uint8 範圍"""
    data = np.nan_to_num(data) # 處理 NaN 或 Inf
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.full(data.shape, 0, dtype=np.uint8) # 如果都一樣，設為 0
    scaled_data = 255.0 * (data - min_val) / (max_val - min_val)
    return scaled_data.astype(np.uint8)

def process_image_tiff(tiff_path, vh_png_path, vv_png_path, npy_output_path):
    """處理原始影像 TIF (float32, 2ch) -> 儲存 VH, VV 的 PNG & 原始 NPY"""
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            data = tif.pages[0].asarray()

        if not (data.dtype == np.float32 and len(data.shape) == 3 and data.shape[2] == 2):
            print(f"  [警告] 影像 {os.path.basename(tiff_path)} 格式不符 (需要 float32, 2ch)，跳過。")
            return False

        os.makedirs(os.path.dirname(npy_output_path), exist_ok=True)
        np.save(npy_output_path, data)

        vh = data[:, :, 0]
        vv = data[:, :, 1]
        vh_norm = normalize_float_to_uint8(vh)
        vv_norm = normalize_float_to_uint8(vv)

        img_pil_vh = Image.fromarray(vh_norm, 'L')
        os.makedirs(os.path.dirname(vh_png_path), exist_ok=True)
        img_pil_vh.save(vh_png_path, "PNG")

        img_pil_vv = Image.fromarray(vv_norm, 'L')
        os.makedirs(os.path.dirname(vv_png_path), exist_ok=True)
        img_pil_vv.save(vv_png_path, "PNG")

        return True

    except Exception as e:
        print(f"  [錯誤] 處理影像 {os.path.basename(tiff_path)} 失敗: {e}")
        return False

def process_mask_tiff(tiff_path, png_output_path, npy_output_path):
    """處理遮罩 TIF (uint8, 1ch) -> PNG & NPY, 並回傳 0/1 遮罩"""
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            data = tif.pages[0].asarray()

        if not (data.dtype == np.uint8 and (len(data.shape) == 2 or (len(data.shape) == 3 and data.shape[2] == 1))):
            print(f"  [警告] 遮罩 {os.path.basename(tiff_path)} 格式不符，跳過。")
            return False, None

        if len(data.shape) == 3:
            data = data.squeeze()

        mask_01 = (data > 0).astype(np.uint8)
        os.makedirs(os.path.dirname(npy_output_path), exist_ok=True)
        np.save(npy_output_path, mask_01)

        mask_255 = mask_01 * 255
        img_pil = Image.fromarray(mask_255, 'L')
        os.makedirs(os.path.dirname(png_output_path), exist_ok=True)
        img_pil.save(png_output_path, "PNG")
        return True, mask_01

    except Exception as e:
        print(f"  [錯誤] 處理遮罩 {os.path.basename(tiff_path)} 失敗: {e}")
        return False, None

def generate_yolo_txt(mask_01, txt_output_path, class_id=0):
    """從 0/1 遮罩產生 YOLO 格式的 TXT 標註檔"""
    try:
        if mask_01 is None or np.max(mask_01) == 0:
            return True

        h, w = mask_01.shape
        contours, _ = cv2.findContours(mask_01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return True

        os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)
        with open(txt_output_path, 'w') as f:
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                width_norm = bw / w
                height_norm = bh / h
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        return True

    except Exception as e:
        print(f"  [錯誤] 產生 YOLO TXT {os.path.basename(txt_output_path)} 失敗: {e}")
        return False

def find_files_and_map(base_input_dir):
    """掃描指定的六個資料夾，建立影像和遮罩的對應表"""
    image_map = {}
    mask_map = {}
    target_folders = [
        "Train_Val_Lookalike_images", "Train_Val_Lookalike_mask",
        "Train_Val_No_Oil_Images", "Train_Val_No_Oil_mask",
        "Train_Val_Oil_Spill_images", "Train_Val_Oil_Spill_mask"
    ]
    print("正在掃描指定的六個資料夾...")
    for folder_name in target_folders:
        top_folder_path = os.path.join(base_input_dir, folder_name)
        if not os.path.isdir(top_folder_path):
            print(f"[警告] 指定的資料夾不存在，跳過: {top_folder_path}")
            continue

        is_mask = '_mask' in folder_name.lower()
        category = folder_name.replace('_images', '').replace('_Images', '').replace('_mask', '').replace('_Mask', '')
        print(f"  [掃描] {folder_name} | 類別: {category}")

        for dirpath, _, filenames in os.walk(top_folder_path):
            for filename in filenames:
                if filename.lower().endswith(('.tif', '.tiff')):
                    full_path = os.path.join(dirpath, filename)
                    key = (category, filename)
                    if is_mask:
                        mask_map[key] = full_path
                    else:
                        image_map[key] = full_path
    
    print("-" * 30)
    print(f"掃描完成：找到 {len(image_map)} 個影像檔，{len(mask_map)} 個遮罩檔。")
    return image_map, mask_map

def process_dataset(base_input_dir, base_output_dir):
    """
    (修改) 主處理函數，掃描並轉換整個資料集，並產生指定結構的輸出。
    """
    print(f"開始處理資料集於: {base_input_dir}")
    print(f"所有輸出將存放於: {base_output_dir}")
    print("-" * 30)
    
    # 定義新的輸出子目錄
    img_vv_dir = os.path.join(base_output_dir, "images", "VV")
    img_vh_dir = os.path.join(base_output_dir, "images", "VH")
    mask_dir = os.path.join(base_output_dir, "masks") # 遮罩和標籤共用此目錄
    npy_img_dir = os.path.join(base_output_dir, "npy", "images")
    npy_mask_dir = os.path.join(base_output_dir, "npy", "masks")
    
    image_map, mask_map = find_files_and_map(base_input_dir)
    processed_count = 0
    error_count = 0

    print("開始進行檔案配對與轉換...")
    for key, img_filepath in image_map.items():
        category, filename = key

        if key in mask_map:
            mask_filepath = mask_map[key]

            base_name, _ = os.path.splitext(filename)
            img_rel_path = os.path.relpath(os.path.dirname(img_filepath), base_input_dir)
            mask_rel_path = os.path.relpath(os.path.dirname(mask_filepath), base_input_dir)

            # 定義各類輸出路徑
            img_vh_out = os.path.join(img_vh_dir, img_rel_path, f"{base_name}.png")
            img_vv_out = os.path.join(img_vv_dir, img_rel_path, f"{base_name}.png")
            mask_png_out = os.path.join(mask_dir, mask_rel_path, f"{base_name}.png")
            img_npy_out = os.path.join(npy_img_dir, img_rel_path, f"{base_name}.npy")
            mask_npy_out = os.path.join(npy_mask_dir, mask_rel_path, f"{base_name}.npy")
            
            # (***修改處***) TXT 的路徑現在與遮罩 PNG 的路徑在同一個資料夾下
            txt_out = os.path.join(mask_dir, mask_rel_path, f"{base_name}.txt")

            # 處理影像和遮罩
            img_ok = process_image_tiff(img_filepath, img_vh_out, img_vv_out, img_npy_out)
            mask_ok, mask_data = process_mask_tiff(mask_filepath, mask_png_out, mask_npy_out)
            
            txt_ok = True
            if img_ok and mask_ok:
                if 'Oil_Spill' in category and mask_data is not None and np.max(mask_data) > 0:
                    txt_ok = generate_yolo_txt(mask_data, txt_out, class_id=0)
            
            if img_ok and mask_ok and txt_ok:
                processed_count += 1
            else:
                error_count += 1
                print(f"  [錯誤] 處理 {filename} (類別: {category}) 時發生問題。")

        else:
            print(f"[警告] 影像 {img_filepath} (檔名: {filename}) 找不到對應遮罩，跳過。")
            error_count += 1

    print("-" * 30)
    print(f"處理完成！")
    print(f"成功處理 {processed_count} 組檔案。")
    print(f"發生 {error_count} 次錯誤或警告。")

# --- 主程式設定 ---
if __name__ == "__main__":
    # 1. 輸入資料集的根目錄 (此目錄下應包含 Train_Val_..._images 等六個資料夾)
    BASE_INPUT_DIRECTORY = r"E:\data\input_dataset"

    # 2. 所有輸出檔案的根目錄 (腳本會在此目錄下自動建立 'images', 'masks' 等資料夾)
    BASE_OUTPUT_DIRECTORY = r"E:\data\output_dataset"
    
    # ################################################

    process_dataset(BASE_INPUT_DIRECTORY, BASE_OUTPUT_DIRECTORY)