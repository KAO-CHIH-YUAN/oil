import os
import numpy as np
from PIL import Image
import random
import shutil
import cv2 # 匯入 OpenCV
import tifffile # 僅用於掃描配對

def find_files_and_map_v5(base_input_dir):
    """(同 V10/V13) 掃描指定的六個 TIF 資料夾，建立影像和遮罩的對應表"""
    image_map = {}
    mask_map = {}
    target_folders = [
        "Train_Val_Lookalike_images", "Train_Val_Lookalike_mask",
        "Train_Val_No_Oil_Images", "Train_Val_No_Oil_mask",
        "Train_Val_Oil_Spill_images", "Train_Val_Oil_Spill_mask"
    ]
    print("正在掃描原始 TIF 資料夾以建立檔案清單...")
    for top_folder_name in target_folders:
        top_folder_path = os.path.join(base_input_dir, top_folder_name)
        if not os.path.isdir(top_folder_path):
            print(f"[警告] 指定的 TIF 資料夾不存在，跳過: {top_folder_path}")
            continue
        is_mask = '_mask' in top_folder_name.lower()
        is_image = '_images' in top_folder_name.lower()
        category = top_folder_name.replace('_images', '').replace('_Images', '').replace('_mask', '').replace('_Mask', '')
        subfolders = [d for d in os.listdir(top_folder_path) if os.path.isdir(os.path.join(top_folder_path, d))]
        if len(subfolders) != 1:
            print(f"    [警告] {top_folder_path} 中找到 {len(subfolders)} 個子資料夾 (應為 1)，跳過。")
            continue
        subfolder_path = os.path.join(top_folder_path, subfolders[0])
        for filename in os.listdir(subfolder_path):
            if filename.lower().endswith(('.tif', '.tiff')):
                full_path = os.path.join(subfolder_path, filename)
                key = (category, filename) 
                if is_image: image_map[key] = full_path
                elif is_mask: mask_map[key] = full_path
    print(f"掃描完成：找到 {len(image_map)} 個 TIF 影像檔，{len(mask_map)} 個 TIF 遮罩檔。")
    return image_map, mask_map

# *** 新增函數：PNG (0/255) 轉 TXT ***
def convert_png_to_yolo_txt(png_path, txt_path, img_width=2048, img_height=2048):
    """將 0/255 PNG 遮罩轉換為 YOLO TXT 格式"""
    try:
        # 讀取 PNG 並轉為灰階
        mask_pil = Image.open(png_path).convert('L')
        mask_np = np.array(mask_pil)

        # 二值化：將 0/255 轉為 0/1
        mask_01 = (mask_np > 0).astype(np.uint8) 

        # 如果全是 0，建立空 TXT
        if np.max(mask_01) == 0:
            open(txt_path, 'w').close()
            return True

        # 尋找輪廓 (在 0/1 影像上找)
        contours, _ = cv2.findContours(mask_01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 寫入 TXT
        with open(txt_path, 'w') as f:
            for contour in contours:
                if cv2.contourArea(contour) < 10: continue
                points = contour.squeeze()
                if len(points.shape) < 2 : continue 
                normalized_points = [f"{p[0] / img_width:.6f} {p[1] / img_height:.6f}" for p in points]
                f.write(f"0 {' '.join(normalized_points)}\n")
        
        return True

    except Exception as e:
        print(f"  [錯誤] 轉換 PNG {os.path.basename(png_path)} 到 TXT 失敗: {e}")
        return False

# *** 修改函數：從 PNG 產生 TXT 並複製 ***
def split_and_create_txt_from_png(base_input_dir, png_dir, yolo_output_dir, image_map, mask_map, seed=42):
    """
    分割資料集，複製影像 PNG，並從遮罩 PNG 產生 TXT 標籤。
    """
    print("\n" + "=" * 30)
    print("開始分割資料集 (PNG -> TXT)...")
    print(f"從 PNG 目錄: {png_dir}")
    print(f"YOLO 輸出至: {yolo_output_dir}")
    print("=" * 30)

    category_prefix_map = {
        'Train_Val_Lookalike': 'looklike',
        'Train_Val_No_Oil': 'no_oil',
        'Train_Val_Oil_Spill': 'oil'
    }

    file_pairs = []
    for key, img_tif_path in image_map.items():
        if key in mask_map:
            mask_tif_path = mask_map[key]
            file_pairs.append({'key': key, 'img_tif': img_tif_path, 'mask_tif': mask_tif_path})
        else:
            print(f"[警告] TIF 影像 {img_tif_path} (Key: {key}) 找不到對應遮罩。")
    print(f"共有 {len(file_pairs)} 組有效的影像-遮罩對將進行分割。")

    random.seed(seed)
    random.shuffle(file_pairs)

    total_files = len(file_pairs)
    train_split = int(0.64 * total_files)
    val_split = int(0.16 * total_files)
    train_end_index = train_split
    val_end_index = train_split + val_split
    print(f"總數: {total_files} | 訓練集: {train_end_index} | 驗證集: {val_split} | 測試集: {total_files - val_end_index}")

    sets = ['train', 'val', 'test']
    for s in sets:
        os.makedirs(os.path.join(yolo_output_dir, 'images', s), exist_ok=True)
        os.makedirs(os.path.join(yolo_output_dir, 'labels', s), exist_ok=True) # Labels 現在是 TXT
    print("YOLO 資料夾結構已建立。")

    print("開始複製影像並轉換遮罩為 TXT...")
    copied_count = 0
    error_count = 0

    for i, pair in enumerate(file_pairs):
        img_tif_path = pair['img_tif']
        mask_tif_path = pair['mask_tif']
        category, original_filename_tif = pair['key']

        if i < train_end_index: set_name = 'train'
        elif i < val_end_index: set_name = 'val'
        else: set_name = 'test'

        try:
            img_rel_path = os.path.relpath(img_tif_path, base_input_dir)
            mask_rel_path = os.path.relpath(mask_tif_path, base_input_dir)
            base_img_rel, _ = os.path.splitext(img_rel_path)
            base_mask_rel, _ = os.path.splitext(mask_rel_path)
            
            png_img_path = os.path.join(png_dir, f"{base_img_rel}.png")
            png_mask_path = os.path.join(png_dir, f"{base_mask_rel}.png") # *** 讀取 PNG 遮罩 ***

            if not os.path.exists(png_img_path) or not os.path.exists(png_mask_path):
                print(f"  [錯誤] 找不到 {png_img_path} 或 {png_mask_path}，跳過。")
                error_count += 1
                continue

            original_base = os.path.splitext(original_filename_tif)[0]
            prefix = category_prefix_map.get(category, 'unknown')
            
            new_img_filename = f"{prefix}_{original_base}.png"
            new_txt_filename = f"{prefix}_{original_base}.txt" # *** 標籤檔名是 .txt ***

            dest_img_path = os.path.join(yolo_output_dir, 'images', set_name, new_img_filename)
            dest_label_path = os.path.join(yolo_output_dir, 'labels', set_name, new_txt_filename)

            # 複製影像 PNG
            shutil.copy(png_img_path, dest_img_path)

            # *** 轉換 PNG 為 TXT ***
            success = convert_png_to_yolo_txt(png_mask_path, dest_label_path)
            
            if success:
                copied_count += 1
            else:
                error_count += 1

        except Exception as e:
            print(f"  [嚴重錯誤] 處理 {img_tif_path} 時失敗: {e}")
            error_count += 1
            
    print("-" * 30)
    print("YOLO 資料集建立完成！")
    print(f"成功複製/處理 {copied_count} 組檔案。")
    print(f"發生 {error_count} 次錯誤。")


# --- 主程式設定 ---
if __name__ == "__main__":
    BASE_INPUT_DIRECTORY = r"E:\yuan_oil\oil_spill_data\dataset_zenodo\original\Train_Val_Oil_Spill_images\Train_Val_Oil_Spill_images"
    PNG_INPUT_DIRECTORY = r"E:\yuan_oil\oil_spill_data\dataset_zenodo\processed\new_yuan2\png_vv_only" 
    YOLO_OUTPUT_DIRECTORY = r"E:\yuan_oil\oil_spill_data\dataset_zenodo\processed\All_split_dataset_zenodo_onlyVV" # 建議用新名稱

    # --- 1. 掃描 TIF 取得檔案清單與配對 ---
    image_map, mask_map = find_files_and_map_v5(BASE_INPUT_DIRECTORY)

    # --- 2. 執行分割與複製 (PNG -> TXT) ---
    if image_map and mask_map:
         split_and_create_txt_from_png(
             BASE_INPUT_DIRECTORY, 
             PNG_INPUT_DIRECTORY, 
             YOLO_OUTPUT_DIRECTORY, 
             image_map, 
             mask_map
         )
    else:
        print("未找到任何影像或遮罩，無法進行分割。請檢查 BASE_INPUT_DIRECTORY。")
