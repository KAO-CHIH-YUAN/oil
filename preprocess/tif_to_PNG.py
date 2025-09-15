import os
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_tif_to_png(input_folder, output_folder):
    """
    遞迴掃描 TIF 檔案並轉換為 PNG。
    此版本會智慧判斷 TIF 的資料類型：
    - 如果是 8-bit (uint8)，則直接複製數值，達成無損轉換。
    - 如果是高位元深度 (如 uint16)，則進行全域線性拉伸以保留細節。
    """
    print("--- 開始將 TIF 轉換為 PNG (智慧判斷模式) ---")
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"輸出檔案將儲存於: {os.path.abspath(output_folder)}")

    tif_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                tif_files.append(os.path.join(root, file))

    if not tif_files:
        print("錯誤：在來源資料夾中沒有找到任何 .tif 或 .tiff 檔案。")
        return

    print(f"共找到 {len(tif_files)} 個 TIF 檔案，準備開始轉換...")

    for tif_path in tqdm(tif_files, desc="轉換進度"):
        try:
            with rasterio.open(tif_path) as src:
                if src.count < 3:
                    tqdm.write(f"警告：檔案 '{os.path.basename(tif_path)}' 只有 {src.count} 個通道，不足 3 個，已跳過。")
                    continue
                
                # --- ★★★ 智慧判斷資料類型 ★★★ ---
                # 檢查第一個通道的資料類型
                source_dtype = src.dtypes[0]

                if source_dtype == 'uint8':
                    # 如果來源已是 8-bit，直接讀取，不需任何轉換
                    tqdm.write(f"資訊：檔案 '{os.path.basename(tif_path)}' 是 8-bit，進行直接數值複製。")
                    r_band = src.read(1)
                    g_band = src.read(2)
                    b_band = src.read(3)
                else:
                    # 如果來源是 16-bit 或其他類型，則進行全域線性拉伸
                    tqdm.write(f"資訊：檔案 '{os.path.basename(tif_path)}' 是 {source_dtype}，進行線性轉換。")
                    
                    def normalize_full_range(band):
                        min_val, max_val = band.min(), band.max()
                        if max_val == min_val:
                            return np.zeros(band.shape, dtype=np.uint8)
                        scaled_band = 255.0 * (band - min_val) / (max_val - min_val)
                        return scaled_band.astype(np.uint8)
                    
                    r_band = normalize_full_range(src.read(1))
                    g_band = normalize_full_range(src.read(2))
                    b_band = normalize_full_range(src.read(3))
                # --- ★★★ 修改結束 ★★★ ---

                # 將三個單獨的通道堆疊成一個 RGB 影像陣列
                rgb_image_array = np.dstack((r_band, g_band, b_band))
                img = Image.fromarray(rgb_image_array, 'RGB')

                # 建立輸出路徑並儲存為 PNG
                base_filename = os.path.splitext(os.path.basename(tif_path))[0]
                output_path = os.path.join(output_folder, f"{base_filename}.png")
                img.save(output_path, 'PNG')

        except Exception as e:
            tqdm.write(f"錯誤：處理檔案 '{os.path.basename(tif_path)}' 時發生問題: {e}")

    print("\n--- 所有檔案處理完成！ ---")


if __name__ == "__main__":
    input_dir = 'OIL_PROJECT/dataset/dataset_optical/optical_tif'
    output_dir = 'OIL_PROJECT/dataset/dataset_optical/optical_png'


    if not os.path.isdir(input_dir):
        print(f"錯誤：來源路徑 '{input_dir}' 不是一個有效的資料夾。")
    elif not output_dir:
        print("錯誤：輸出資料夾路徑不能為空。")
    else:
        convert_tif_to_png(input_dir, output_dir)
