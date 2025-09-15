import os
import rasterio
import numpy as np
from PIL import Image

# 避免 Pillow 處理大圖時產生警告
Image.MAX_IMAGE_PIXELS = None

def convert_single_channel_tif_to_rgb_png(input_path, output_folder):
    """
    讀取單一通道的 TIF 檔案，將其通道複製三份形成 RGB 圖像，
    並以無損的 PNG 格式儲存。
    """
    print("--- 開始處理單通道 TIF 檔案 ---")

    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    print(f"輸出檔案將儲存於: {os.path.abspath(output_folder)}")

    try:
        with rasterio.open(input_path) as src:
            # 檢查至少有一個通道
            if src.count < 1:
                print(f"錯誤：檔案 '{os.path.basename(input_path)}' 不包含任何通道。")
                return

            # 讀取第一個通道的資料
            gray_band = src.read(1)
            source_dtype = gray_band.dtype
            
            print(f"成功讀取檔案: '{os.path.basename(input_path)}' (資料類型: {source_dtype})")

            # 智慧判斷資料類型並正規化
            if source_dtype == 'uint8':
                # 如果來源已是 8-bit，直接使用，不需任何轉換
                print("偵測到 8-bit 資料，進行直接數值複製。")
                final_gray_band = gray_band
            else:
                # 如果來源是 16-bit 或其他類型，則進行全域線性拉伸
                print(f"偵測到 {source_dtype} 資料，進行線性轉換至 8-bit。")
                min_val, max_val = gray_band.min(), gray_band.max()
                if max_val == min_val:
                    final_gray_band = np.zeros(gray_band.shape, dtype=np.uint8)
                else:
                    scaled_band = 255.0 * (gray_band - min_val) / (max_val - min_val)
                    final_gray_band = scaled_band.astype(np.uint8)
            
            # --- ★★★ 核心步驟：將單一通道複製三份 ★★★ ---
            # 使用 np.dstack 將二維的灰階陣列堆疊成三維的 RGB 陣列
            print("正在將單一通道複製為三通道 (R=G=B)...")
            rgb_image_array = np.dstack((final_gray_band, final_gray_band, final_gray_band))

            # 使用 Pillow 從 numpy 陣列建立影像物件
            img = Image.fromarray(rgb_image_array, 'RGB')

            # 建立輸出路徑並儲存為 PNG
            base_filename = os.path.splitext(os.path.basename(input_path))[0]
            # 可以在檔名加上後綴以茲區別
            output_filename = f"{base_filename}.png" 
            output_path = os.path.join(output_folder, output_filename)
            
            # PNG 預設就是無損壓縮
            img.save(output_path, 'PNG')
            
            print("\n--- 處理完成！ ---")
            print(f"新的三通道 PNG 圖片已儲存至:\n{os.path.abspath(output_path)}")

    except FileNotFoundError:
        print(f"錯誤：找不到指定的檔案 '{input_path}'")
    except Exception as e:
        print(f"處理檔案時發生問題: {e}")


if __name__ == "__main__":
    print("--- 單通道 TIF 轉三通道 PNG 工具 ---")
    
    # 提示使用者輸入單一檔案的路徑
    input_file_path = '/home/yuan/OIL_PROJECT/dataset/dataset_SAR_3/S1_tif/S1A_20230824_Asc_4P_8b.tif' #input("請輸入要處理的單一 TIF 檔案完整路徑: ")
    output_dir_path = '/home/yuan/OIL_PROJECT/dataset/dataset_SAR_3/S1_PNG' #input("請輸入儲存 PNG 檔案的輸出資料夾路徑: ")

    if not os.path.isfile(input_file_path):
        print(f"錯誤：來源路徑 '{input_file_path}' 不是一個有效的檔案。")
    elif not output_dir_path:
        print("錯誤：輸出資料夾路徑不能為空。")
    else:
        convert_single_channel_tif_to_rgb_png(input_file_path, output_dir_path)