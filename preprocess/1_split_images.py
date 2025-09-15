import os
from PIL import Image
from tqdm import tqdm

# --- 使用者設定 ---
# 來源資料夾：存放您原始大張 PNG 圖片的地方
SOURCE_FOLDER = r"OIL_Another_Test/check_samples_SAR3"
# 輸出資料夾：用來存放所有切割後圖片的根目錄
OUTPUT_FOLDER = r"OIL_Another_Test/check_samples_SAR3_Slice_9_grids"

Image.MAX_IMAGE_PIXELS = None

def split_images_into_grids(source_dir, output_dir):
    """
    將來源資料夾中的所有 PNG 圖片切割成 3x3 的九宮格，
    並為每張原始圖片建立一個獨立的資料夾來存放其 9 個圖塊。
    """
    print(f"--- 步驟一：開始將圖片切割為 3x3 (九宮格) ---")
    os.makedirs(output_dir, exist_ok=True)
    print(f"來源資料夾: {os.path.abspath(source_dir)}")
    print(f"輸出根目錄: {os.path.abspath(output_dir)}")

    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.png')]
    if not image_files:
        print(f"錯誤：在來源資料夾中找不到任何 PNG 圖片。")
        return

    print(f"共找到 {len(image_files)} 張圖片，準備進行切割...")

    for filename in tqdm(image_files, desc="切割進度"):
        try:
            base_name = os.path.splitext(filename)[0]
            
            # --- 修改點 (1/2)：為每張原始圖片建立一個專屬的輸出子資料夾 ---
            specific_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(specific_output_dir, exist_ok=True)
            # --- 修改結束 ---

            with Image.open(os.path.join(source_dir, filename)) as img:
                width, height = img.size
                
                # --- 修改點 (2/2)：將切割邏輯從 2x2 改為 3x3 ---
                # 計算 3x3 網格的切割點
                x_points = [0, width // 3, (width * 2) // 3, width]
                y_points = [0, height // 3, (height * 2) // 3, height]

                # 使用迴圈來切割 9 個象限
                for row in range(3):
                    for col in range(3):
                        # 定義切割區域 (left, upper, right, lower)
                        box = (x_points[col], y_points[row], x_points[col+1], y_points[row+1])
                        
                        grid_img = img.crop(box)
                        
                        # 命名規則改為 _row_col，例如 _00, _01, _02, _10 ...
                        output_filename = f"{base_name}_{row}{col}.png"
                        
                        # 將圖塊儲存到專屬的子資料夾中
                        grid_img.save(os.path.join(specific_output_dir, output_filename))
                # --- 修改結束 ---

        except Exception as e:
            tqdm.write(f"處理檔案 '{filename}' 時發生錯誤: {e}")

    print("\n--- 所有圖片切割完成！ ---")
    print(f"下一步：請到 '{output_dir}' 的各個子資料夾中使用 LabelMe 進行標註。")


if __name__ == "__main__":
    split_images_into_grids(SOURCE_FOLDER, OUTPUT_FOLDER)