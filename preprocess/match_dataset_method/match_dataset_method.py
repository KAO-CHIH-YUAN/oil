import os
from PIL import Image
from tqdm import tqdm

def sync_and_convert_images(folder_a, folder_b, folder_c):
    """
    根據資料夾 A 的 .txt 檔名，將資料夾 B 中對應的 .jpg 檔案
    以 .png 格式複製到資料夾 C。

    Args:
        folder_a (str): 包含 .txt 檔案的來源資料夾 A。
        folder_b (str): 包含 .jpg 檔案的來源資料夾 B。
        folder_c (str): 儲存 .png 結果的輸出資料夾 C。
    """
    print("--- 開始比對檔案並進行轉換 ---")
    
    # 確保輸出資料夾存在
    os.makedirs(folder_c, exist_ok=True)
    print(f"基準資料夾 (A): {os.path.abspath(folder_a)}")
    print(f"圖片來源 (B): {os.path.abspath(folder_b)}")
    print(f"輸出位置 (C): {os.path.abspath(folder_c)}")

    # 1. 取得資料夾 A 中所有 .txt 檔案的主檔名，並存入一個集合 (set) 以快速查找
    try:
        a_files = os.listdir(folder_a)
        a_basenames = {os.path.splitext(f)[0] for f in a_files if f.lower().endswith('.txt')}
        print(f"\n在 A 資料夾中找到 {len(a_basenames)} 個 .txt 基準檔名。")
    except FileNotFoundError:
        print(f"錯誤：找不到資料夾 A: '{folder_a}'")
        return

    # 2. 準備掃描資料夾 B
    try:
        b_jpg_files = [f for f in os.listdir(folder_b) if f.lower().endswith(('.jpg', '.jpeg'))]
        print(f"在 B 資料夾中找到 {len(b_jpg_files)} 個 .jpg 圖片檔案，準備開始比對...")
    except FileNotFoundError:
        print(f"錯誤：找不到資料夾 B: '{folder_b}'")
        return
        
    if not b_jpg_files:
        print("資料夾 B 中沒有找到任何 .jpg 檔案，程式結束。")
        return

    # 3. 遍歷 B 資料夾的圖片，進行比對、複製和轉換
    success_count = 0
    for jpg_filename in tqdm(b_jpg_files, desc="處理進度"):
        jpg_basename = os.path.splitext(jpg_filename)[0]

        # 檢查 B 的主檔名是否存在於 A 的主檔名集合中
        if jpg_basename in a_basenames:
            try:
                # 建立完整的來源與目標路徑
                source_path = os.path.join(folder_b, jpg_filename)
                output_path = os.path.join(folder_c, f"{jpg_basename}.png")

                # 開啟 JPG 圖片並另存為 PNG
                with Image.open(source_path) as img:
                    img.save(output_path, 'PNG')
                
                success_count += 1
            except Exception as e:
                tqdm.write(f"處理檔案 '{jpg_filename}' 時發生錯誤: {e}")

    print("\n--- 所有任務已完成！ ---")
    print(f"✅ 總共成功複製並轉換了 {success_count} 個檔案。")

if __name__ == "__main__":
    print("--- 依檔名同步圖片並轉換格式工具 ---")
    
    # --- 使用者設定區塊 ---
    folder_a_path = 'OIL_PROJECT/dataset/dataset_UAV/partial_5280*2970_dataset/labels' # input("請輸入來源資料夾 A 的路徑 (.txt 所在位置): ")
    folder_b_path = 'OIL_PROJECT/dataset/dataset_UAV/partial_5280x2970_with_json/5280x2970' # input("請輸入來源資料夾 B 的路徑 (.jpg 所在位置): ")
    folder_c_path = 'OIL_PROJECT/dataset/dataset_UAV/partial_5280*2970_dataset/images' # input("請輸入輸出資料夾 C 的路徑 (儲存 .png 的位置): ")

    # 檢查路徑有效性
    if not os.path.isdir(folder_a_path):
        print(f"錯誤：來源路徑 A '{folder_a_path}' 不是一個有效的資料夾。")
    elif not os.path.isdir(folder_b_path):
        print(f"錯誤：來源路徑 B '{folder_b_path}' 不是一個有效的資料夾。")
    elif not folder_c_path:
        print("錯誤：輸出資料夾 C 的路徑不能為空。")
    else:
        sync_and_convert_images(folder_a_path, folder_b_path, folder_c_path)