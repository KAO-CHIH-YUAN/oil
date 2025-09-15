"""
功能說明
讀取設定：您可以在程式碼開頭的「使用者設定區塊」輕鬆設定：

INPUT_DIR：您的主要來源資料夾路徑。

OUTPUT_DIR：您要儲存結果的輸出資料夾路徑。

FILENAMES_TO_COPY：一個 Python 列表，裡面包含了您所有想要複製的檔案主檔名（不需要副檔名）。

維持結構：程式會自動在輸出資料夾中，建立與來源資料夾相同的子資料夾結構，即 images/test 和 labels/test。

精準複製：

程式會掃描來源資料夾中的 images/test 和 labels/test。

只有當一個檔案的主檔名存在於您的 FILENAMES_TO_COPY 列表中時，該檔案（無論是 .jpg, .png 還是 .txt）才會被複製。

例如，如果您的列表包含 'file_A'，那麼 images/test/file_A.jpg 和 labels/test/file_A.txt 都會被複製。

進度報告：程式會顯示進度條，並在結束時回報總共複製了多少個檔案。

"""


import os
import shutil
from tqdm import tqdm

# --- 使用者設定區塊 ---

# 1. 設定主要來源資料夾
INPUT_DIR = r"/home/yuan/OIL_PROJECT/dataset/dataset_SAR_3/S1_PNG_output"

# 2. 設定主要輸出資料夾 (如果不存在，程式會自動建立)
OUTPUT_DIR = r"/home/yuan/OIL_PROJECT/dataset/dataset_SAR_3/S1_PNG_output_confidence_higher_than_0"

# 3. 在這裡填入您所有想要複製的檔案「主檔名」（不需要副檔名）
FILENAMES_TO_COPY = [  # Taiwan index labels有無合併問題。
    "S1A_20230920_Des_3P_8b", # 2 without 40
    "s1_20230321_mosaic_8b", # 7 without 6
    "S1A_20241017_Asc_8b", # 16 without 12 15
    "S1A_20250415_Asc_8b", # 17 
    'S1A_20230905_Asc_4P_8b_UTM', # 18 22
    "S1A_20230731_Asc_3P_8b", # 23
    "S1A_20250217_Des_8b", # 34
    "s1a_20230404_mosaic_8b", # 38 without 57
    "S1A_20230628_Des_3P_8b", # 42 55 without 44
    "S1A_20240306_Des_8b", # 45 
    "s1_20230314_mosaic_8b" , # 60
    # ... 您可以繼續往下加 ...
]

# ----------------------

def selective_copy_files(input_base, output_base, target_basenames):
    """
    根據提供的檔名列表，選擇性地從來源資料夾複製檔案到輸出資料夾，並保持結構。
    """
    print("--- 開始根據檔名列表選擇性複製檔案 ---")
    print(f"來源資料夾: {os.path.abspath(input_base)}")
    print(f"輸出資料夾: {os.path.abspath(output_base)}")
    
    # 將列表轉換為集合 (set) 以大幅提升查找效率
    target_set = set(target_basenames)
    print(f"將根據 {len(target_set)} 個指定的檔名進行匹配。")
    
    # 定義需要處理的子資料夾結構
    sub_dirs_to_process = [
        os.path.join("images", "test"),
        os.path.join("labels", "test")
    ]
    
    total_copied_count = 0

    for sub_dir in sub_dirs_to_process:
        source_path = os.path.join(input_base, sub_dir)
        dest_path = os.path.join(output_base, sub_dir)

        print(f"\n正在處理子資料夾: {source_path}")

        # 檢查來源子資料夾是否存在
        if not os.path.isdir(source_path):
            print(f"  [警告] 來源子資料夾不存在，已跳過。")
            continue

        # 建立對應的輸出子資料夾
        os.makedirs(dest_path, exist_ok=True)

        # 掃描來源資料夾中的所有檔案
        files_to_scan = os.listdir(source_path)
        
        copied_in_this_dir = 0
        for filename in tqdm(files_to_scan, desc=f"  掃描 {sub_dir}"):
            # 取得檔案的主檔名 (不含副檔名)
            basename = os.path.splitext(filename)[0]

            # 如果主檔名在我們的目標清單中，就執行複製
            if basename in target_set:
                try:
                    source_file_path = os.path.join(source_path, filename)
                    dest_file_path = os.path.join(dest_path, filename)
                    
                    # 使用 shutil.copy2 可以同時複製檔案內容和元數據
                    shutil.copy2(source_file_path, dest_file_path)
                    copied_in_this_dir += 1
                except Exception as e:
                    print(f"\n  [錯誤] 複製檔案 '{filename}' 時發生問題: {e}")
        
        print(f"  完成。在此資料夾中複製了 {copied_in_this_dir} 個檔案。")
        total_copied_count += copied_in_this_dir

    print("\n--- 所有任務已完成！ ---")
    print(f"✅ 總共成功複製了 {total_copied_count} 個檔案。")

if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        print(f"錯誤：指定的來源資料夾不存在: '{INPUT_DIR}'")
    else:
        selective_copy_files(INPUT_DIR, OUTPUT_DIR, FILENAMES_TO_COPY)