import os

def validate_yolo_segmentation_labels(labels_dir):
    """
    掃描並驗證指定目錄下的所有 YOLO Segmentation TXT 檔案。
    """
    print(f"開始掃描並驗證標註檔案於: {labels_dir}")
    print("-" * 30)

    error_files = {}
    total_files_checked = 0

    if not os.path.isdir(labels_dir):
        print(f"[錯誤] 目錄不存在: {labels_dir}")
        return

    for dirpath, _, filenames in os.walk(labels_dir):
        for filename in filenames:
            if filename.lower().endswith('.txt'):
                total_files_checked += 1
                file_path = os.path.join(dirpath, filename)
                errors_in_file = []

                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:  # 跳過空行 (這是合法的)
                            continue

                        parts = line.split()
                        
                        # 1. 檢查 Class ID
                        if parts[0] != '0':
                            errors_in_file.append(f"  - 第 {line_num} 行: Class ID 不是 '0' (實際為 '{parts[0]}')")
                        
                        # 2. 檢查座標點數量是否為偶數
                        num_coords = len(parts) - 1
                        if num_coords % 2 != 0:
                            errors_in_file.append(f"  - 第 {line_num} 行: 座標點總數為奇數 ({num_coords} 個)，無法配對。")
                        
                        # 3. 檢查點數是否足夠構成多邊形
                        if num_coords < 6:
                             errors_in_file.append(f"  - 第 {line_num} 行: 座標點少於3對 (共 {num_coords} 個)，無法構成多邊形。")

                if errors_in_file:
                    error_files[file_path] = errors_in_file
    
    print(f"總共檢查了 {total_files_checked} 個檔案。")
    print("-" * 30)

    if not error_files:
        print("✅ 恭喜！所有標註檔案格式均正確！")
    else:
        print(f"❌ 發現 {len(error_files)} 個檔案存在格式問題：")
        for file_path, errors in error_files.items():
            print(f"\n檔案: {file_path}")
            for error in errors:
                print(error)

# --- 主程式設定 ---
if __name__ == "__main__":
    # 請將此路徑設定為您 YOLO 資料集配置檔 (data.yaml) 中
    # train 和 val 標籤路徑的 "共同上層目錄"。
    # 根據您的日誌，它應該是這個路徑。
    LABELS_BASE_DIRECTORY = r"/home/yuan/OIL_PROJECT/dataset/dataset_zenodo/zenodo/labels"
    
    validate_yolo_segmentation_labels(LABELS_BASE_DIRECTORY)