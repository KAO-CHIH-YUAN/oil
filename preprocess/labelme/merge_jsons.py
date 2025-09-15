import os
import json
from collections import defaultdict
from tqdm import tqdm
import argparse

def merge_json_files(input_dir, output_dir):
    """
    掃描輸入資料夾，將同名的 JSON 檔案合併，並儲存到輸出資料夾。
    例如：'sss.1.json' 和 'sss.2.json' 會被合併成 'sss.json'。
    因為SkyTruth 有可能會是相同的時間段的油汙，但是不同位置標註，所以需要合併。
    """
    
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)
    
    files_to_process = defaultdict(list)

    # 1. 根據主要檔名將所有 JSON 檔案分組
    print("正在掃描並分組檔案...")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.json'):
            # 以最後一個 '.' 為分隔，取得主要檔名
            parts = filename.rsplit('.', 2)
            if len(parts) >= 2: # 確保檔名至少包含一個 '.'
                base_name = parts[0]
                full_path = os.path.join(input_dir, filename)
                files_to_process[base_name].append(full_path)

    if not files_to_process:
        print(f"在 '{input_dir}' 中找不到任何符合 'name.number.json' 格式的檔案。")
        return

    print(f"找到 {len(files_to_process)} 組需要合併的檔案。開始處理...")

    # 2. 遍歷每個群組，進行合併
    for base_name, file_paths in tqdm(files_to_process.items(), desc="合併進度"):
        
        # 如果一個群組只有一個檔案，也可以選擇直接複製或跳過
        if len(file_paths) == 1:
            print(f"群組 '{base_name}' 只有一個檔案，將直接複製。")
            # 這裡我們選擇直接複製
            import shutil
            shutil.copy(file_paths[0], os.path.join(output_dir, f"{base_name}.json"))
            continue

        merged_data = None
        
        # 為了確保合併順序 (e.g., .1.json, .2.json, ... .10.json)，進行排序
        file_paths.sort()

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if merged_data is None:
                    # 如果是第一個檔案，直接賦值
                    merged_data = data
                else:
                    # 合併邏輯：
                    # 如果是列表，就串接
                    if isinstance(merged_data, list) and isinstance(data, list):
                        merged_data.extend(data)
                    # 如果是字典，就更新 (後者覆蓋前者)
                    elif isinstance(merged_data, dict) and isinstance(data, dict):
                        merged_data.update(data)
                    else:
                        print(f"警告：群組 '{base_name}' 中的檔案類型不一致 (一個是列表，一個是字典)，無法合併。將跳過此檔案 {os.path.basename(file_path)}。")

            except json.JSONDecodeError:
                print(f"錯誤：無法解析檔案 {os.path.basename(file_path)}，可能不是有效的 JSON。跳過。")
            except Exception as e:
                print(f"處理檔案 {os.path.basename(file_path)} 時發生未知錯誤: {e}")

        # 3. 將合併後的數據寫入新檔案
        if merged_data is not None:
            output_path = os.path.join(output_dir, f"{base_name}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"\n合併完成！輸出檔案已儲存至 '{output_dir}' 資料夾中。")

def main():
    parser = argparse.ArgumentParser(description="合併具有相同主檔名的多個 JSON 檔案。")
    
    parser.add_argument('--input_dir', type=str, default=r'c:\Users\Lamulam\Downloads\json',
                        help='包含要合併的 JSON 檔案的來源資料夾。(預設: ./source_jsons)')
    parser.add_argument('--output_dir', type=str, default=r'c:\Users\Lamulam\Downloads\output_json',
                        help='儲存合併後 JSON 檔案的輸出資料夾。(預設: ./merged_jsons)')

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"錯誤：輸入資料夾 '{args.input_dir}' 不存在。請建立它並放入您的 JSON 檔案。")
        return
        
    merge_json_files(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()