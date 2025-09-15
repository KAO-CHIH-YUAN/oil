import os
import json
from PIL import Image, ImageDraw
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def find_image_file(directory, base_name):
    """在目錄中尋找對應的圖片檔案（支援多種格式）"""
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIF', '.TIFF']:
        image_path = os.path.join(directory, base_name + ext)
        if os.path.exists(image_path):
            return image_path
    return None

def convert_labelme_to_yolo_seg(input_dir, output_dir):
    """
    將 LabelMe 的 JSON 檔案轉換為 YOLO Segmentation 所需的 .png 遮罩和 .txt 格式。
    """
    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- 開始轉換 LabelMe JSON 到 YOLO Segmentation 格式 ---")
    print(f"輸出資料將儲存於: {os.path.abspath(output_dir)}")

    all_files = os.listdir(input_dir)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    total_image_count = sum(1 for f in all_files if f.lower().endswith(tuple(image_extensions)))
    json_files = [f for f in all_files if f.endswith('.json')]
    total_json_count = len(json_files)
    
    success_count = 0
    failure_count = 0
    
    if not json_files:
        print(f"錯誤：在資料夾 '{input_dir}' 中找不到任何 .json 檔案。")
        return

    print(f"\n--- 檔案掃描結果 ---")
    print(f"在來源資料夾中找到 {total_image_count} 個圖片檔案。")
    print(f"在來源資料夾中找到 {total_json_count} 個 JSON 檔案，準備開始處理...")
    print("-" * 25)

    for json_filename in tqdm(json_files, desc="處理進度"):
        json_path = os.path.join(input_dir, json_filename)
        base_name = os.path.splitext(json_filename)[0]

        # --- ★★★ 新增修改點：處理檔名結尾的 "__1" ★★★ ---
        # 如果檔名以 "__1" 結尾，則將其去除，以匹配原始圖片檔名
        if base_name.endswith('__1'):
            base_name = base_name[:-3]  # 去除最後三個字元
        # --- ★★★ 修改結束 ★★★ ---

        # 尋找對應的圖片檔案
        image_path = find_image_file(input_dir, base_name)
        if image_path is None:
            tqdm.write(f"警告：找不到 '{json_filename}' 對應的圖片檔案 (嘗試匹配 '{base_name}'), 已跳過。")
            failure_count += 1
            continue

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            mask = Image.new('L', (img_width, img_height), 0)
            draw = ImageDraw.Draw(mask)
            yolo_txt_lines = []

            for shape in data['shapes']:
                if shape['label'] == '0' or shape['label'] == 'oil':
                    polygon = shape['points']
                    
                    polygon_tuples = [tuple(p) for p in polygon]
                    draw.polygon(polygon_tuples, fill=255)

                    normalized_points = []
                    for x, y in polygon:
                        norm_x = x / img_width
                        norm_y = y / img_height
                        normalized_points.append(f"{norm_x:.6f} {norm_y:.6f}")
                    
                    yolo_line = "0 " + " ".join(normalized_points)
                    yolo_txt_lines.append(yolo_line)

            mask_output_path = os.path.join(output_dir, base_name + '.png')
            mask.save(mask_output_path)

            if yolo_txt_lines:
                txt_output_path = os.path.join(output_dir, base_name + '.txt')
                with open(txt_output_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_txt_lines))
            
            success_count += 1

        except Exception as e:
            tqdm.write(f"錯誤：處理檔案 '{json_filename}' 時發生問題: {e}")
            failure_count += 1

    print("\n--- 所有檔案處理完成！ ---")
    print("\n--- 處理結果統計 ---")
    print(f"來源圖片總數: {total_image_count}")
    print(f"來源 JSON 總數: {total_json_count}")
    print("----------------------")
    print(f"✅ 成功轉換數量: {success_count}")
    print(f"❌ 失敗或跳過數量: {failure_count}")
    print("----------------------")


if __name__ == "__main__":
    # --- 使用者設定區塊 ---
    input_directory = r'/home/yuan/OIL_PROJECT/dataset/dataset_UAV/partial_5280x2970_with_json/5280x2970' # input("請輸入包含圖片和 LabelMe (.json) 檔案的來源資料夾路徑: ")
    output_directory = r'/home/yuan/OIL_PROJECT/dataset/dataset_UAV/partial_5280x2970_mask' # input("請輸入儲存 YOLO 格式標籤的輸出資料夾路徑: ")

    if not os.path.isdir(input_directory):
        print(f"錯誤：來源路徑 '{input_directory}' 不是一個有效的資料夾。")
    elif not output_directory:
        print("錯誤：輸出資料夾路徑不能為空。")
    else:
        convert_labelme_to_yolo_seg(input_directory, output_directory)