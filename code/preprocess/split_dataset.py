import os
import shutil
import random
from tqdm import tqdm

def split_dataset(image_src_folder, label_src_folder, output_folder, train_ratio=0.64, val_ratio=0.16):
    """
    將影像和標籤資料集拆分為訓練、驗證和測試集，並建立 YOLO 資料夾結構。

    Args:
        image_src_folder (str): 包含原始影像的來源資料夾路徑。
        label_src_folder (str): 包含 .txt 和 .png 標籤的來源資料夾路徑。
        output_folder (str): 輸出結果的目標資料夾路徑。
        train_ratio (float): 訓練集的比例。
        val_ratio (float): 驗證集的比例。
    """
    # 測試集的比例會自動計算
    test_ratio = 1 - train_ratio - val_ratio
    print(f"資料集拆分比例: 訓練集={train_ratio:.2%}, 驗證集={val_ratio:.2%}, 測試集={test_ratio:.2%}")

    # --- 1. 建立 YOLO 所需的資料夾結構 ---
    print("正在建立目標資料夾結構...")
    subfolders = ['train', 'val', 'test']
    for sub in subfolders:
        os.makedirs(os.path.join(output_folder, 'images', sub), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'labels', sub), exist_ok=True)
    print("資料夾結構建立完成。")

    # --- 2. 獲取所有影像檔案並隨機打亂 ---
    # 獲取所有影像檔名 (不含副檔名)
    image_files = [os.path.splitext(f)[0] for f in os.listdir(image_src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("錯誤: 影像來源資料夾中沒有找到任何影像檔案。")
        return
        
    random.shuffle(image_files)
    total_files = len(image_files)
    print(f"共找到 {total_files} 個影像檔案。")

    # --- 3. 計算各資料集的分割點 ---
    train_split_idx = int(total_files * train_ratio)
    val_split_idx = int(total_files * (train_ratio + val_ratio))

    # 分割檔案列表
    train_files = image_files[:train_split_idx]
    val_files = image_files[train_split_idx:val_split_idx]
    test_files = image_files[val_split_idx:]
    
    print(f"分配數量: 訓練集={len(train_files)}, 驗證集={len(val_files)}, 測試集={len(test_files)}")

    # --- 4. 定義複製檔案的輔助函式 ---
    def copy_files(file_list, split_name):
        """
        複製指定列表中的檔案到對應的 train/val/test 資料夾。
        
        Args:
            file_list (list): 待複製的檔案名稱列表 (不含副檔名)。
            split_name (str): 資料集類型 ('train', 'val', 'test')。
        """
        print(f"\n正在複製 {split_name} 資料...")
        for basename in tqdm(file_list, desc=f"複製 {split_name} 檔案"):
            # 找到原始影像的完整檔名 (處理不同副檔名的情況)
            img_ext = '.png' # 預設副檔名
            for ext in ['.jpg', '.jpeg', '.png']:
                if os.path.exists(os.path.join(image_src_folder, basename + ext)):
                    img_ext = ext
                    break
            
            # 定義來源路徑
            src_image_path = os.path.join(image_src_folder, basename + img_ext)
            src_label_txt_path = os.path.join(label_src_folder, basename + '.txt')
            src_label_png_path = os.path.join(label_src_folder, basename + '.png')

            # 定義目標路徑
            dest_image_path = os.path.join(output_folder, 'images', split_name, basename + img_ext)
            dest_label_txt_path = os.path.join(output_folder, 'labels', split_name, basename + '.txt')
            dest_label_png_path = os.path.join(output_folder, 'labels', split_name, basename + '.png')

            # 執行複製操作，並檢查來源檔案是否存在
            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dest_image_path)
            if os.path.exists(src_label_txt_path):
                shutil.copy2(src_label_txt_path, dest_label_txt_path)
            if os.path.exists(src_label_png_path):
                shutil.copy2(src_label_png_path, dest_label_png_path)

    # --- 5. 執行複製 ---
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print("\n資料集拆分與複製完成！")

if __name__ == "__main__":
    # --- 主程式執行區塊 ---
    print("--- YOLO 資料集拆分工具 ---")
    
    # 提示使用者輸入路徑
    image_folder_path = r"/home/yuan/OIL_PROJECT/dataset/dataset_zenodo/processed_yuan/images/Train_Val_Oil_Spill_images/Oil" # input("請輸入【影像來源資料夾】的路徑: ")
    label_folder_path = r"/home/yuan/OIL_PROJECT/dataset/dataset_zenodo/processed_yuan/labels/Train_Val_Oil_Spill_mask/Mask_oil" # input("請輸入【標籤來源資料夾】的路徑: ")
    output_folder_path = r"/home/yuan/OIL_PROJECT/dataset/dataset_zenodo/zenodo" #　input("請輸入【輸出資料夾】的路徑: ")

    # 檢查路徑有效性
    if not os.path.isdir(image_folder_path):
        print(f"錯誤: 影像來源路徑 '{image_folder_path}' 不是一個有效的資料夾。")
    elif not os.path.isdir(label_folder_path):
        print(f"錯誤: 標籤來源路徑 '{label_folder_path}' 不是一個有效的資料夾。")
    else:
        try:
            split_dataset(image_folder_path, label_folder_path, output_folder_path)
        except Exception as e:
            print(f"處理過程中發生未預期的錯誤: {e}")