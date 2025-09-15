import os
import shutil
import random
from tqdm import tqdm

def copy_files(file_list, split_name, image_src_folder, label_src_folder, output_folder):
    """
    【輔助函式】複製指定列表中的檔案到對應的 train/val/test 資料夾。
    此函式經過修改，可以接收指定的來源資料夾。

    Args:
        file_list (list): 待複製的檔案名稱列表 (不含副檔名)。
        split_name (str): 資料集類型 ('train', 'val', 'test')。
        image_src_folder (str): 該批次檔案的影像來源資料夾。
        label_src_folder (str): 該批次檔案的標籤來源資料夾。
        output_folder (str): 總輸出目標資料夾。
    """
    print(f"\n正在複製 {split_name} 資料...")
    for basename in tqdm(file_list, desc=f"複製 {split_name} 檔案"):
        # 找到原始影像的完整檔名 (處理不同副檔名的情況)
        img_ext = None
        for ext in ['.png', '.jpg', '.jpeg']:
            if os.path.exists(os.path.join(image_src_folder, basename + ext)):
                img_ext = ext
                break
        
        # 如果找不到對應的影像檔，就跳過這個檔案
        if img_ext is None:
            print(f"警告: 在 {image_src_folder} 中找不到檔案 {basename} 的影像，已跳過。")
            continue

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


def split_dataset_new(train_val_image_src, train_val_label_src,
                      test_image_src, test_label_src,
                      output_folder, train_ratio=0.8):
    """
    將兩組不同的資料集處理後放入 YOLO 資料夾結構。
    - 第一組資料集將被拆分為訓練集和驗證集。
    - 第二組資料集將被直接當作測試集。

    Args:
        train_val_image_src (str): 【第一組】包含訓練/驗證影像的來源資料夾。
        train_val_label_src (str): 【第一組】包含訓練/驗證標籤的來源資料夾。
        test_image_src (str): 【第二組】包含測試影像的來源資料夾。
        test_label_src (str): 【第二組】包含測試標籤的來源資料夾。
        output_folder (str): 輸出結果的目標資料夾路徑。
        train_ratio (float): 訓練集在第一組資料中的比例。驗證集比例會自動計算。
    """
    val_ratio = 1 - train_ratio
    print(f"第一組資料集拆分比例: 訓練集={train_ratio:.2%}, 驗證集={val_ratio:.2%}")
    print("第二組資料集將全部作為測試集。")

    # --- 1. 建立 YOLO 所需的資料夾結構 ---
    print("\n正在建立目標資料夾結構...")
    subfolders = ['train', 'val', 'test']
    for sub in subfolders:
        os.makedirs(os.path.join(output_folder, 'images', sub), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'labels', sub), exist_ok=True)
    print("資料夾結構建立完成。")

    # --- 2. 處理第一組資料 (拆分為 Train 和 Val) ---
    print("\n--- 正在處理第一組資料集 (訓練集/驗證集) ---")
    image_files_1 = [os.path.splitext(f)[0] for f in os.listdir(train_val_image_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files_1:
        print("警告: 第一組影像來源資料夾中沒有找到任何影像檔案。")
    else:
        random.shuffle(image_files_1)
        total_files_1 = len(image_files_1)
        print(f"第一組共找到 {total_files_1} 個影像檔案。")

        # 計算分割點並分割檔案列表
        train_split_idx = int(total_files_1 * train_ratio)
        train_files = image_files_1[:train_split_idx]
        val_files = image_files_1[train_split_idx:]
        
        print(f"分配數量: 訓練集={len(train_files)}, 驗證集={len(val_files)}")

        # 複製訓練集和驗證集檔案
        copy_files(train_files, 'train', train_val_image_src, train_val_label_src, output_folder)
        copy_files(val_files, 'val', train_val_image_src, train_val_label_src, output_folder)

    # --- 3. 處理第二組資料 (全部作為 Test) ---
    print("\n--- 正在處理第二組資料集 (測試集) ---")
    test_files = [os.path.splitext(f)[0] for f in os.listdir(test_image_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not test_files:
        print("警告: 第二組影像來源資料夾中沒有找到任何影像檔案。")
    else:
        total_files_2 = len(test_files)
        print(f"第二組共找到 {total_files_2} 個影像檔案，將全部作為測試集。")
        print(f"分配數量: 測試集={len(test_files)}")
        
        # 複製測試集檔案
        copy_files(test_files, 'test', test_image_src, test_label_src, output_folder)

    print("\n資料集拆分與複製完成！")

if __name__ == "__main__":
    print("--- YOLO 資料集拆分工具 (雙來源版本) ---")
    
    # --- 請在此處設定您的路徑 ---

    # 第一組路徑 (將被拆分為 80% 訓練集, 20% 驗證集)
    train_val_image_folder = r"/home/yuan/OIL_PROJECT/dataset/dataset_zenodo/processed_yuan/images/Train_Val_Oil_Spill_images/Oil"  # <--- 請修改為您的第一組影像路徑
    train_val_label_folder = r"/home/yuan/OIL_PROJECT/dataset/dataset_zenodo/processed_yuan/labels/Train_Val_Oil_Spill_mask/Oil"  # <--- 請修改為您的第一組標籤路徑

    # 第二組路徑 (將全部作為測試集)
    test_image_folder = r"/home/yuan/OIL_PROJECT/dataset/dataset_zenodo/processed_yuan_test_dataset/images/Test_images_and_ground_truth/Oil"      # <--- 請修改為您的第二組影像路徑
    test_label_folder = r"/home/yuan/OIL_PROJECT/dataset/dataset_zenodo/processed_yuan_test_dataset/labels/Test_images_and_ground_truth/Oil"      # <--- 請修改為您的第二組標籤路徑

    # 輸出資料夾路徑
    output_folder_path = r"/home/yuan/OIL_PROJECT/dataset/dataset_zenodo/zenodo"     # <--- 請修改為您的輸出路徑

    # --- 路徑檢查與執行 ---
    paths_to_check = {
        "第一組影像來源": train_val_image_folder,
        "第一組標籤來源": train_val_label_folder,
        "第二組影像來源": test_image_folder,
        "第二組標籤來源": test_label_folder
    }

    all_paths_valid = True
    for name, path in paths_to_check.items():
        if not os.path.isdir(path):
            print(f"錯誤: {name}路徑 '{path}' 不是一個有效的資料夾。")
            all_paths_valid = False
    
    if all_paths_valid:
        try:
            # 呼叫新的函式，設定訓練集比例為 80%
            split_dataset_new(
                train_val_image_folder,
                train_val_label_folder,
                test_image_folder,
                test_label_folder,
                output_folder_path,
                train_ratio=0.8
            )
        except Exception as e:
            print(f"處理過程中發生未預期的錯誤: {e}")