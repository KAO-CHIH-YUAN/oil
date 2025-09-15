# ======================================================================================
# ===     preprocess/assemble_final_dataset.py (包含 Test Set 組裝功能的最終版)      ===
# ======================================================================================
import os
import shutil
from pathlib import Path
import random

def assemble_dataset(positive_source_dir, hard_negatives_dir, output_dir, val_split_ratio=0.2, random_seed=42):
    """
    自動化組裝最終的、包含 train/val/test 的完整資料集。
    - train/val: 由正樣本和挖掘出的困難負樣本按比例組合而成。
    - test: 由來源目錄中完整的 test_pos 和 test_neg 合併而成。
    """
    print("--- 開始組裝最終的 train/val/test 完整資料集 ---")
    random.seed(random_seed)

    # --- 步驟 1: 設定與驗證路徑 ---
    positive_source_dir = Path(positive_source_dir)
    hard_negatives_dir = Path(hard_negatives_dir)
    output_dir = Path(output_dir)

    # 驗證所有需要的來源路徑
    pos_train_img_dir = positive_source_dir / "images" / "train_pos"
    test_pos_img_dir = positive_source_dir / "images" / "test_pos"
    test_neg_img_dir = positive_source_dir / "images" / "test_neg"
    if not pos_train_img_dir.is_dir(): raise NotADirectoryError(f"正樣本訓練資料夾不存在: {pos_train_img_dir}")
    if not hard_negatives_dir.is_dir(): raise NotADirectoryError(f"困難負樣本來源資料夾不存在: {hard_negatives_dir}")
    if not test_pos_img_dir.is_dir(): raise NotADirectoryError(f"測試集正樣本資料夾不存在: {test_pos_img_dir}")
    if not test_neg_img_dir.is_dir(): raise NotADirectoryError(f"測試集負樣本資料夾不存在: {test_neg_img_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    # 建立完整的 train/val/test 結構
    out_train_img = output_dir / "images" / "train"; out_train_lbl = output_dir / "labels" / "train"
    out_val_img = output_dir / "images" / "val"; out_val_lbl = output_dir / "labels" / "val"
    out_test_img = output_dir / "images" / "test"; out_test_lbl = output_dir / "labels" / "test"
    out_train_img.mkdir(parents=True); out_train_lbl.mkdir(parents=True)
    out_val_img.mkdir(parents=True); out_val_lbl.mkdir(parents=True)
    out_test_img.mkdir(parents=True); out_test_lbl.mkdir(parents=True)

    # --- 步驟 2: 檔案複製輔助函式 (保持不變) ---
    def copy_files(file_list, src_img_dir, src_lbl_dir, dest_img_dir, dest_lbl_dir):
        for base_name in file_list:
            if (src_img_dir / (base_name + '.png')).exists(): shutil.copy(src_img_dir / (base_name + '.png'), dest_img_dir)
            if (src_lbl_dir / (base_name + '.txt')).exists(): shutil.copy(src_lbl_dir / (base_name + '.txt'), dest_lbl_dir)
            if (src_lbl_dir / (base_name + '.png')).exists(): shutil.copy(src_lbl_dir / (base_name + '.png'), dest_lbl_dir)
        return len(file_list)

    # --- 步驟 3: 組裝 Train 和 Val 集合 ---
    print("\n正在組裝 Train 和 Val 集合...")
    pos_lbl_dir = positive_source_dir / "labels" / "train_pos"
    positive_files = [p.stem for p in pos_train_img_dir.glob("*.png")]; random.shuffle(positive_files)
    pos_split_idx = int(len(positive_files) * (1 - val_split_ratio)); pos_train_files, pos_val_files = positive_files[:pos_split_idx], positive_files[pos_split_idx:]
    
    neg_img_dir = hard_negatives_dir / "images"; neg_lbl_dir = hard_negatives_dir / "labels"
    negative_files = [p.stem for p in neg_img_dir.glob("*.png")]; random.shuffle(negative_files)
    neg_split_idx = int(len(negative_files) * (1 - val_split_ratio)); neg_train_files, neg_val_files = negative_files[:neg_split_idx], negative_files[neg_split_idx:]
    
    train_pos_count = copy_files(pos_train_files, pos_train_img_dir, pos_lbl_dir, out_train_img, out_train_lbl)
    val_pos_count = copy_files(pos_val_files, pos_train_img_dir, pos_lbl_dir, out_val_img, out_val_lbl)
    train_neg_count = copy_files(neg_train_files, neg_img_dir, neg_lbl_dir, out_train_img, out_train_lbl)
    val_neg_count = copy_files(neg_val_files, neg_img_dir, neg_lbl_dir, out_val_img, out_val_lbl)

    # --- ⭐⭐⭐ 步驟 4: 組裝 Test 集合 (全新功能) ⭐⭐⭐ ---
    print("\n正在組裝 Test 集合...")
    # 獲取 test_pos 的所有檔案
    test_pos_lbl_dir = positive_source_dir / "labels" / "test_pos"
    test_pos_files = [p.stem for p in test_pos_img_dir.glob("*.png")]
    
    # 獲取 test_neg 的所有檔案
    test_neg_lbl_dir = positive_source_dir / "labels" / "test_neg"
    test_neg_files = [p.stem for p in test_neg_img_dir.glob("*.png")]

    # 將 test_pos 和 test_neg 的所有檔案複製到最終的 test 資料夾
    test_pos_count = copy_files(test_pos_files, test_pos_img_dir, test_pos_lbl_dir, out_test_img, out_test_lbl)
    test_neg_count = copy_files(test_neg_files, test_neg_img_dir, test_neg_lbl_dir, out_test_img, out_test_lbl)

    # --- 步驟 5: 顯示最終報告 ---
    print(f"\n--- 組裝完成！---")
    print(f"訓練集: {train_pos_count} 正樣本 + {train_neg_count} 困難負樣本 = {train_pos_count + train_neg_count} 總樣本")
    print(f"驗證集: {val_pos_count} 正樣本 + {val_neg_count} 困難負樣本 = {val_pos_count + val_neg_count} 總樣本")
    print(f"測試集: {test_pos_count} 正樣本 + {test_neg_count} 負樣本 = {test_pos_count + test_neg_count} 總樣本")
    print(f"最終的完整資料集已儲存至: {output_dir}")


if __name__ == '__main__':
    
    # --- ⭐⭐⭐ 請在此處設定您的路徑和參數 ⭐⭐⭐ ---
    # 這是您唯一需要修改的地方。
    
    # 1. 指定包含 _pos 分離資料夾的基礎來源目錄 (例如 '.../SAR_2')
    #    這個路徑應該是 patch_method.py 的輸出目錄下的那個 Category 資料夾
    POSITIVE_SOURCE_DIRECTORY = "/home/yuan/OIL_PROJECT/dataset/dataset_optical/IR_Patch/Patched_P640_O128_BG100p_Separated/IR"

    # 2. 指定由 mine_hard_negatives.py 挖掘出的困難負樣本資料夾
    HARD_NEGATIVES_DIRECTORY = "/home/yuan/OIL_PROJECT/dataset/dataset_optical/IR_Patch/HNM_Step4_Negatives_Mined_IR_P640__Threshold0p001_v2"

    # 3. 指定最終要輸出的黃金資料集的名稱與路徑
    OUTPUT_DIRECTORY = "/home/yuan/OIL_PROJECT/dataset/dataset_optical/IR_Patch/HNM_Step5_Final_IR_P640_v2"
    
    # 4. 設定要從最終組合的樣本中，劃分多少比例作為驗證集
    VALIDATION_RATIO = 0.3

    # --- ⭐⭐⭐ 設定結束，以下為程式執行區塊，無需修改 ⭐⭐⭐ ---

    print("開始執行最終資料集組裝腳本...")
    
    assemble_dataset(
        positive_source_dir=POSITIVE_SOURCE_DIRECTORY,
        hard_negatives_dir=HARD_NEGATIVES_DIRECTORY, 
        output_dir=OUTPUT_DIRECTORY, 
        val_split_ratio=VALIDATION_RATIO
    )

    print("\n腳本執行完畢。")