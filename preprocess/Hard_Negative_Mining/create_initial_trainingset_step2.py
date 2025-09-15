# ======================================================================================
# ===       preprocess/create_initial_trainingset.py (全新、完整版)                ===
# ======================================================================================
  # 階段一：資料製備 (一次性，產生所有基礎材料)
  # 目標：執行一次圖塊切割腳本，從您的原始資料中，產生後續所有階段所需的、已分離的「正樣本」與「負樣本」池。
  # 使用工具：preprocess/patch_method.py (包含分離輸出與日誌功能的最終版)

  # 階段二：建立初始訓練集 (全自動)
  # 目標：自動從階段一產生的「正樣本池」和「負樣本池」中取樣，建立一個規模較小、用於快速訓練第一版模型 (Model v1) 的初始訓練集。
  # 使用工具：preprocess/create_initial_trainingset.py (已修正路徑結構的最終版)

  # 階段三：訓練基準模型 (Model v1)
  # 目標：使用階段二建立的初始訓練集，訓練出一個效能尚可的基準模型。這個模型的主要任務不是追求完美，而是要能找出讓它「混淆」的困難樣本。
  # 使用工具：main/main_runner.py

  # 階段四：執行挖掘 (全自動)
  # 目標：使用您剛剛訓練好的 Model v1，從階段一產生的完整「負樣本池 (train_neg)」中，篩選出所有會讓模型誤判的困難負樣本。
  # 使用工具：preprocess/mine_hard_negatives.py (無命令列參數的最終版)

  # 階段五：組裝黃金資料集 (全自動)
  # 目標：自動將階段一的全部「純正樣本 (train_pos)」、階段四挖掘出的**「困難負樣本」，以及階段一的完整 test 集合**，合併成最終的黃金資料集。
  # 使用工具：preprocess/assemble_final_dataset.py

  # 階段六：訓練終極模型 (Model v2)
  # 目標：在階段五組裝好的高品質黃金資料集上，對 Model v1 進行長時間的微調 (fine-tuning)，得到一個誤報率極低且效能強大的終極模型。
import os
import shutil
from pathlib import Path
import random
import argparse

# --- ⭐ 智慧型預設路徑設定 ⭐ ---
# 假設 patch_method.py 的輸出位於此處
DEFAULT_SOURCE_DIR = Path("/home/yuan/OIL_PROJECT/dataset/dataset_optical/IR_Patch/Patched_P640_O128_BG100p_Separated/IR")
# 預設的 Model v1 訓練集輸出路徑
DEFAULT_OUTPUT_DIR = Path("/home/yuan/OIL_PROJECT/dataset/dataset_optical/IR_Patch/HNM_Step1_TrainingSet-P640") # dir 命名


def create_initial_set(source_dir, output_dir, neg_sample_ratio=0.2, val_split_ratio=0.2, random_seed=42):
    """
    自動化建立用於訓練第一版模型的初始訓練集與驗證集。
    此腳本會合併全部的正樣本和指定比例的負樣本。

    Args:
        source_dir (str): 分離的正負樣本的基礎來源資料夾 (例如 '.../SAR_2')。
        output_dir (str): 初始訓練集的輸出路徑。
        neg_sample_ratio (float): 要從負樣本中隨機抽樣的比例。
        val_split_ratio (float): 從組合後的樣本中劃分多少比例作為驗證集。
        random_seed (int): 隨機數種子，確保每次取樣結果一致。
    """
    print("--- 開始建立 Model v1 的初始訓練集 ---")
    random.seed(random_seed)

    # --- 步驟 1: 設定與驗證路徑 ---
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.is_dir():
        raise NotADirectoryError(f"來源基礎資料夾不存在: {source_dir}")

    # ⭐⭐⭐ 根據正確的結構來定義路徑 ⭐⭐⭐
    pos_img_dir = source_dir / "images" / "train_pos"
    pos_lbl_dir = source_dir / "labels" / "train_pos"
    neg_img_dir = source_dir / "images" / "train_neg"
    neg_lbl_dir = source_dir / "labels" / "train_neg"

    if not pos_img_dir.is_dir(): raise NotADirectoryError(f"正樣本影像資料夾不存在: {pos_img_dir}")
    if not neg_img_dir.is_dir(): raise NotADirectoryError(f"負樣本影像資料夾不存在: {neg_img_dir}")

    if output_dir.exists():
        print(f"[警告] 偵測到現有輸出資料夾，將被清空: {output_dir}")
        shutil.rmtree(output_dir)

    out_train_img = output_dir / "images" / "train"
    out_train_lbl = output_dir / "labels" / "train"
    out_val_img = output_dir / "images" / "val"
    out_val_lbl = output_dir / "labels" / "val"
    out_train_img.mkdir(parents=True); out_train_lbl.mkdir(parents=True)
    out_val_img.mkdir(parents=True); out_val_lbl.mkdir(parents=True)

    # --- 步驟 2: 獲取並取樣檔案列表 ---
    positive_files = [p.stem for p in pos_img_dir.glob("*.png")]
    all_negative_files = [p.stem for p in neg_img_dir.glob("*.png")]
    num_to_sample = int(len(all_negative_files) * neg_sample_ratio)
    sampled_negative_files = random.sample(all_negative_files, num_to_sample)
    
    print(f"找到 {len(positive_files)} 個正樣本。")
    print(f"從 {len(all_negative_files)} 個負樣本中，隨機取樣 {len(sampled_negative_files)} 個。")

    combined_files = positive_files + sampled_negative_files
    random.shuffle(combined_files)

    # --- 步驟 3: 拆分訓練集與驗證集 ---
    split_idx = int(len(combined_files) * (1 - val_split_ratio))
    train_files = combined_files[:split_idx]
    val_files = combined_files[split_idx:]

    # --- 步驟 4: 定義輔助函式並執行複製 ---
    def copy_files(file_list, dest_img_dir, dest_lbl_dir):
        for base_name in file_list:
            if base_name in positive_files:
                src_img_dir, src_lbl_dir = pos_img_dir, pos_lbl_dir
            else:
                src_img_dir, src_lbl_dir = neg_img_dir, neg_lbl_dir

            shutil.copy(src_img_dir / (base_name + '.png'), dest_img_dir / (base_name + '.png'))
            if (src_lbl_dir / (base_name + '.txt')).exists():
                shutil.copy(src_lbl_dir / (base_name + '.txt'), dest_lbl_dir / (base_name + '.txt'))
            if (src_lbl_dir / (base_name + '.png')).exists():
                shutil.copy(src_lbl_dir / (base_name + '.png'), dest_lbl_dir / (base_name + '.png'))

    print("\n正在複製檔案並建立訓練/驗證集...")
    copy_files(train_files, out_train_img, out_train_lbl)
    copy_files(val_files, out_val_img, out_val_lbl)

    print("\n--- 初始訓練集建立完成！ ---")
    print(f"訓練集總數: {len(train_files)} | 驗證集總數: {len(val_files)}")
    print(f"最終資料集已儲存至: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Initial Training Set for HNM Workflow")
    # ⭐⭐⭐ 新的、更合理的參數 ⭐⭐⭐
    parser.add_argument('--source-dir', type=str, default=str(DEFAULT_SOURCE_DIR), help='分離的正負樣本的基礎來源資料夾 (例如 .../SAR_2)')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR), help='初始訓練集的輸出路徑')
    parser.add_argument('--neg-ratio', type=float, default=0.2, help='負樣本抽樣比例 (預設: 0.2)')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='驗證集拆分比例 (預設: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='隨機數種子 (預設: 42)')
    
    args = parser.parse_args()
    
    create_initial_set(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        neg_sample_ratio=args.neg_ratio,
        val_split_ratio=args.val_ratio,
        random_seed=args.seed
    )