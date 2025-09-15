# ======================================================================================
# ===     preprocess/mine_hard_negatives.py (無命令列參數、內部設定、完整最終版)     ===
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
from ultralytics import YOLO
from tqdm import tqdm
# argparse 已被完全移除

def mine_hard_negatives(model_path, source_images_dir, source_labels_dir, output_dir, conf_threshold):
    """
    使用一個預訓練的模型，在大量的背景樣本中挖掘出「困難負樣本」。
    困難負樣本指的是那些被模型錯誤地偵測出目標的背景影像。

    Args:
        model_path (str): 初步訓練好的 YOLO 模型權重路徑 (best.pt)。
        source_images_dir (str): 包含大量背景圖塊（負樣本）的影像來源資料夾。
        source_labels_dir (str): 對應的標籤資料夾 (理論上裡面的 .txt 都應該是空的)。
        output_dir (str): 用來存放挖掘出的困難負樣本的新資料夾路徑。
        conf_threshold (float): 偵測時使用的信心度閾值。可以設低一點以挖掘更多潛在樣本。
    """
    print("--- 開始進行困難負樣本挖掘 ---")
    
    # --- 步驟 1: 參數驗證與路徑設定 ---
    model_path = Path(model_path)
    source_images_dir = Path(source_images_dir)
    source_labels_dir = Path(source_labels_dir)
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")

    if not source_images_dir.is_dir():
        raise NotADirectoryError(f"來源影像資料夾不存在: {source_images_dir}")
        
    if not source_labels_dir.is_dir():
        raise NotADirectoryError(f"來源標籤資料夾不存在: {source_labels_dir}")

    # 建立輸出的影像和標籤資料夾
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    
    # 如果輸出資料夾已存在，先清空，確保每次挖掘都是全新的
    if output_dir.exists():
        print(f"[警告] 偵測到現有輸出資料夾，將會被清空: {output_dir}")
        shutil.rmtree(output_dir)
        
    output_images_dir.mkdir(parents=True)
    output_labels_dir.mkdir(parents=True)

    print(f"模型路徑: {model_path}")
    print(f"來源影像: {source_images_dir}")
    print(f"來源標籤: {source_labels_dir}")
    print(f"輸出路徑: {output_dir}")
    print(f"信心度閾值: {conf_threshold}")
    print("-" * 30)

    # --- 步驟 2: 載入模型並獲取所有背景影像 ---
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[錯誤] 載入 YOLO 模型失敗: {e}")
        return

    all_background_images = list(source_images_dir.glob('*.jpg')) + list(source_images_dir.glob('*.png'))
    
    if not all_background_images:
        print("[資訊] 在來源資料夾中沒有找到任何影像檔案。")
        return
        
    hard_negatives_count = 0

    # --- 步驟 3: 遍歷所有背景影像，執行預測並篩選 ---
    print(f"正在對 {len(all_background_images)} 個背景影像進行分析...")
    for img_path in tqdm(all_background_images, desc="挖掘困難負樣本"):
        # 為了安全起見，再次確認這是一個負樣本 (其 .txt 標籤檔應該是空的)
        label_path = source_labels_dir / (img_path.stem + '.txt')
        if label_path.exists() and label_path.stat().st_size > 0:
            # 如果標籤檔不是空的，代表這不是一個純粹的負樣本，跳過它
            continue

        # 使用模型進行預測
        results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)
        
        # 檢查模型是否在此背景影像上做出了任何預測
        if results and results[0].boxes and len(results[0].boxes.data) > 0:
            # 如果模型做出了預測，代表這是一個「困難負樣本」
            hard_negatives_count += 1
            
            # 將這個困難的影像複製到輸出資料夾
            shutil.copy(img_path, output_images_dir / img_path.name)
            
            # 因為這是負樣本，我們需要為它創建一個空的 .txt 和 .png 標籤，以保持結構完整性
            # 創建空的 .txt
            (output_labels_dir / (img_path.stem + '.txt')).touch()
            
            # 複製對應的、全黑的 .png 遮罩檔
            png_mask_path = source_labels_dir / (img_path.stem + '.png')
            if png_mask_path.exists():
                shutil.copy(png_mask_path, output_labels_dir / png_mask_path.name)

    # --- 步驟 4: 顯示結果 ---
    print("-" * 30)
    print("困難負樣本挖掘完成！")
    print(f"總共分析了 {len(all_background_images)} 個背景樣本。")
    print(f"成功挖掘出 {hard_negatives_count} 個困難負樣本。")
    print(f"挖掘出的樣本已儲存至: {output_dir}")


# ======================================================================================
# ===                            腳本執行區塊                                        ===
# ======================================================================================
if __name__ == '__main__':
    
    # --- ⭐⭐⭐ 請在此處設定您的路徑和參數 ⭐⭐⭐ ---
    # 這是您唯一需要修改的地方。
    
    # 1. 指定您初步訓練好的 Model v1 權重檔案路徑
    MODEL_V1_WEIGHTS_PATH = "/home/yuan/OIL_PROJECT/result_IR/20250820-043626_HNM_Step3_IR_P640/weights/best.pt"

    # 2. 指定包含【全部】背景負樣本的來源資料夾
    #    這是從 patch_method.py 產生的 _neg 資料夾
    SOURCE_IMAGES_DIRECTORY = "/home/yuan/OIL_PROJECT/dataset/dataset_optical/IR_Patch/Patched_P640_O128_BG100p_Separated/IR/images/train_neg"
    SOURCE_LABELS_DIRECTORY = "/home/yuan/OIL_PROJECT/dataset/dataset_optical/IR_Patch/Patched_P640_O128_BG100p_Separated/IR/labels/train_neg"

    # 3. 指定要將挖掘出的困難負樣本儲存到哪裡
    OUTPUT_DIRECTORY = "/home/yuan/OIL_PROJECT/dataset/dataset_optical/IR_Patch/HNM_Step4_Negatives_Mined_IR_P640__Threshold0p001_v2"

    # 4. 設定偵測時使用的信心度閾值 (建議設低一點，以挖掘更多潛在樣本)
    CONFIDENCE_THRESHOLD = 0.001
    
    # --- ⭐⭐⭐ 設定結束，以下為程式執行區塊，無需修改 ⭐⭐⭐ ---
    
    print("開始執行困難負樣本挖掘腳本...")
    
    # 呼叫主函式，並傳入上面設定好的參數
    mine_hard_negatives(
        model_path=MODEL_V1_WEIGHTS_PATH,
        source_images_dir=SOURCE_IMAGES_DIRECTORY,
        source_labels_dir=SOURCE_LABELS_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        conf_threshold=CONFIDENCE_THRESHOLD
    )

    print("\n腳本執行完畢。")