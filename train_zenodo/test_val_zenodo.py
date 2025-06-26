import os
from ultralytics import YOLO

def evaluate_model_on_test(
    model_path: str,
    data_yaml_path: str,
    output_project: str = 'result', # 預設的專案根目錄
    output_name: str = 'val',             # 預設的實驗名稱
    iou_threshold: float = 0.7,
    conf_threshold: float = None,
    exist_ok: bool = False # 是否允許覆蓋同名資料夾 (預設不允許)
):
    """
    載入 YOLO 模型並在指定的測試集上執行評估，並可自訂輸出資料夾。

    Args:
        model_path (str): 訓練好的 .pt 模型檔案路徑。
        data_yaml_path (str): data.yaml 檔案的路徑。
        output_project (str, optional): 儲存評估結果的專案根目錄。
        output_name (str, optional): 儲存評估結果的實驗名稱。
                                    Defaults to 'val' (會自動遞增)。
        iou_threshold (float, optional): NMS IoU 閾值。 Defaults to 0.7.
        conf_threshold (float, optional): 信賴度閾值。 Defaults to None.
        exist_ok (bool, optional): 如果實驗名稱已存在，是否覆蓋。
                                  Defaults to False (會建立 ...2, ...3)。
    """
    print("-" * 60)
    print(f"開始評估模型: {os.path.basename(model_path)}")
    print(f"  - 資料集設定: {data_yaml_path}")
    print(f"  - 輸出位置: {os.path.join(output_project, output_name)}")
    print(f"  - NMS IoU 閾值: {iou_threshold}")
    print(f"  - 信賴度閾值: {conf_threshold if conf_threshold else '預設 (用於 mAP)'}")
    print("-" * 60)

    try:
        # 載入模型
        model = YOLO(model_path)

        # 執行驗證/評估，加入 project 和 name 參數
        metrics = model.val(data=data_yaml_path, 
                            split='test',
                            iou=iou_threshold,
                            conf=conf_threshold,
                            project=output_project, # <--- 設定專案路徑
                            name=output_name,     # <--- 設定實驗名稱
                            exist_ok=exist_ok)    # <--- 設定是否覆蓋

        # --- 輸出結果 (保持不變) ---
        print("\n--- 測試集評估結果 ---")
        print("\n[邊界框 (Box) 指標]")
        print(f"  Precision(B): {metrics.box.mp:.4f}")
        print(f"  Recall(B):    {metrics.box.mr:.4f}")
        print(f"  mAP50(B):     {metrics.box.map50:.4f}")
        print(f"  mAP50-95(B):  {metrics.box.map:.4f}")
        print("\n[分割遮罩 (Mask) 指標]")
        print(f"  Precision(M): {metrics.seg.mp:.4f}")
        print(f"  Recall(M):    {metrics.seg.mr:.4f}")
        print(f"  mAP50(M):     {metrics.seg.map50:.4f}")
        print(f"  mAP50-95(M):  {metrics.seg.map:.4f}")
        print("-" * 60)
        print(f"評估完成: {os.path.basename(model_path)}")
        print("-" * 60 + "\n")

    except Exception as e:
        print(f"\n處理模型 {os.path.basename(model_path)} 時發生錯誤: {e}\n")

# --- 如何呼叫這個函式 ---
if __name__ == '__main__':
    print("開始執行批次模型評估任務...")

    # --- 定義您的 data.yaml 路徑 ---
    DATA_YAML = 'train_zenodo\data.yaml' # <--- 請務必改成您的 YAML 檔案路徑

    # --- 定義您要評估的模型和對應的輸出名稱 ---
    models_to_evaluate = [
        {
            'model_path': 'runs\segment\zenodo\yolo(11n-s-500epoch)_dataset(zenodo_onlyVV_only_oil)\weights\\last.pt', # <--- 改成您的模型 1
            'output_name': 'yolo(11n-s-500epoch)_dataset(zenodo_onlyVV_only_oil)' # <--- 為模型 1 設定名稱
        },
        # {
        #     'model_path':  'runs\segment\yolo11m-seg-zenodo\weights\\best.pt', # <--- 改成您的模型 2
        #     'output_name': 'zenodo_yolo11m' # <--- 為模型 2 設定名稱
        # },
        # {
        #     'model_path': 'runs\segment\yolo11n-seg-zenodo2\weights\\best.pt', # <--- 改成您的模型 3
        #     'output_name': 'zenodo_yolo11n_500epoch', # <--- 可以用不同參數跑同個模型
        # },
    ]

    # --- 定義您想儲存所有評估結果的專案根目錄 ---
    EVAL_PROJECT_FOLDER = 'result' 

    # --- 迴圈執行所有模型的評估 ---
    for config in models_to_evaluate:
        model_file = config['model_path']
        if os.path.exists(model_file):
            evaluate_model_on_test(
                model_path=model_file,
                data_yaml_path=DATA_YAML,
                output_project=EVAL_PROJECT_FOLDER, # <--- 傳入專案路徑
                output_name=config['output_name'], # <--- 傳入實驗名稱
            )
        else:
            print(f"警告：找不到模型檔案 {model_file}，跳過評估。")

    print("所有模型評估任務執行完畢。")
