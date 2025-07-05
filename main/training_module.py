import time, torch, psutil
from ultralytics import YOLO
from pathlib import Path
from utils import create_temp_data_yaml

def train_model(config, results_path):
    """
    執行單次 YOLO 模型訓練任務。

    Args:
        config (dict): 包含當前訓練任務所有參數的設定字典。
        results_path (Path): 本次訓練結果的儲存路徑。

    Returns:
        dict or None: 如果訓練成功，返回包含訓練結果(如時間、模型路徑)的字典；否則返回 None。
    """
    exp_name = config['experiment_name']
    print(f"\n{'='*20} 開始訓練: {exp_name} {'='*20}")
    print(f"結果將儲存於: {results_path}")

    try:
        # --- 步驟 1: 準備資料集設定檔 ---
        dataset_config = config.get('dataset', {})
        if not dataset_config:
            print("  [錯誤] 訓練任務未定義 'dataset'。")
            return None
        
        # 為 dataset 設定提供預設值，以簡化 yaml 設定檔
        dataset_config.setdefault('train', 'images/train')
        dataset_config.setdefault('val', 'images/val')
        dataset_config.setdefault('test', 'images/test')
        dataset_config.setdefault('nc', 1)
        dataset_config.setdefault('names', ['oil'])
        
        # 動態產生 YOLO 所需的 data.yaml 檔案
        temp_yaml_path = create_temp_data_yaml(dataset_config, results_path)

        # --- 步驟 2: 初始化模型並開始訓練 ---
        model = YOLO(config['base_model'])
        start_time = time.time()
        process = psutil.Process()
        initial_ram = process.memory_info().rss
        
        # 呼叫 YOLO 的核心訓練函式
        model.train(
            data=str(temp_yaml_path), # 指定動態產生的 yaml 檔
            epochs=config.get('epochs', 100),
            imgsz=config.get('imgsz', 640),
            batch=config.get('batch_size', 16),
            patience=config.get('patience', 100),
            project=str(results_path.parent), # 專案目錄設定為 results_path 的上一層
            name=results_path.name,           # 實驗名稱設定為 results_path 的資料夾名稱
            exist_ok=True,
            conf=config.get('eval_conf', 0.25), # 訓練期間驗證(validation)所用的信心度
            iou=config.get('eval_iou', 0.6)    # 訓練期間驗證(validation)所用的 IoU 閾值
        )

        # --- 步驟 3: 計算並回傳訓練效能指標 ---
        training_time = time.time() - start_time
        final_ram = process.memory_info().rss
        best_model_path = results_path / 'weights' / 'best.pt'
        
        if not best_model_path.exists():
            print("  [錯誤] 訓練結束，但找不到 best.pt 檔案。")
            return None
            
        training_results = {
            'training_time_minutes': f"{training_time / 60:.2f}",
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'Ram_GB': f"{(final_ram - initial_ram) / (1024 ** 3):.2f}",
            'best_model_path': str(best_model_path)
        }
        print(f"訓練完成！耗時: {training_results['training_time_minutes']} 分鐘")
        return training_results
        
    except Exception as e:
        print(f"訓練過程中發生嚴重錯誤: {e}")
        import traceback; traceback.print_exc()
        return None