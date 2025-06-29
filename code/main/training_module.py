import time, torch, psutil
from ultralytics import YOLO
from pathlib import Path
from utils import create_temp_data_yaml # 引入函式

def train_model(config, results_path): # 移除 data_yaml_path 參數
    exp_name = config['experiment_name']
    print(f"\n{'='*20} 開始訓練: {exp_name} {'='*20}")
    print(f"結果將儲存於: {results_path}")

    try:
        # 在訓練開始前建立暫存 data.yaml
        dataset_config = config.get('dataset', {})
        if not dataset_config:
            print("  [錯誤] 訓練任務未定義 'dataset'。")
            return None
        temp_yaml_path = create_temp_data_yaml(dataset_config, results_path)

        model = YOLO(config['base_model'])
        start_time = time.time(); process = psutil.Process(); initial_ram = process.memory_info().rss
        
        # ⭐ 核心修改：使用新的 `eval_iou` 和 `eval_conf`
        # `model.train` 內部會執行驗證，這些參數會被用上
        model.train(
            data=str(temp_yaml_path),
            epochs=config.get('epochs', 100),
            imgsz=config.get('imgsz', 640),
            batch=config.get('batch_size', 16),
            patience=config.get('patience', 100),
            project=str(results_path.parent),
            name=results_path.name,
            exist_ok=True,
            conf=config.get('eval_conf'), # 驗證時的 conf
            iou=config.get('eval_iou')    # 驗證時的 iou
        )

        training_time = time.time() - start_time; final_ram = process.memory_info().rss; best_model_path = results_path / 'weights' / 'best.pt'
        if not best_model_path.exists(): return None
        training_results = {'training_time_minutes': f"{training_time / 60:.2f}", 'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', 'Ram_GB': f"{(final_ram - initial_ram) / (1024 ** 3):.2f}", 'best_model_path': str(best_model_path)}
        print(f"訓練完成！耗時: {training_results['training_time_minutes']} 分鐘"); return training_results
    except Exception as e:
        print(f"訓練過程中發生嚴重錯誤: {e}"); import traceback; traceback.print_exc()
        return None