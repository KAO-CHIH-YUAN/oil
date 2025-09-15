import time, torch, psutil
from ultralytics import YOLO
from pathlib import Path
from utils import create_temp_data_yaml
from deeplab_adapter import DeepLabAdapter 

def train_model(config, results_path):
    exp_name = config['experiment_name']
    print(f"\n{'='*20} 開始訓練: {exp_name} {'='*20}")
    print(f"結果將儲存於: {results_path}")

    try:
        dataset_config = config.get('dataset', {})
        if not dataset_config:
            print("  [錯誤] 訓練任務未定義 'dataset'。")
            return None
        
        dataset_config.setdefault('train', 'images/train')
        dataset_config.setdefault('val', 'images/val')
        dataset_config.setdefault('test', 'images/test')
        dataset_config.setdefault('nc', 1)
        dataset_config.setdefault('names', ['oil'])
        
        temp_yaml_path = create_temp_data_yaml(dataset_config, results_path)

        # --- 根據 architecture 參數選擇模型 ---
        if config.get('architecture') == 'deeplabv3+':
            model = DeepLabAdapter(config.get('base_model'))
        else:
            model = YOLO(config['base_model'])
        
        start_time = time.time()
        process = psutil.Process()
        initial_ram = process.memory_info().rss
        
        # --- 呼叫模型的 train 方法 (無論是 YOLO 還是我們的 Adapter) ---
        training_output = model.train(
            data=str(temp_yaml_path),
            epochs=config.get('epochs', 100),
            imgsz=config.get('imgsz', 640),
            batch=config.get('batch_size', 16),
            patience=config.get('patience', 100),
            project=str(results_path.parent),
            name=results_path.name,
            exist_ok=True,
            conf=config.get('eval_conf', 0.25),
            iou=config.get('eval_iou', 0.6),
            cls=config.get('cls_weight', 0.5)
        )

        training_time = time.time() - start_time
        final_ram = process.memory_info().rss
        
        # 兼容 YOLO 和我們的 Adapter 的回傳結果
        best_model_path_str = training_output.get('best_model_path') if isinstance(training_output, dict) else str(results_path / 'weights' / 'best.pt')
        best_model_path = Path(best_model_path_str)

        if not best_model_path.exists():
            print("  [錯誤] 訓練結束，但找不到 best.pt 或 best.pth 檔案。")
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