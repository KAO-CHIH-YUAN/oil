import yaml
from pathlib import Path
import datetime
from collections import OrderedDict
import copy

from training_module import train_model
from evaluation_module import evaluate_and_visualize
from tracking_module import log_to_excel
from utils import get_image_counts, create_temp_data_yaml

def run_evaluation_job(exp_config, model_to_evaluate, results_path, excel_path, desired_order, run_timestamp, training_metrics=None):
    """
    一個輔助函式，專門用來執行一次評估並記錄結果。
    """
    print(f"\n--- 開始評估任務: {exp_config.get('test_name', exp_config.get('experiment_name'))} ---")
    print(f"--- 使用模型: {model_to_evaluate} ---")

    eval_dataset_config = exp_config.get('dataset', {})
    if not eval_dataset_config:
        print("  [錯誤] 評估任務未定義 'dataset'。")
        return

    eval_results_path = results_path
    eval_results_path.mkdir(exist_ok=True, parents=True)
    temp_yaml_path = create_temp_data_yaml(eval_dataset_config, eval_results_path)

    # --- 準備記錄檔 ---
    log_data = {k: v for k, v in exp_config.items() if not isinstance(v, (dict, list))}

    # 將傳入的唯一時間戳加入 log_data
    log_data['run_timestamp'] = run_timestamp

    # 處理並分離 Experiment_name 和 Test name
    log_data['Experiment_name'] = log_data.pop('experiment_name', 'N/A')
    if 'test_name' in log_data:
        log_data['Test name'] = log_data.pop('test_name')

    # 統一參數名稱，避免重複 (使用 .pop(key, None) 確保鍵不存在時不報錯)
    if 'imgsz' in log_data: log_data['Image size'] = log_data.pop('imgsz')
    if 'epochs' in log_data: log_data['Epochs'] = log_data.pop('epochs', None)
    if 'batch_size' in log_data: log_data['Batch size'] = log_data.pop('batch_size', None)

    # 合併從主流程傳入的訓練指標 (如果有的話)
    if training_metrics:
        log_data.update(training_metrics)

    log_data.update(get_image_counts(eval_dataset_config, eval_results_path))
    log_data['best_model_path'] = str(model_to_evaluate)
    log_data['Results_folder'] = str(eval_results_path)

    # --- 執行評估 ---
    eval_metrics = evaluate_and_visualize(exp_config, temp_yaml_path, model_to_evaluate, eval_results_path)
    if eval_metrics:
        log_data.update(eval_metrics)

    # --- 排序並記錄到 Excel ---
    ordered_log_data = OrderedDict()
    for key in desired_order:
        if key in log_data: ordered_log_data[key] = log_data[key]
    for key, value in log_data.items():
        if key not in ordered_log_data: ordered_log_data[key] = value

    log_to_excel(excel_path, ordered_log_data, desired_order)


def main():
    try:
        with open('OIL_PROJECT/code/main/experiments.yaml', 'r', encoding='utf-8') as f:
            master_config = yaml.safe_load(f)
    except FileNotFoundError:
        print("錯誤：找不到 experiments.yaml 檔案！")
        return

    results_base_dir = Path(master_config['results_base_dir'])
    excel_path = master_config['excel_log_path']
    completed_experiments_paths = {}

    # 定義 Excel 的欄位順序
    desired_order = [
        'Experiment_name', 'Test name', 'run_timestamp', 'training_time_minutes',
        'Ram_GB', 'mode', 'experiment_type', 'Results_folder', 'best_model_path',
        'base_model', 'Epochs', 'Batch size', 'Image size', 'patience',
        'eval_conf', 'eval_iou', 'train_count', 'val_count', 'test_count',
        'Precision(B)', 'Recall(B)', 'mAP50(B)', 'mAP50-95(B)', 'F1-score(B)',
        'Precision(M)', 'Recall(M)', 'mAP50(M)', 'mAP50-95(M)', 'F1-score(M)',
        'Accuracy(pixel)', 'IoU(pixel)', 'reconstruction_mean_iou'
    ]

    for exp_config in master_config.get('experiments', []):
        if not exp_config.get('run', True):
            continue

        # 為每個主實驗任務產生一個時間戳，精確到秒以確保唯一性
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        current_exp_config = copy.deepcopy(exp_config)

        # --- 路徑與依賴處理 ---
        base_model_path_str = str(current_exp_config['base_model'])
        is_finetune_dependency, dependency_failed, results_path = False, False, None
        for name, completed_path in completed_experiments_paths.items():
            placeholder = f'{{{{{name}}}}}'
            if placeholder in base_model_path_str:
                if completed_path is None:
                    dependency_failed = True
                    break
                current_exp_config['base_model'] = base_model_path_str.replace(placeholder, str(completed_path))
                if current_exp_config['mode'] == 'train':
                    is_finetune_dependency = True
                    results_path = completed_path / f"finetune_{timestamp}_{current_exp_config['experiment_name']}"
                break
        if dependency_failed:
            print(f"跳過實驗 '{current_exp_config['experiment_name']}'，因為其依賴的實驗 '{name}' 執行失敗。")
            continue
            
        if results_path is None:
            results_path = results_base_dir / f"{timestamp}_{current_exp_config['experiment_name']}"
        results_path.mkdir(exist_ok=True, parents=True)

        # --- 定義 experiment_type ---
        exp_type = 'test'  # 預設為 test
        if current_exp_config['mode'] == 'train':
            exp_type = 'finetune' if is_finetune_dependency else 'train'
        current_exp_config['experiment_type'] = exp_type

        # --- 執行主任務 (Train or Test) ---
        model_to_evaluate, training_success = None, False
        training_results = {}  # 初始化為空字典，以處理 test-only 模式

        if current_exp_config['mode'] == 'train':
            # 確保 training_results 即使訓練失敗也是一個字典
            training_results = train_model(current_exp_config, results_path) or {}
            if training_results and 'best_model_path' in training_results:
                model_to_evaluate = training_results.get('best_model_path')
                training_success = True
        elif current_exp_config['mode'] == 'test':
            model_to_evaluate = current_exp_config['base_model']

        # --- 處理評估與日誌記錄 ---
        if model_to_evaluate and Path(model_to_evaluate).exists():
            post_tests = current_exp_config.get('post_tests', [])
            if not post_tests:
                print(f"\n沒有 post_tests, 對主任務 '{current_exp_config['experiment_name']}' 進行評估。")
                # 傳入主任務的時間戳 timestamp
                run_evaluation_job(current_exp_config, model_to_evaluate, results_path, excel_path, desired_order, timestamp, training_results)
            else:
                print(f"\n偵測到 post_tests, 開始執行後續測試任務...")
                for test_job in post_tests:
                    if not test_job.get('run', True):
                        continue

                    # 為每個 post_test 產生獨立且唯一的時間戳
                    test_timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                    test_results_path = results_path / f"post_test_{test_timestamp}_{test_job['test_name']}"

                    test_run_config = copy.deepcopy(current_exp_config)
                    test_run_config.update(test_job)

                    # 傳入 post_test 自己的時間戳 test_timestamp
                    run_evaluation_job(test_run_config, model_to_evaluate, test_results_path, excel_path, desired_order, test_timestamp, training_results)

        if current_exp_config['mode'] == 'train':
            completed_experiments_paths[current_exp_config['experiment_name']] = results_path if training_success else None

    print("\n所有已設定的實驗執行完畢！")


if __name__ == '__main__':
    main()