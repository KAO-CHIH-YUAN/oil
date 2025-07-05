import yaml
from pathlib import Path
import datetime
from collections import OrderedDict
import copy
import pandas as pd

from training_module import train_model
from evaluation_module import evaluate_and_visualize
from tracking_module import log_to_excel
from utils import get_image_counts, create_temp_data_yaml

def run_evaluation_job(exp_config, model_to_evaluate, results_path, excel_path, desired_order, run_timestamp, training_metrics=None):
    """
    一個獨立的評估工作函式，負責執行單次評估並記錄結果。
    """
    print(f"\n--- 開始評估任務: {exp_config.get('test_name', exp_config.get('experiment_name'))} ---")
    print(f"--- 使用模型: {model_to_evaluate} ---")

    eval_dataset_config = exp_config.get('dataset', {})
    if not eval_dataset_config:
        print("  [錯誤] 評估任務未定義 'dataset'。")
        return

    eval_dataset_config.setdefault('train', 'images/train')
    eval_dataset_config.setdefault('val', 'images/val')
    eval_dataset_config.setdefault('test', 'images/test')
    eval_dataset_config.setdefault('nc', 1)
    eval_dataset_config.setdefault('names', ['oil'])
    
    eval_results_path = results_path
    eval_results_path.mkdir(exist_ok=True, parents=True)
    
    temp_yaml_path = create_temp_data_yaml(eval_dataset_config, eval_results_path)

    log_data = {k: v for k, v in exp_config.items() if not isinstance(v, (dict, list))}
    log_data['run_timestamp'] = run_timestamp
    log_data['Experiment_name'] = log_data.pop('experiment_name', 'N/A')
    if 'test_name' in log_data:
        log_data['Test name'] = log_data.pop('test_name')

    if 'imgsz' in log_data: log_data['Image size'] = log_data.pop('imgsz')
    if 'epochs' in log_data: log_data['Epochs'] = log_data.pop('epochs', None)
    if 'batch_size' in log_data: log_data['Batch size'] = log_data.pop('batch_size', None)

    if training_metrics:
        log_data.update(training_metrics)

    log_data.update(get_image_counts(eval_dataset_config, eval_results_path))
    log_data['best_model_path'] = str(model_to_evaluate)
    log_data['Results_folder'] = str(eval_results_path)

    eval_metrics = evaluate_and_visualize(exp_config, temp_yaml_path, model_to_evaluate, eval_results_path)
    if eval_metrics:
        log_data.update(eval_metrics)

    ordered_log_data = OrderedDict()
    for key in desired_order:
        if key in log_data: ordered_log_data[key] = log_data[key]
    for key, value in log_data.items():
        if key not in ordered_log_data: ordered_log_data[key] = value

    log_to_excel(excel_path, ordered_log_data, desired_order)


def main():
    """
    專案主進入點函式。
    """
    try:
        with open('OIL_PROJECT/code/main/experiments.yaml', 'r', encoding='utf-8') as f:
            master_config = yaml.safe_load(f)
    except FileNotFoundError:
        print("錯誤：找不到 experiments.yaml 檔案！")
        return

    results_base_dir = Path(master_config['results_base_dir'])
    excel_path = master_config['excel_log_path']
    completed_experiments_paths = {}

    desired_order = [
        'Experiment_name', 'Test name', 'run_timestamp', 'training_time_minutes',
        'Ram_GB', 'gpu_name', 'mode', 'experiment_type', 'Results_folder', 'best_model_path',
        'base_model', 'Epochs', 'Batch size', 'Image size', 'patience',
        'eval_conf', 'eval_iou', 'train_count', 'val_count', 'test_count',
        'Precision(B)', 'Recall(B)', 'mAP50(B)', 'mAP50-95(B)', 'F1-score(B)',
        'Precision(M)', 'Recall(M)', 'mAP50(M)', 'mAP50-95(M)', 'F1-score(M)',
        'Accuracy(pixel)', 'IoU(pixel)', 
        'reconstruction_accuracy', 'reconstruction_f1_score', 'reconstruction_mean_iou'
    ]

    for exp_config in master_config.get('experiments', []):
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        current_exp_config = copy.deepcopy(exp_config)

        current_exp_config.setdefault('run', True)
        current_exp_config.setdefault('eval_conf', 0.25)
        current_exp_config.setdefault('eval_iou', 0.6)
        
        if not current_exp_config['run']:
            continue

        results_path = None
        dependency_failed = False
        is_finetune = False
        base_model_path_str = str(current_exp_config.get('base_model', ''))
        
        # ⭐⭐⭐ 新增: 加入偵錯日誌 ⭐⭐⭐
        print("\n" + "="*80)
        print(f"==> 開始處理實驗: {current_exp_config.get('experiment_name')}")
        print(f"    - 當前已完成的依賴項: {list(completed_experiments_paths.keys())}")
        print(f"    - 此實驗的 base_model: {base_model_path_str}")
        print("="*80)

        if '{{' in base_model_path_str:
            is_finetune = True
            match_found = False
            for name, completed_path in completed_experiments_paths.items():
                placeholder = f'{{{{{name}}}}}'
                if placeholder in base_model_path_str:
                    match_found = True
                    if completed_path is None:
                        print(f"==> [錯誤] 跳過實驗 '{current_exp_config['experiment_name']}'，因其依賴的實驗 '{name}' 執行失敗。")
                        dependency_failed = True
                        break
                    
                    new_base_model = base_model_path_str.replace(placeholder, str(completed_path))
                    print(f"==> [成功] 找到依賴 '{name}'，將 base_model 解析為: {new_base_model}")
                    current_exp_config['base_model'] = new_base_model
                    
                    if current_exp_config['mode'] == 'train':
                        results_path = completed_path / f"finetune_{timestamp}_{current_exp_config['experiment_name']}"
                    break
            
            if not match_found:
                 print(f"==> [警告] 在 base_model 中找到佔位符，但在已完成的實驗中找不到匹配的依賴項。請檢查 yaml 中的 experiment_name 是否有錯字。")

        if dependency_failed:
            continue
            
        if results_path is None:
            if current_exp_config['mode'] == 'test':
                base_model_path = Path(current_exp_config['base_model'])
                if base_model_path.exists() and len(base_model_path.parents) > 1:
                    results_path = base_model_path.parents[1]
                else: 
                    results_path = results_base_dir / f"{timestamp}_{current_exp_config['experiment_name']}"
            else:
                results_path = results_base_dir / f"{timestamp}_{current_exp_config['experiment_name']}"
        
        results_path.mkdir(exist_ok=True, parents=True)

        inherited_training_metrics = {}
        model_to_evaluate = None
        training_success = False

        if current_exp_config['mode'] == 'test':
            model_to_evaluate = current_exp_config['base_model']
            # ... (繼承指標的邏輯不變)
        elif current_exp_config['mode'] == 'train':
            current_exp_config['experiment_type'] = 'finetune' if is_finetune else 'train'
            training_results = train_model(current_exp_config, results_path) or {}
            if training_results and 'best_model_path' in training_results:
                model_to_evaluate = training_results.get('best_model_path')
                training_success = True
                inherited_training_metrics = training_results

        if model_to_evaluate and Path(model_to_evaluate).exists():
            # ... (後續評估邏輯不變)
            post_tests = current_exp_config.get('post_tests', [])
            if not post_tests:
                if current_exp_config['mode'] == 'train':
                     print(f"\n沒有 post_tests, 對主任務 '{current_exp_config['experiment_name']}' 進行評估。")
                     run_evaluation_job(current_exp_config, model_to_evaluate, results_path, excel_path, desired_order, timestamp, inherited_training_metrics)
            else:
                print(f"\n偵測到 post_tests, 開始執行後續測試任務...")
                for test_job in post_tests:
                    test_job.setdefault('run', True)
                    if not test_job['run']: continue

                    test_timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                    test_results_path = results_path / f"post_test_{test_timestamp}_{test_job['test_name']}"

                    test_run_config = copy.deepcopy(current_exp_config)
                    test_run_config.update(test_job)
                    
                    test_run_config.setdefault('eval_conf', current_exp_config['eval_conf'])
                    test_run_config.setdefault('eval_iou', current_exp_config['eval_iou'])

                    run_evaluation_job(test_run_config, model_to_evaluate, test_results_path, excel_path, desired_order, test_timestamp, inherited_training_metrics)

        if current_exp_config['mode'] == 'train':
            completed_experiments_paths[current_exp_config['experiment_name']] = results_path if training_success else None

    print("\n所有已設定的實驗執行完畢！")


if __name__ == '__main__':
    main()