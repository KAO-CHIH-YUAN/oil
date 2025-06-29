import yaml
from pathlib import Path

def create_temp_data_yaml(dataset_config, target_dir):
    """在目標資料夾中建立一個暫時的 data.yaml 檔案。"""
    temp_yaml_path = target_dir / 'data.yaml'
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, allow_unicode=True, sort_keys=False)
    return temp_yaml_path

def get_image_counts(dataset_config, yaml_parent_dir):
    """
    根據 dataset 字典計算 train/val/test 圖片數量。
    """
    try:
        # 假設 dataset_config['path'] 是相對於專案根目錄的相對路徑
        project_root = Path.cwd() 
        # 使用 lstrip('../') 移除可能的前導 '..'，使其成為從根目錄開始的相對路徑
        base_path = project_root / dataset_config['path'].lstrip('../')
        
        counts = {}
        for split in ['train', 'val', 'test']:
            if split in dataset_config:
                image_dir = base_path / dataset_config[split]
                if image_dir.is_dir():
                    # 計算 .jpg 和 .png 檔案的總數
                    num_images = len(list(image_dir.glob('*.jpg'))) + len(list(image_dir.glob('*.png')))
                    counts[f'{split}_count'] = num_images
                else:
                    counts[f'{split}_count'] = 'Path not found'
            else:
                counts[f'{split}_count'] = 'Not specified'
        return counts
    except Exception as e:
        print(f" 計算圖片數量時出錯: {e}")
        return {'train_count': -1, 'val_count': -1, 'test_count': -1}