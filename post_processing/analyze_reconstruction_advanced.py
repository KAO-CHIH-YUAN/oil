import pandas as pd
from pathlib import Path
import numpy as np

def analyze_reconstruction_advanced(results_base_path, experiment_folders):
    """
    分析 Reconstruction_Advanced_Analysis.xlsx，
    合併各 Fold 的結果，並統計誤報 (False Alarm) 與漏報 (Missed) 情形。
    """
    results_base = Path(results_base_path)
    
    print("\n" + "="*80)
    print("Reconstruction Advanced Analysis Report")
    print("="*80)

    for exp_folder in experiment_folders:
        exp_path = results_base / exp_folder
        if not exp_path.exists():
            print(f"\nExperiment: {exp_folder} [Not Found]")
            continue
            
        print(f"\nExperiment: {exp_folder}")
        print("-" * 60)
        
        # 1. 收集所有 Fold 的資料
        all_folds_df = []
        folds = sorted([d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith('fold_')])
        
        for fold_dir in folds:
            analysis_file = fold_dir / 'Reconstruction_Advanced_Analysis.xlsx'
            if analysis_file.exists():
                try:
                    df = pd.read_excel(analysis_file)
                    df['Fold'] = fold_dir.name # 標記來源 Fold
                    all_folds_df.append(df)
                except Exception as e:
                    print(f"  [Error] Failed to read {fold_dir.name}: {e}")
            else:
                # 嘗試找 post_test 裡面的 (如果 fold root 沒有)
                # 但根據觀察，Reconstruction_Advanced_Analysis.xlsx 通常在 fold root
                pass

        if not all_folds_df:
            print("  No analysis files found.")
            continue
            
        # 合併所有 Fold 的資料
        full_df = pd.concat(all_folds_df, ignore_index=True)
        
        # 2. 計算總體統計
        total_gt = full_df['GT_Count'].sum()
        total_pred = full_df['Pred_Count'].sum()
        total_missed = full_df['Missed'].sum()
        total_fa = full_df['False_Alarm'].sum()
        avg_hd95 = full_df['HD95'].mean()
        
        print(f"  Total Images Processed: {len(full_df)}")
        print(f"  Total Ground Truth Objects: {total_gt}")
        print(f"  Total Predicted Objects:    {total_pred}")
        print(f"  Total Missed Objects:       {total_missed}")
        print(f"  Total False Alarms:         {total_fa}")
        print(f"  Average HD95:               {avg_hd95:.4f}")
        
        # 3. 找出問題最大的圖片 (Top 5)
        
        # (A) False Alarm Top 5
        print("\n  [Top 5 False Alarm Images]")
        fa_top5 = full_df.nlargest(5, 'False_Alarm')
        if not fa_top5.empty and fa_top5['False_Alarm'].sum() > 0:
            for _, row in fa_top5.iterrows():
                if row['False_Alarm'] > 0:
                    print(f"    - {row['filename']} (Fold: {row['Fold']}): {row['False_Alarm']} False Alarms")
        else:
            print("    None")

        # (B) Missed Top 5
        print("\n  [Top 5 Missed Images]")
        missed_top5 = full_df.nlargest(5, 'Missed')
        if not missed_top5.empty and missed_top5['Missed'].sum() > 0:
            for _, row in missed_top5.iterrows():
                if row['Missed'] > 0:
                    print(f"    - {row['filename']} (Fold: {row['Fold']}): {row['Missed']} Missed")
        else:
            print("    None")

        # (C) HD95 Top 5 (Worst Shape Match)
        print("\n  [Top 5 Worst HD95 Images]")
        hd95_top5 = full_df.nlargest(5, 'HD95')
        if not hd95_top5.empty:
            for _, row in hd95_top5.iterrows():
                print(f"    - {row['filename']} (Fold: {row['Fold']}): HD95 = {row['HD95']:.4f}")

if __name__ == "__main__":
    # ==========================================
    # 使用者設定區
    # ==========================================
    
    RESULTS_BASE_PATH = "/home/yuan/Yuan/OIL_Project_12_7/result/model/DV4_All_transferPaperRGB"
    
    EXPERIMENT_FOLDERS = [
        "20251229-005506_Segformer_2048_transferPaperRGB",
        "20251229-030752_Segformer_2048_transferPaperRGB",
    ]
    
    # ==========================================
    
    analyze_reconstruction_advanced(RESULTS_BASE_PATH, EXPERIMENT_FOLDERS)
