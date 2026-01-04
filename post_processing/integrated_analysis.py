import pandas as pd
from pathlib import Path
import numpy as np
import os

def analyze_integrated(results_base_path, excel_log_path, experiment_folders):
    """
    整合分析腳本：
    1. 找出每個實驗的最佳 Fold (Best Fold)。
    2. 從 Excel Log 讀取該 Fold 的基礎指標 (IoU, F1, etc.)。
    3. 從 Best Fold 的 Reconstruction_Advanced_Analysis.xlsx 讀取重建指標 (HD95, Missed, False Alarm)。
    4. 從 Best Fold 的 post_test 資料夾讀取 Patch-level 指標。
    5. (選用) 統計所有 Fold 的錯誤分析 (Top 5 Worst Images)。
    """
    
    results_base = Path(results_base_path)
    excel_path = Path(excel_log_path)
    
    # --- 1. 載入 Excel Log ---
    df_excel = None
    if excel_path.exists():
        print(f"Loading Excel log from: {excel_path}")
        try:
            df_raw = pd.read_excel(excel_path)
            if 'Metric / Parameter' in df_raw.columns:
                df_raw.set_index('Metric / Parameter', inplace=True)
                df_excel = df_raw.T
                print("  [Info] Transposed Excel data to match expected format.")
            else:
                df_excel = df_raw
        except Exception as e:
            print(f"[Error] Failed to read Excel file: {e}")
    else:
        print(f"[Warning] Excel log file not found: {excel_path}")

    # 定義要提取的指標 (來自 Excel Log)
    metrics_from_log = [
        'Precision(pixel)',
        'Recall(pixel)',
        'F1-score(pixel)',
        'Accuracy(pixel)',
        'IoU(pixel)',
        'IoU_Bg(pixel)',
        'mIoU(pixel)',
        'reconstruction_accuracy',
        'reconstruction_f1_score',
        'reconstruction_mean_iou',
        'reconstruction_iou_oil',
        'reconstruction_iou_bg'
    ]
    
    collected_data = []

    print("\n" + "="*120)
    print(f"Processing {len(experiment_folders)} experiments...")
    print("="*120)

    for exp_folder_name in experiment_folders:
        exp_path = results_base / exp_folder_name
        
        if not exp_path.exists():
            print(f"Skipping {exp_folder_name}: Not Found")
            continue

        # --- 2. 找出該實驗中最佳的 Fold ---
        folds = sorted([d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith('fold_')])
        
        best_fold_name = None
        best_val_iou = -1.0
        best_fold_path = None

        for fold_dir in folds:
            log_file = fold_dir / 'training_log.csv'
            if not log_file.exists():
                continue
            
            try:
                df_log = pd.read_csv(log_file)
                if 'val_iou' in df_log.columns and not df_log.empty:
                    max_iou = df_log['val_iou'].max()
                    if max_iou > best_val_iou:
                        best_val_iou = max_iou
                        best_fold_name = fold_dir.name
                        best_fold_path = fold_dir
            except Exception:
                pass
        
        if not best_fold_name:
            print(f"Skipping {exp_folder_name}: No valid training log found.")
            continue

        # 初始化資料字典
        data_entry = {
            'Experiment': exp_folder_name,
            'Best Fold': best_fold_name,
            'Val IoU': best_val_iou
        }

        # --- 3. 提取 Excel Log 指標 ---
        if df_excel is not None:
            target_path_str = str(best_fold_path.resolve())
            matched_row = None
            
            if 'Results_folder' in df_excel.columns:
                # 嘗試精確匹配
                matches = df_excel[df_excel['Results_folder'].astype(str) == target_path_str]
                if not matches.empty:
                    matched_row = matches.iloc[0]
                else:
                    # 嘗試寬鬆匹配
                    mask = df_excel['Results_folder'].astype(str).apply(
                        lambda x: exp_folder_name in x and best_fold_name in x
                    )
                    matches = df_excel[mask]
                    if not matches.empty:
                        matched_row = matches.iloc[-1]
            
            if matched_row is not None:
                for metric in metrics_from_log:
                    data_entry[metric] = matched_row.get(metric, np.nan)
            else:
                # print(f"  [Warning] No Excel entry found for {exp_folder_name} ({best_fold_name})")
                pass

        # --- 4. 提取 Reconstruction Analysis (Best Fold) ---
        recon_file = best_fold_path / 'Reconstruction_Advanced_Analysis.xlsx'
        if not recon_file.exists():
            # 嘗試找 post_test
            post_test_files = list(best_fold_path.glob('post_test_*/Advanced_Analysis.xlsx'))
            if post_test_files:
                recon_file = post_test_files[0] # 暫時用這個代替，雖然通常結構不同
        
        if recon_file.exists():
            try:
                df_recon = pd.read_excel(recon_file)
                # 檢查是否有需要的欄位
                if 'HD95' in df_recon.columns:
                    data_entry['Recon_HD95'] = df_recon['HD95'].mean()
                    data_entry['Recon_GT'] = df_recon['GT_Count'].sum()
                    data_entry['Recon_Missed'] = df_recon['Missed'].sum()
                    data_entry['Recon_FA'] = df_recon['False_Alarm'].sum()
            except Exception as e:
                print(f"  [Error] Reading Reconstruction Analysis: {e}")

        # --- 5. 提取 Patch Level Analysis (Best Fold) ---
        # 通常在 post_test_.../Advanced_Analysis.xlsx
        post_test_dirs = sorted(list(best_fold_path.glob('post_test_*')))
        if post_test_dirs:
            patch_file = post_test_dirs[-1] / 'Advanced_Analysis.xlsx'
            if patch_file.exists():
                try:
                    df_patch = pd.read_excel(patch_file)
                    last_row = df_patch.iloc[-1] # Summary row
                    
                    # 假設最後一列是統計值
                    data_entry['Patch_HD95'] = last_row.get('HD95', np.nan)
                    data_entry['Patch_GT'] = last_row.get('GT_Count', np.nan)
                    data_entry['Patch_Missed'] = last_row.get('Missed', np.nan)
                    data_entry['Patch_FA'] = last_row.get('False_Alarm', np.nan)
                except Exception as e:
                    print(f"  [Error] Reading Patch Analysis: {e}")

        collected_data.append(data_entry)

    # --- 6. 輸出報表 ---
    if not collected_data:
        print("No data collected.")
        return

    df_results = pd.DataFrame(collected_data)
    
    # 調整欄位順序
    cols_order = ['Experiment', 'Best Fold', 'Val IoU']
    # 加入其他存在的欄位
    for col in df_results.columns:
        if col not in cols_order:
            cols_order.append(col)
    
    df_results = df_results[cols_order]

    print("\n" + "="*150)
    print("Integrated Analysis Report (Best Folds) - Side by Side")
    print("="*150)
    
    # 轉置 DataFrame 以便並排顯示
    df_t = df_results.set_index('Experiment').T
    
    # 取得實驗名稱
    exp_names = df_t.columns.tolist()
    
    # 設定欄寬
    metric_col_width = 30
    exp_col_width = 45
    
    # 印出標頭
    header = f"{'Metric':<{metric_col_width}}"
    for name in exp_names:
        # 如果名稱太長，截斷顯示
        display_name = (name[:exp_col_width-3] + '...') if len(name) > exp_col_width else name
        header += f" | {display_name:<{exp_col_width}}"
    print(header)
    print("-" * len(header))
    
    # 印出每一列 (Metric)
    for metric in df_t.index:
        row_str = f"{metric:<{metric_col_width}}"
        for name in exp_names:
            val = df_t.loc[metric, name]
            if isinstance(val, (float, np.floating)):
                if pd.isna(val):
                    val_str = "NaN"
                elif val.is_integer():
                    val_str = f"{int(val)}"
                else:
                    val_str = f"{val:.3f}"
            else:
                val_str = str(val)
            row_str += f" | {val_str:<{exp_col_width}}"
        print(row_str)
    
    # 輸出 CSV
    output_csv = results_base / 'integrated_analysis_report.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"\nReport saved to: {output_csv}")

    # --- 6.5 計算平均與標準差 ---
    print("\n" + "="*120)
    print("Average Results across Experiments (Best Folds)")
    print("="*120)
    
    # 篩選出數值欄位進行計算
    numeric_cols = [col for col in df_results.columns if col not in ['Experiment', 'Best Fold']]
    
    stats_data = []
    for col in numeric_cols:
        # 確保轉為數值型別，無法轉換的變成 NaN
        series = pd.to_numeric(df_results[col], errors='coerce')
        if series.notna().any():
            mean_val = series.mean()
            std_val = series.std()
            stats_data.append({
                'Metric': col,
                'Mean': mean_val,
                'Std': std_val
            })
            print(f"{col:<30}: {mean_val:.3f} ± {std_val:.3f}")
    
    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        stats_csv = results_base / 'integrated_analysis_stats.csv'
        df_stats.to_csv(stats_csv, index=False)
        print(f"\nStats saved to: {stats_csv}")

    # --- 7. 詳細錯誤分析 (All Folds) ---
    print("\n" + "="*120)
    print("Detailed Error Analysis (All Folds - Top 5 Worst Images)")
    print("="*120)
    
    for exp_folder in experiment_folders:
        exp_path = results_base / exp_folder
        if not exp_path.exists():
            continue
            
        print(f"\nExperiment: {exp_folder}")
        print("-" * 60)
        
        # 收集所有 Fold 的資料
        all_folds_df = []
        folds = sorted([d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith('fold_')])
        
        for fold_dir in folds:
            # 優先找 Reconstruction_Advanced_Analysis.xlsx
            analysis_file = fold_dir / 'Reconstruction_Advanced_Analysis.xlsx'
            if not analysis_file.exists():
                 # 嘗試找 post_test
                 post_test_files = list(fold_dir.glob('post_test_*/Advanced_Analysis.xlsx'))
                 if post_test_files:
                     analysis_file = post_test_files[0]

            if analysis_file.exists():
                try:
                    df = pd.read_excel(analysis_file)
                    df['Fold'] = fold_dir.name
                    all_folds_df.append(df)
                except Exception:
                    pass

        if not all_folds_df:
            print("  No analysis files found.")
            continue
            
        full_df = pd.concat(all_folds_df, ignore_index=True)
        
        # (A) False Alarm Top 5
        print("  [Top 5 False Alarm Images]")
        if 'False_Alarm' in full_df.columns:
            fa_top5 = full_df.nlargest(5, 'False_Alarm')
            if not fa_top5.empty and fa_top5['False_Alarm'].sum() > 0:
                for _, row in fa_top5.iterrows():
                    if row['False_Alarm'] > 0:
                        print(f"    - {row['filename']} ({row['Fold']}): {row['False_Alarm']} False Alarms")
            else:
                print("    None")

        # (B) Missed Top 5
        print("\n  [Top 5 Missed Images]")
        if 'Missed' in full_df.columns:
            missed_top5 = full_df.nlargest(5, 'Missed')
            if not missed_top5.empty and missed_top5['Missed'].sum() > 0:
                for _, row in missed_top5.iterrows():
                    if row['Missed'] > 0:
                        print(f"    - {row['filename']} ({row['Fold']}): {row['Missed']} Missed")
            else:
                print("    None")

        # (C) HD95 Top 5
        print("\n  [Top 5 Worst HD95 Images]")
        if 'HD95' in full_df.columns:
            hd95_top5 = full_df.nlargest(5, 'HD95')
            if not hd95_top5.empty:
                for _, row in hd95_top5.iterrows():
                    print(f"    - {row['filename']} ({row['Fold']}): HD95 = {row['HD95']:.4f}")

if __name__ == "__main__":
    # ==========================================
    # 使用者設定區
    # ==========================================

    RESULTS_BASE_PATH = "/home/yuan/Yuan/OIL_Project_12_7/result/model/DV4_small_transferPaperRGB_CV"
    EXCEL_LOG_PATH = "/home/yuan/Yuan/OIL_Project_12_7/result/excel_log/DV4_small_transferPaperRGB_CV_log.xlsx"
    
    EXPERIMENT_FOLDERS = [
        "20251219-141140_Segformer_2048_transferPaperRGB",
        "20251219-220542_Segformer_2048_transferPaperRGB",
        "20251221-135401_Segformer_2048_transferPaperRGB",
        "20251221-194328_Segformer_2048_transferPaperRGB",
    ]
    
    # RESULTS_BASE_PATH = "/home/yuan/Yuan/OIL_Project_12_7/result/model/DV4_small_transferPaperRGB_CV"
    # EXCEL_LOG_PATH = "/home/yuan/Yuan/OIL_Project_12_7/result/excel_log/DV4_small_transferPaperRGB_CV_log.xlsx"
    
    # EXPERIMENT_FOLDERS = [
    #     "20251229-070926_Segformer_2048_transferPaperRGB_dice_bce",
    #     "20251229-085921_Segformer_2048_transferPaperRGB_dice_bce",
    #     "20251229-110449_Segformer_2048_transferPaperRGB_dice_bce",
    # ]

    # RESULTS_BASE_PATH = "/home/yuan/Yuan/OIL_Project_12_7/result/model/DV4_All_transferPaperRGB"
    # EXCEL_LOG_PATH = "/home/yuan/Yuan/OIL_Project_12_7/result/excel_log/DV4_All_transferPaperRGB_CV_log.xlsx"
    
    # EXPERIMENT_FOLDERS = [
    #     "20251229-005506_Segformer_2048_transferPaperRGB",
    #     "20251229-030752_Segformer_2048_transferPaperRGB",
    # ]
    
    # ==========================================
    
    analyze_integrated(RESULTS_BASE_PATH, EXCEL_LOG_PATH, EXPERIMENT_FOLDERS)
