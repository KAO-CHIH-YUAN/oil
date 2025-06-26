from ultralytics import YOLO
from pathlib import Path
import cv2
import yaml
import os
import torch # 為了 DEVICE 設定

# --- 1. 設定參數 ---
# ------------------------------------------------------------------------------------
# 請根據您的實際情況修改以下路徑和參數
# ------------------------------------------------------------------------------------
MODEL_PATH = 'runs\detect\yolo(11n-500epoch)_dataset(zenodo)_dataset(SAR)_detection\weights\\best.pt'  # <<----------- 修改為您訓練好的 YOLO 模型權重檔案路徑 (例如 runs/detect/train/weights/best.pt)
YAML_DATA_PATH = 'train_datasetv2\data_datasetv2.yaml' # <<------ 修改為您的 data.yaml 檔案路徑

# 輸出資料夾設定 (這些資料夾會自動建立)
OUTPUT_BASE_DIR = Path('result\yolo(11n-500epoch)_dataset(zenodo)_dataset(SAR)_detection') # 主要的輸出基礎資料夾
OUTPUT_TXT_DIR_NAME = 'bounding_box_txt'       # 儲存 bounding box 文字檔案的子資料夾名稱
OUTPUT_IMG_DIR_NAME = 'detected_images'        # 儲存已繪製偵測框圖片的子資料夾名稱

CONF_THRESHOLD = 0.4  # 物件偵測的置信度閾值 (0.0 - 1.0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 自動選擇 CUDA 或 CPU
# ------------------------------------------------------------------------------------

def run_detection():
    """
    執行 YOLO 物件偵測，並儲存 bounding box 文字檔案和已繪製的圖片。
    """
    print(f"使用的設備: {DEVICE}")
    
    # --- 2. 建立輸出資料夾 ---
    output_txt_path = OUTPUT_BASE_DIR / OUTPUT_TXT_DIR_NAME
    output_img_path = OUTPUT_BASE_DIR / OUTPUT_IMG_DIR_NAME
    try:
        output_txt_path.mkdir(parents=True, exist_ok=True)
        output_img_path.mkdir(parents=True, exist_ok=True)
        print(f"Bounding box 文字檔案將儲存於: {output_txt_path.resolve()}")
        print(f"已偵測圖片將儲存於: {output_img_path.resolve()}")
    except OSError as e:
        print(f"建立輸出資料夾失敗: {e}")
        return

    # --- 3. 載入 YAML 設定以取得測試圖片路徑和類別名稱 ---
    try:
        with open(YAML_DATA_PATH, 'r', encoding='utf-8') as file:
            yaml_config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"錯誤：找不到 YAML 檔案 '{YAML_DATA_PATH}'")
        return
    except Exception as e:
        print(f"讀取 YAML 檔案 '{YAML_DATA_PATH}' 時發生錯誤: {e}")
        return

    try:
        # YAML 檔案中的 'path' 是相對於 YAML 檔案本身的路徑
        # YAML 檔案中的 'test' 是相對於上面 'path' 的路徑
        yaml_file_dir = Path(YAML_DATA_PATH).parent
        dataset_root_in_yaml = yaml_config.get('path', '.') # 資料集根目錄 (相對於 YAML)
        test_imgs_subdir_in_yaml = yaml_config.get('test')  # test 資料夾 (相對於 dataset_root_in_yaml)
        class_names = yaml_config.get('names', ['object']) # 類別名稱

        if test_imgs_subdir_in_yaml is None:
            print(f"錯誤：YAML 檔案 '{YAML_DATA_PATH}' 中未指定 'test' 路徑。")
            return

        # 組成完整的測試圖片資料夾路徑
        test_image_dir = (yaml_file_dir / dataset_root_in_yaml / test_imgs_subdir_in_yaml).resolve()
        
        if not test_image_dir.is_dir():
            print(f"錯誤：測試圖片資料夾 '{test_image_dir}' 不存在或不是一個目錄。")
            print(f"  - YAML 檔案位置: {Path(YAML_DATA_PATH).resolve()}")
            print(f"  - YAML 'path' 鍵值: {dataset_root_in_yaml}")
            print(f"  - YAML 'test' 鍵值: {test_imgs_subdir_in_yaml}")
            return
        print(f"從 YAML 解析得到的測試圖片資料夾: '{test_image_dir}'")

    except KeyError as e:
        print(f"錯誤：YAML 檔案 '{YAML_DATA_PATH}' 缺少必要的鍵 '{e}' (例如 'path', 'test', 'names')")
        return
    except Exception as e:
        print(f"處理 YAML 設定時發生錯誤: {e}")
        return

    # --- 4. 載入 YOLO 模型 ---
    try:
        model = YOLO(MODEL_PATH)
        model.to(DEVICE) # 將模型移至指定設備
        print(f"YOLO 模型 '{MODEL_PATH}' 載入成功。")
    except FileNotFoundError:
        print(f"錯誤：找不到 YOLO 模型檔案 '{MODEL_PATH}'。請確認路徑是否正確。")
        return
    except Exception as e:
        print(f"載入 YOLO 模型 '{MODEL_PATH}' 失敗: {e}")
        return

    # --- 5. 取得測試圖片列表 ---
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [p for p in test_image_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]

    if not image_files:
        print(f"在 '{test_image_dir}' 中沒有找到任何支援的圖片檔案。")
        return

    print(f"找到 {len(image_files)} 張圖片進行偵測...")

    # --- 6. 遍歷圖片進行偵測並儲存結果 ---
    for image_file_path in image_files:
        print(f"\n正在處理圖片: {image_file_path.name}")
        try:
            # 執行偵測
            # results = model.predict(source=str(image_file_path), conf=CONF_THRESHOLD, device=DEVICE)
            results = model(str(image_file_path), conf=CONF_THRESHOLD, device=DEVICE) # predict 是舊版用法，新版直接呼叫 model
        except Exception as e:
            print(f"  對圖片 '{image_file_path.name}' 進行偵測時發生錯誤: {e}")
            continue

        if not results or len(results) == 0:
            print(f"  未偵測到任何結果 (results 列表為空)。")
            continue
        
        result = results[0] # 通常一張圖片對應一個 result 物件
        boxes = result.boxes  # Boxes 物件

        # 載入原始圖片以供繪製
        try:
            img_cv = cv2.imread(str(image_file_path))
            if img_cv is None:
                print(f"  錯誤：無法使用 OpenCV 讀取圖片 '{image_file_path.name}'。")
                continue
        except Exception as e:
            print(f"  使用 OpenCV 讀取圖片 '{image_file_path.name}' 時發生錯誤: {e}")
            continue

        # 準備儲存 bounding box 文字檔案的內容
        txt_output_lines = []

        if boxes is not None and len(boxes.cls) > 0:
            print(f"  偵測到 {len(boxes.cls)} 個物件。")
            for i in range(len(boxes.cls)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                # 使用 boxes.xyxy 獲取像素座標 [x1, y1, x2, y2]
                xyxy_pixel = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy_pixel

                class_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"

                # 準備文字檔案中的一行 (格式：x1 y1 x2 y2 confidence class_id class_name)
                # 這個格式與您先前 FastSAM 腳本中讀取的格式類似
                txt_line = f"{x1} {y1} {x2} {y2} {conf:.4f} {cls_id} {class_name}\n"
                txt_output_lines.append(txt_line)

                # 在圖片上繪製 bounding box 和標籤
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2) # 綠色框
                
                # 放置標籤文字
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                y1_label = max(y1, label_size[1] + 10) # 確保標籤在圖片內
                cv2.rectangle(img_cv, (x1, y1_label - label_size[1] - 10), 
                              (x1 + label_size[0], y1_label - base_line -10 ), (0, 255, 0), cv2.FILLED)
                cv2.putText(img_cv, label, (x1, y1_label - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # 黑色文字

        else:
            print(f"  未偵測到符合閾值的物件。")
            txt_output_lines.append("# No objects detected or below confidence threshold\n")


        # --- 儲存 bounding box 文字檔案 ---
        txt_filename = output_txt_path / (image_file_path.stem + ".txt")
        try:
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.writelines(txt_output_lines)
            # print(f"  Bounding box 文字檔案儲存至: {txt_filename}")
        except IOError as e:
            print(f"  寫入 bounding box 文字檔案 '{txt_filename}' 失敗: {e}")

        # --- 儲存已繪製偵測框的圖片 ---
        img_output_filename = output_img_path / (image_file_path.stem + "_detected" + image_file_path.suffix)
        try:
            cv2.imwrite(str(img_output_filename), img_cv)
            # print(f"  已偵測圖片儲存至: {img_output_filename}")
        except Exception as e:
            print(f"  儲存已偵測圖片 '{img_output_filename}' 失敗: {e}")
            
    print("\n所有圖片處理完成。")

if __name__ == '__main__':
    # 確保 `from pathlib import Path` 等 import 在頂部
    # 執行偵測
    run_detection()

# from ultralytics import YOLO
# from pathlib import Path
# import cv2
# import yaml
# import os
# import torch # 為了 DEVICE 設定

# # --- 1. 設定參數 ---
# # ------------------------------------------------------------------------------------
# # 請根據您的實際情況修改以下路徑和參數
# # ------------------------------------------------------------------------------------
# MODEL_PATH = 'runs\detect\yolo(11n-500epoch)_dataset(SAR)_detection\weights\\best.pt'  # <<----------- 修改為您訓練好的 YOLO 模型權重檔案路徑 (例如 runs/detect/train/weights/best.pt)
# YAML_DATA_PATH = 'train_datasetv2\data_datasetv2.yaml' # <<------ 修改為您的 data.yaml 檔案路徑

# # 輸出資料夾設定 (這些資料夾會自動建立)
# OUTPUT_BASE_DIR = Path('result\yolo(11n-500epoch)_dataset(SAR)_detection') # 主要的輸出基礎資料夾
# OUTPUT_TXT_DIR_NAME = 'bounding_box_txt'       # 儲存 bounding box 文字檔案的子資料夾名稱
# OUTPUT_IMG_DIR_NAME = 'detected_images'        # 儲存已繪製偵測框圖片的子資料夾名稱

# CONF_THRESHOLD = 0.4  # 物件偵測的置信度閾值 (0.0 - 1.0)
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 自動選擇 CUDA 或 CPU
# # ------------------------------------------------------------------------------------

# def run_detection():
#     """
#     執行 YOLO 物件偵測，並儲存 bounding box 文字檔案和已繪製的圖片，
#     最後執行模型在測試集上的評估並印出相關指標。
#     """
#     print(f"使用的設備: {DEVICE}")

#     # --- 2. 建立輸出資料夾 ---
#     output_txt_path = OUTPUT_BASE_DIR / OUTPUT_TXT_DIR_NAME
#     output_img_path = OUTPUT_BASE_DIR / OUTPUT_IMG_DIR_NAME
#     try:
#         output_txt_path.mkdir(parents=True, exist_ok=True)
#         output_img_path.mkdir(parents=True, exist_ok=True)
#         print(f"Bounding box 文字檔案將儲存於: {output_txt_path.resolve()}")
#         print(f"已偵測圖片將儲存於: {output_img_path.resolve()}")
#     except OSError as e:
#         print(f"建立輸出資料夾失敗: {e}")
#         return

#     # --- 3. 載入 YAML 設定以取得測試圖片路徑和類別名稱 ---
#     try:
#         with open(YAML_DATA_PATH, 'r', encoding='utf-8') as file:
#             yaml_config = yaml.safe_load(file)
#     except FileNotFoundError:
#         print(f"錯誤：找不到 YAML 檔案 '{YAML_DATA_PATH}'")
#         return
#     except Exception as e:
#         print(f"讀取 YAML 檔案 '{YAML_DATA_PATH}' 時發生錯誤: {e}")
#         return

#     test_image_dir = None # 初始化
#     try:
#         yaml_file_dir = Path(YAML_DATA_PATH).parent
#         dataset_root_in_yaml = yaml_config.get('path', '.')
#         test_imgs_subdir_in_yaml = yaml_config.get('test') # 取得測試集圖片的路徑設定
#         class_names = yaml_config.get('names', ['object'])

#         if test_imgs_subdir_in_yaml is None:
#             print(f"警告：YAML 檔案 '{YAML_DATA_PATH}' 中未指定 'test' 路徑。將無法執行個別圖片的推論儲存，也可能影響模型評估的 'test' 部分。")
#         else:
#             test_image_dir = (yaml_file_dir / dataset_root_in_yaml / test_imgs_subdir_in_yaml).resolve()
#             if not test_image_dir.is_dir():
#                 print(f"警告：測試圖片資料夾 '{test_image_dir}' 不存在或不是一個目錄。將無法執行個別圖片的推論儲存。")
#                 print(f"  - YAML 檔案位置: {Path(YAML_DATA_PATH).resolve()}")
#                 print(f"  - YAML 'path' 鍵值: {dataset_root_in_yaml}")
#                 print(f"  - YAML 'test' 鍵值: {test_imgs_subdir_in_yaml}")
#                 test_image_dir = None # 設為 None，避免後續處理出錯
#             else:
#                 print(f"從 YAML 解析得到的測試圖片資料夾 (用於個別推論): '{test_image_dir}'")

#     except KeyError as e:
#         print(f"錯誤：YAML 檔案 '{YAML_DATA_PATH}' 缺少必要的鍵 '{e}' (例如 'path', 'names')")
#         return
#     except Exception as e:
#         print(f"處理 YAML 設定時發生錯誤: {e}")
#         return

#     # --- 4. 載入 YOLO 模型 ---
#     try:
#         model = YOLO(MODEL_PATH)
#         model.to(DEVICE)
#         print(f"YOLO 模型 '{MODEL_PATH}' 載入成功。")
#     except FileNotFoundError:
#         print(f"錯誤：找不到 YOLO 模型檔案 '{MODEL_PATH}'。請確認路徑是否正確。")
#         return
#     except Exception as e:
#         print(f"載入 YOLO 模型 '{MODEL_PATH}' 失敗: {e}")
#         return

#     # --- 5. 取得測試圖片列表 (如果 test_image_dir 有效) ---
#     # 這部分是針對 test 資料夾中的圖片進行逐一偵測並儲存結果
#     if test_image_dir:
#         image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
#         image_files = [p for p in test_image_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]

#         if not image_files:
#             print(f"在 '{test_image_dir}' 中沒有找到任何支援的圖片檔案 (用於個別推論)。")
#         else:
#             print(f"找到 {len(image_files)} 張圖片進行個別偵測並儲存結果...")

#             # --- 6. 遍歷圖片進行偵測並儲存結果 ---
#             for image_file_path in image_files:
#                 print(f"\n正在處理圖片 (個別推論): {image_file_path.name}")
#                 try:
#                     results = model(str(image_file_path), conf=CONF_THRESHOLD, device=DEVICE)
#                 except Exception as e:
#                     print(f"  對圖片 '{image_file_path.name}' 進行偵測時發生錯誤: {e}")
#                     continue

#                 if not results or len(results) == 0:
#                     print(f"  未偵測到任何結果 (results 列表為空)。")
#                     continue

#                 result = results[0]
#                 boxes = result.boxes

#                 try:
#                     img_cv = cv2.imread(str(image_file_path))
#                     if img_cv is None:
#                         print(f"  錯誤：無法使用 OpenCV 讀取圖片 '{image_file_path.name}'。")
#                         continue
#                 except Exception as e:
#                     print(f"  使用 OpenCV 讀取圖片 '{image_file_path.name}' 時發生錯誤: {e}")
#                     continue

#                 txt_output_lines = []

#                 if boxes is not None and len(boxes.cls) > 0:
#                     print(f"  偵測到 {len(boxes.cls)} 個物件。")
#                     for i in range(len(boxes.cls)):
#                         cls_id = int(boxes.cls[i])
#                         conf = float(boxes.conf[i])
#                         xyxy_pixel = boxes.xyxy[i].cpu().numpy().astype(int)
#                         x1, y1, x2, y2 = xyxy_pixel
#                         class_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
#                         txt_line = f"{x1} {y1} {x2} {y2} {conf:.4f} {cls_id} {class_name}\n"
#                         txt_output_lines.append(txt_line)
#                         label = f"{class_name} {conf:.2f}"
#                         cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#                         y1_label = max(y1, label_size[1] + 10)
#                         cv2.rectangle(img_cv, (x1, y1_label - label_size[1] - 10),
#                                       (x1 + label_size[0], y1_label - base_line -10 ), (0, 255, 0), cv2.FILLED)
#                         cv2.putText(img_cv, label, (x1, y1_label - 7),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#                 else:
#                     print(f"  未偵測到符合閾值的物件。")
#                     txt_output_lines.append("# No objects detected or below confidence threshold\n")

#                 txt_filename = output_txt_path / (image_file_path.stem + ".txt")
#                 try:
#                     with open(txt_filename, 'w', encoding='utf-8') as f:
#                         f.writelines(txt_output_lines)
#                 except IOError as e:
#                     print(f"  寫入 bounding box 文字檔案 '{txt_filename}' 失敗: {e}")

#                 img_output_filename = output_img_path / (image_file_path.stem + "_detected" + image_file_path.suffix)
#                 try:
#                     cv2.imwrite(str(img_output_filename), img_cv)
#                 except Exception as e:
#                     print(f"  儲存已偵測圖片 '{img_output_filename}' 失敗: {e}")

#             print("\n所有圖片的個別推論與儲存處理完成。")
#     else:
#         print("\n由於未提供有效的 'test' 圖片資料夾路徑 (來自YAML)，跳過個別圖片的偵測與儲存處理。")


#     # --- 7. 執行模型在測試集上的評估 (計算 precision, recall, mAP 等) ---
#     print(f"\n開始執行模型在測試集上的評估 (使用 YAML 檔案 '{YAML_DATA_PATH}' 中定義的 'test' 資料集)...")
#     print(f"請確保 '{YAML_DATA_PATH}' 中已正確設定 'test' 的路徑，並且該路徑下包含測試用的圖片及對應的標籤檔案，")
#     print(f"否則 precision, recall, mAP 等指標可能無法計算或無意義。")
#     try:
#         # model.val() 使用 YAML 檔案中 'test' 路徑指定的資料集進行評估。
#         # split='test' 表示明確使用測試集。
#         metrics = model.val(data=YAML_DATA_PATH, device=DEVICE, split='test')

#         if metrics:
#             print("\n--- 模型在測試集上的評估結果 ---")
#             # metrics.box 是 DetectionMetrics 物件
#             mAP50_95 = metrics.box.map    # mAP50-95
#             mAP50 = metrics.box.map50  # mAP@.50

#             print(f"  mAP50-95 (mean Average Precision @ IoU=.50-.95): {mAP50_95:.4f}")
#             print(f"  mAP50 (mean Average Precision @ IoU=.50):      {mAP50:.4f}")

#             precision = None
#             recall = None
#             f1_score_val = None

#             # 嘗試從 metrics.results_dict 取得 P, R, F1
#             # 鍵的名稱可能因 ultralytics 版本而異
#             if hasattr(metrics, 'results_dict') and metrics.results_dict:
#                 results_dict = metrics.results_dict
#                 precision_key_B = 'metrics/precision(B)' # (B) 通常指 Bounding Box metrics
#                 recall_key_B = 'metrics/recall(B)'
#                 f1_key_B = 'metrics/f1(B)'
#                 precision_key_simple = 'precision'
#                 recall_key_simple = 'recall'
#                 f1_key_simple = 'f1'


#                 if precision_key_B in results_dict:
#                     precision = results_dict[precision_key_B]
#                 elif precision_key_simple in results_dict:
#                     precision = results_dict[precision_key_simple]

#                 if recall_key_B in results_dict:
#                     recall = results_dict[recall_key_B]
#                 elif recall_key_simple in results_dict:
#                     recall = results_dict[recall_key_simple]

#                 if f1_key_B in results_dict:
#                     f1_score_val = results_dict[f1_key_B]
#                 elif f1_key_simple in results_dict:
#                     f1_score_val = results_dict[f1_key_simple]


#                 if precision is not None:
#                     print(f"  Precision:                                     {precision:.4f}")
#                 else:
#                     print(f"  Precision: (未從 metrics.results_dict 取得)")

#                 if recall is not None:
#                     print(f"  Recall:                                        {recall:.4f}")
#                 else:
#                     print(f"  Recall:    (未從 metrics.results_dict 取得)")

#                 if f1_score_val is not None:
#                      print(f"  F1-score (來自 metrics.results_dict):          {f1_score_val:.4f}")
#                 elif precision is not None and recall is not None and (precision + recall) > 0:
#                     f1_score_calc = 2 * (precision * recall) / (precision + recall)
#                     print(f"  F1-score (由 Precision 和 Recall 計算):        {f1_score_calc:.4f}")
#                 elif precision is not None and recall is not None and (precision + recall) == 0: # P, R 都是0
#                      print(f"  F1-score: (Precision 和 Recall 均為 0，無法計算 F1-score)")
#                 else: # P 或 R 或兩者都未取得
#                     print(f"  F1-score:  (無法計算，Precision 或 Recall 未從 metrics.results_dict 取得)")


#             # 若無法從 results_dict 取得，嘗試從 metrics.box 的屬性取得 (備用方案)
#             elif hasattr(metrics.box, 'mp') and hasattr(metrics.box, 'mr'):
#                 precision = metrics.box.mp # Mean Precision
#                 recall = metrics.box.mr    # Mean Recall
#                 print(f"  Precision (來自 metrics.box.mp):                 {precision:.4f}")
#                 print(f"  Recall (來自 metrics.box.mr):                    {recall:.4f}")
#                 if hasattr(metrics.box, 'f1'):
#                     f1_score_val = metrics.box.f1 # Mean F1-score
#                     print(f"  F1-score (來自 metrics.box.f1):                  {f1_score_val:.4f}")
#                 elif (precision + recall) > 0 :
#                     f1_score_calc = 2 * (precision * recall) / (precision + recall)
#                     print(f"  F1-score (由 P 和 R 計算):                 {f1_score_calc:.4f}")
#                 else: # P, R 都是0
#                      print(f"  F1-score: (Precision 和 Recall 均為 0，無法計算 F1-score)")
#             else:
#                 print("  Precision, Recall, F1-score: (無法從 metrics.results_dict 或 metrics.box 屬性取得)")
#                 print("  YOLO 通常會在控制台輸出這些評估指標的詳細摘要。請檢查控制台輸出。")


#             # 關於 Accuracy (準確率) 的說明:
#             print(f"\n  關於 Accuracy (準確率):")
#             print(f"  在物件偵測任務中, mAP (mean Average Precision) 是衡量模型性能的主要綜合指標。")
#             print(f"  它同時考慮了分類的正確性和定位的精準度。")
#             print(f"  一般分類任務中常見的單一 'Accuracy' 指標，通常不直接適用於評估物件偵測模型的整體性能，")
#             print(f"  因為物件偵測不僅要分類正確，還需要正確定位物件。")
#         else:
#             print("  模型在測試集上的評估未返回任何指標。可能是測試集未包含標籤，或 YAML 設定有誤。")

#     except AttributeError as ae:
#         print(f"執行模型在測試集上的評估時發生屬性錯誤 (可能 metrics 物件結構與預期不同，或 YAML 中 'test' 路徑的標籤設定不完整): {ae}")
#         print(f"  請檢查您的 ultralytics 套件版本、'{YAML_DATA_PATH}' 中的 'test' 設定 (包含標籤) 以及 metrics 物件的可用屬性。")
#         import traceback
#         traceback.print_exc()
#     except FileNotFoundError as fe: # 通常是 YAML 中 test 路徑下的圖片或標籤檔找不到
#         print(f"執行模型在測試集上的評估時發生檔案未找到錯誤 (可能 YAML 中的 'test' 路徑不正確，或該路徑下缺少圖片/標籤檔案): {fe}")
#         import traceback
#         traceback.print_exc()
#     except Exception as e: # 其他所有未預期的錯誤
#         print(f"執行模型在測試集上的評估時發生未預期的錯誤: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == '__main__':
#     run_detection()