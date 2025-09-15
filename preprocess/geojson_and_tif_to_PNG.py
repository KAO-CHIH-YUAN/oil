import os
import argparse
import rasterio
import geopandas as gpd
import numpy as np
from PIL import Image
from tqdm import tqdm
from rasterio.features import rasterize
from shapely.geometry import mapping
import cv2

# --- 使用者設定區塊 ---
# 設定 True 表示要生成 TIF 的 PNG 視覺化圖片
# 設定 False 表示要跳過此步驟，只生成 labels
GENERATE_IMAGE_PNG = True
# ------------------


def process_geospatial_data(tif_path, geojson_path, output_dir, generate_image_png=True, target_class_index=0):
    """
    處理單一的 TIF 和 GeoJSON/JSON 檔案，生成 YOLO 的 TXT、視覺化的 PNG 以及黑白的遮罩 PNG。
    """
    try:
        # 1. 讀取 TIF 影像和其地理資訊
        with rasterio.open(tif_path) as src:
            tif_crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height
            
            image_array = None
            if generate_image_png:
                if src.count == 1:
                    band = src.read(1)
                    image_array = np.stack([band, band, band], axis=-1)
                else:
                    image_array = src.read([1, 2, 3])
                    image_array = np.moveaxis(image_array, 0, -1)

        # 2. 讀取 GeoJSON/JSON 並統一座標系統 (CRS)
        gdf = gpd.read_file(geojson_path)
        
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
            
        if gdf.crs != tif_crs:
            gdf = gdf.to_crs(tif_crs)

        # 3. 將多邊形「畫」到陣列上，生成遮罩
        mask = np.zeros((height, width), dtype=np.uint8)
        if not gdf.empty:
            shapes = [mapping(geom) for geom in gdf.geometry]
            mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, default_value=1, dtype=np.uint8)

        # 4. 從遮罩生成 YOLO 格式的 TXT 標籤
        yolo_lines = []
        if mask.any():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if contour.size > 4:
                    contour = contour.squeeze()
                    if contour.ndim == 1:
                       contour = contour.reshape(1, -1)
                    
                    normalized_coords = [f"{coord / dim:.6f}" for point in contour for coord, dim in zip(point, (width, height))]
                    line = f"{target_class_index} " + " ".join(normalized_coords)
                    yolo_lines.append(line)

        # 5. 準備輸出路徑
        base_filename = os.path.splitext(os.path.basename(tif_path))[0]
        images_output_folder = os.path.join(output_dir, 'images')
        labels_output_folder = os.path.join(output_dir, 'labels')
        
        os.makedirs(images_output_folder, exist_ok=True)
        os.makedirs(labels_output_folder, exist_ok=True)
        
        png_output_path = os.path.join(images_output_folder, f"{base_filename}.png")
        txt_output_path = os.path.join(labels_output_folder, f"{base_filename}.txt")
        mask_output_path = os.path.join(labels_output_folder, f"{base_filename}.png")

        # 6. 儲存所有檔案
        if generate_image_png and image_array is not None:
            img = Image.fromarray(image_array)
            img.save(png_output_path)
        
        mask_image_array = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_image_array, mode='L')
        mask_image.save(mask_output_path)
        
        if yolo_lines:
            with open(txt_output_path, 'w') as f:
                f.write("\n".join(yolo_lines))
        else:
            open(txt_output_path, 'w').close()
            
        return True

    except Exception as e:
        print(f"處理檔案 {os.path.basename(tif_path)} 時發生錯誤: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="將 GeoTIFF 和 GeoJSON/JSON 轉換為 YOLO Segmentation 格式的數據集。")
    
    # 維持使用命令列參數來設定路徑，方便彈性使用
    parser.add_argument('--input_dir', type=str, default='/home/yuan/OIL_PROJECT/dataset/dataset_SAR_3/S1_tif',
                        help='包含 TIF 和 GeoJSON/JSON 檔案的輸入資料夾路徑。')
    parser.add_argument('--output_dir', type=str, default='/home/yuan/OIL_PROJECT/dataset/dataset_SAR_3/S1_PNG_output',
                        help='儲存 images 和 labels 資料夾的輸出路徑。')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"錯誤：輸入資料夾 '{input_dir}' 不存在。")
        return

    tif_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))]
    
    if not tif_files:
        print(f"在 '{input_dir}' 中找不到任何 TIF 檔案。")
        return

    print(f"找到 {len(tif_files)} 個 TIF 檔案，將使用以下設定進行處理：")
    print(f"  - 輸入資料夾: {input_dir}")
    print(f"  - 輸出資料夾: {output_dir}")
    # --- 修改點：直接讀取檔案上方的設定變數 ---
    print(f"  - 是否生成視覺化 PNG 圖片: {GENERATE_IMAGE_PNG}")
    
    success_count = 0
    for tif_file in tqdm(tif_files, desc="整體進度"):
        base_name = os.path.splitext(tif_file)[0]
        tif_path = os.path.join(args.input_dir, tif_file)

        possible_json_exts = ['.json', '.geo.json', '.geojson']
        json_path = None
        for ext in possible_json_exts:
            path_to_check = os.path.join(args.input_dir, f"{base_name}{ext}")
            if os.path.exists(path_to_check):
                json_path = path_to_check
                break

        if json_path is None:
            print(f"警告：找不到 {tif_file} 對應的 .json, .geo.json 或 .geojson 檔案，跳過。")
            continue
        
        # --- 修改點：將設定變數傳入處理函式 ---
        if process_geospatial_data(tif_path, json_path, output_dir, generate_image_png=GENERATE_IMAGE_PNG):
            success_count += 1
            
    print(f"\n處理完成！成功轉換 {success_count} / {len(tif_files)} 組檔案。")
    print(f"輸出檔案已儲存至 '{output_dir}' 資料夾中。")

if __name__ == '__main__':
    main()