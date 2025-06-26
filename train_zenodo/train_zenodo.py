from ultralytics import YOLO
import os

if __name__ == '__main__': 

    # # 載入模型
    # model = YOLO('yolo11x-seg.pt') # yolo11n-seg.pt  yolo11m-seg.pt  yolo11x-seg.pt 
    # print("開始訓練...")
    # results = model.train(data='data.yaml', 
    #                       name = 'yolo11x-seg-zenodo-500epoch', 
    #                       epochs=500, 
    #                       imgsz=1024,
    #                       batch=6, # 根據您的 GPU 記憶體調整
    #                       )
    # print("訓練結束")
    # print("模型儲存在:", results.save_dir)



    # # 載入模型
    # model = YOLO('yolo11m-seg.pt') # yolo11n-seg.pt  yolo11m-seg.pt  yolo11x-seg.pt 
    # print("開始訓練...")
    # results = model.train(data='data.yaml', 
    #                       name = 'yolo11m-seg-zenodo', 
    #                       epochs=500, 
    #                       imgsz=1024,
    #                       batch=6, # 根據您的 GPU 記憶體調整
    #                       )
    # print("訓練結束")
    # print("模型儲存在:", results.save_dir)



    # 載入模型
    model = YOLO('yolo11n.pt') # yolo11n-seg.pt  yolo11m-seg.pt  yolo11x-seg.pt 
    print("開始訓練...")
    results = model.train(data='data.yaml', 
                          name = r'E:\yuan_oil\yolo_segmentation_project\runs\segment\zenodo\yolo(11n-2000epoch)_datset(zenodo_onlyVV_only_oil)', 
                          epochs=2000, 
                          imgsz=1024,
                          batch=16, # 根據您的 GPU 記憶體調整
                          )
    print("訓練結束")
    print("模型儲存在:", results.save_dir)

    

    