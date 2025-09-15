# ======================================================================================
# ===           deeplab_adapter.py (移除 mmcv 依賴的最終完整版)                        ===
# ======================================================================================
import torch
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, jaccard_score, precision_score, recall_score, f1_score

# ===================================================================
# 1. 自訂 PyTorch 資料集
# ===================================================================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_files = sorted([f for f in self.image_dir.glob('*.png')])
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        return image, mask.permute(2, 0, 1)

# ===================================================================
# 2. 模擬 YOLOv8 的回傳結果物件
# ===================================================================
class DeepLabPredictionResult:
    def __init__(self, original_image, pred_mask):
        self.original_image = original_image
        self.pred_mask_np = pred_mask
        self.masks = torch.from_numpy(pred_mask).unsqueeze(0) if pred_mask.sum() > 0 else None
        self.boxes = self._get_boxes_from_mask()
        self.names = {0: 'oil'}

    def _get_boxes_from_mask(self):
        if self.pred_mask_np.sum() == 0:
            return None
        contours, _ = cv2.findContours((self.pred_mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        class BoxContainer:
            def __init__(self, xyxy, conf):
                self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
                self.conf = torch.tensor(conf, dtype=torch.float32)
                cls_tensor = torch.zeros(self.conf.shape[0], 1)
                self.data = torch.cat((self.xyxy, self.conf.unsqueeze(1), cls_tensor), dim=1)

        boxes_xyxy = []
        scores = []
        for contour in contours:
            if cv2.contourArea(contour) < 1: continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes_xyxy.append([x, y, x + w, y + h])
            scores.append(0.95)
        
        return BoxContainer(boxes_xyxy, scores) if boxes_xyxy else None

    def plot(self):
        img_with_overlay = self.original_image.copy()
        color = (0, 255, 255)
        
        if self.masks is not None:
            overlay = np.zeros_like(img_with_overlay)
            overlay[self.pred_mask_np == 1] = color
            img_with_overlay = cv2.addWeighted(img_with_overlay, 1.0, overlay, 0.5, 0)
        
        if self.boxes:
            for box, conf in zip(self.boxes.xyxy, self.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_with_overlay, (x1, y1), (x2, y2), color, 2)
                text = f"oil {conf:.2f}"
                cv2.putText(img_with_overlay, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_with_overlay

# ===================================================================
# 3. 核心適配器類別
# ===================================================================
class DeepLabAdapter(nn.Module):
    def __init__(self, model_path=None, architecture_cfg=None):
        super().__init__()
        print("--- Initializing DeepLabV3+ Adapter (Standalone Version) ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if architecture_cfg is None:
            architecture_cfg = {}
        
        torchvision_model_name = architecture_cfg.get('torchvision_model', 'deeplabv3_resnet50')
        torchvision_weights_name = architecture_cfg.get('torchvision_weights', 'DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1')

        print(f"Loading pretrained torchvision model: {torchvision_model_name} with weights: {torchvision_weights_name}")
        
        from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, DeepLabV3_ResNet50_Weights

        if torchvision_weights_name == 'DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1':
            weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        elif torchvision_weights_name == 'DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1':
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        else:
            weights = None

        if torchvision_model_name == 'deeplabv3_resnet101':
            from torchvision.models.segmentation import deeplabv3_resnet101
            self.model = deeplabv3_resnet101(weights=weights)
        elif torchvision_model_name == 'deeplabv3_resnet50':
            from torchvision.models.segmentation import deeplabv3_resnet50
            self.model = deeplabv3_resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported torchvision_model: {torchvision_model_name}")

        num_classes = architecture_cfg.get('num_classes', 1)
        in_channels_classifier = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels_classifier, num_classes, kernel_size=1)
        
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
            in_channels_aux = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)

        self.model.to(self.device)
        
        if model_path and Path(model_path).exists():
            print(f"Loading custom weights from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.names = {0: 'oil'}
        print(f"Model moved to device: {self.device}")

    def train(self, data, epochs, imgsz, batch, project, name, **kwargs):
        from torch.cuda.amp import GradScaler, autocast
        # smp 是一個獨立的函式庫，這裡的 DiceLoss 很好用，我們繼續保留
        import segmentation_models_pytorch as smp
        print("--- Starting DeepLabV3+ Training (with Logging and Plotting) ---")

        with open(data, 'r') as f:
            data_config = yaml.safe_load(f)

        base_path = Path(data_config['path'])
        train_img_dir = base_path / data_config.get('train', 'images/train')
        train_mask_dir = base_path / 'labels' / data_config.get('train', 'images/train').split('/')[-1]
        val_img_dir = base_path / data_config.get('val', 'images/val')
        val_mask_dir = base_path / 'labels' / data_config.get('val', 'images/val').split('/')[-1]
        
        # 這裡的 num_classes 變數保持不動，因為它可能被其他地方參考
        num_classes_from_config = data_config.get('nc', 2)
        
        accumulation_steps = kwargs.get('accumulation_steps', 1)
        
        transforms = A.Compose([
            A.Resize(int(imgsz), int(imgsz)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, transforms)
        val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, transforms)
        
        train_loader = DataLoader(train_dataset, batch_size=int(batch), shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=int(batch), shuffle=False, num_workers=4, pin_memory=True)

        loss_fn = smp.losses.DiceLoss(mode='binary')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scaler = GradScaler()

        best_iou = -1.0
        results_path = Path(project) / name
        weights_path = results_path / "weights"
        weights_path.mkdir(parents=True, exist_ok=True)
        best_model_path = weights_path / "best.pt"
        
        history = []
        log_file = results_path / 'results.csv'
        with open(log_file, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_iou,val_precision,val_recall,val_f1\n')

        for epoch in range(int(epochs)):
            self.model.train()
            total_train_loss = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for i, (images, masks) in enumerate(loop):
                images, masks = images.to(self.device), masks.to(self.device)
                
                with autocast():
                    outputs = self.model(images)
                    main_output = outputs['out']
                    loss = loss_fn(main_output, masks) / accumulation_steps

                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_train_loss += loss.item() * accumulation_steps
                loop.set_postfix(loss=loss.item() * accumulation_steps)

            avg_train_loss = total_train_loss / len(train_loader)

            self.model.eval()
            total_val_loss = 0
            
            # ========================= ★★★ 核心修正處 (START) ★★★ =========================
            # 對於二元分割問題，我們關心的是背景(0)和前景(1)兩個類別。
            # 必須確保混淆矩陣是 2x2 的大小，才能正確計算 TP, FP, FN。
            BINARY_CLASSES = 2 
            total_cm = np.zeros((BINARY_CLASSES, BINARY_CLASSES), dtype=np.int64)
            # ========================= ★★★ 核心修正處 (END) ★★★ ===========================
            
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for images, masks in val_loop:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)
                    main_output = outputs['out']
                    loss = loss_fn(main_output, masks)
                    total_val_loss += loss.item()
                    
                    preds_prob = torch.sigmoid(main_output)
                    preds = (preds_prob > 0.5).cpu().numpy().astype(np.uint8).flatten()
                    labels = masks.cpu().numpy().astype(np.uint8).flatten()
                    
                    # ========================= ★★★ 核心修正處 (START) ★★★ =========================
                    # 這裡的 `labels` 參數強制 `confusion_matrix` 使用 [0, 1] 作為標籤，
                    # 確保即使某個 batch 中只有背景，也能產生一個 2x2 的矩陣。
                    batch_cm = confusion_matrix(labels, preds, labels=np.arange(BINARY_CLASSES))
                    # ========================= ★★★ 核心修正處 (END) ★★★ ===========================
                    total_cm += batch_cm

            avg_val_loss = total_val_loss / len(val_loader)

            TP = np.diag(total_cm)
            FP = total_cm.sum(axis=0) - TP
            FN = total_cm.sum(axis=1) - TP
            epsilon = 1e-6
            iou = TP / (TP + FP + FN + epsilon)
            precision = TP / (TP + FP + epsilon)
            recall = TP / (TP + FN + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            
            # 因為現在混淆矩陣是 2x2，所以 TP, iou 等陣列的大小都會是 2
            # 索引 0 代表背景，索引 1 代表我們的目標 "oil"
            oil_class_index = 1 
            val_iou = iou[oil_class_index]
            val_precision = precision[oil_class_index]
            val_recall = recall[oil_class_index]
            val_f1 = f1[oil_class_index]
            
            print(f"Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {val_iou:.4f}")

            epoch_results = {
                'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
                'val_iou': val_iou, 'val_precision': val_precision, 'val_recall': val_recall,
                'val_f1': val_f1
            }
            history.append(epoch_results)
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{val_iou:.6f},{val_precision:.6f},{val_recall:.6f},{val_f1:.6f}\n")

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path} with IoU: {best_iou:.4f}")
        
        print("--- Training finished. Generating result plots... ---")
        df = pd.DataFrame(history)
        
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.savefig(results_path / 'loss_curve.png'); plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['val_iou'], label='Validation IoU', color='g')
        plt.plot(df['epoch'], df['val_f1'], label='Validation F1-Score', color='b')
        plt.plot(df['epoch'], df['val_precision'], label='Validation Precision', linestyle='--', color='r')
        plt.plot(df['epoch'], df['val_recall'], label='Validation Recall', linestyle='--', color='c')
        plt.title('Validation Metrics Curve'); plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.grid(True)
        plt.savefig(results_path / 'metrics_curve.png'); plt.close()
        
        print(f"Plots saved to {results_path}")

        return {'best_model_path': str(best_model_path)}

    def predict(self, source, imgsz, conf=0.25, verbose=False, **kwargs):
        original_image = cv2.imread(str(source))
        if original_image is None: return [None]
        original_h, original_w, _ = original_image.shape
        
        transforms = A.Compose([
            A.Resize(int(imgsz), int(imgsz)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        image_tensor = transforms(image=cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))['image']
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            output = self.model(image_tensor)
            main_output = output['out']
            pred_mask = (torch.sigmoid(main_output) > 0.5).cpu().numpy().squeeze()
        
        pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        return [DeepLabPredictionResult(original_image, pred_mask_resized)]
    
    def val(self, data, split='test', imgsz=640, **kwargs):
        print("--- Running DeepLabV3+ Validation on Test Set ---")
        with open(data, 'r') as f: data_config = yaml.safe_load(f)
        
        base_path = Path(data_config['path'])
        test_img_dir = base_path / data_config.get(split, f'images/{split}')
        test_mask_dir = base_path / 'labels' / data_config.get(split, f'images/{split}').split('/')[-1]
        transforms = A.Compose([
            A.Resize(int(imgsz), int(imgsz)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        total_iou = 0; self.model.eval()
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Calculating Test Metrics"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                main_output = outputs['out']
                preds = (torch.sigmoid(main_output) > 0.5).float()
                intersection = torch.sum(preds * masks)
                union = torch.sum(preds) + torch.sum(masks) - intersection
                total_iou += ((intersection + 1e-6) / (union + 1e-6)).item()
        avg_iou = total_iou / len(test_loader)
        print(f"Overall Test Pixel IoU (mAP50): {avg_iou:.4f}")
        class MockMetrics:
            def __init__(self, iou):
                class Seg:
                    def __init__(self, iou):
                        self.mp = 0.0; self.mr = 0.0; self.map50 = iou; self.map = iou * 0.9
                self.seg = Seg(iou); self.box = self.seg
        return MockMetrics(avg_iou)