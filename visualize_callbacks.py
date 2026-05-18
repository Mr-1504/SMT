import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import lightning.pytorch as L

from torchvision import transforms
from data_augmentation.transforms_custom import ElasticDistortion

class SMTVisualizerCallback(L.Callback):
    def __init__(self, output_dir="visualizations"):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.val_ser = []
        self.val_ned = [] # Chúng ta sẽ dùng LER/CER hoặc SER như NED
        self.epochs = []
        self.current_train_loss = 0.0

    def on_fit_start(self, trainer, pl_module):
        """
        Thực hiện ngay đầu quá trình train:
        1. Tạo ảnh minh họa Data Augmentation (4 ô lưới)
        2. Tạo ảnh minh họa vài mẫu dữ liệu gốc
        """
        print("\n[Visualizer] Đang kết xuất ảnh Data Augmentation và Data Gốc...")
        # trainer.train_dataloader chưa được khởi tạo tại on_fit_start.
        # Tạo DataLoader tạm thời với num_workers=0 chỉ để lấy 1 batch cho visualization.
        from data import batch_preparation_img2seq
        import torch.utils.data
        temp_loader = torch.utils.data.DataLoader(
            trainer.datamodule.train_set,
            batch_size=4,
            num_workers=0,
            collate_fn=batch_preparation_img2seq,
        )
        batch = next(iter(temp_loader))
        x, _, y = batch # x shape: (B, C, H, W)
        
        # Lấy ảnh đầu tiên để minh họa Augmentation
        img_tensor_0 = x[0]
        original_img = transforms.ToPILImage()(img_tensor_0)
        
        # 1. Original
        # 2. Elastic Distortion
        elastic_transform = transforms.RandomApply([ElasticDistortion(grid=(2, 2), magnitude=(10, 10), min_sep=(1, 1))], p=1.0)
        img_elastic = elastic_transform(original_img.copy())
        
        # 3. Random Perspective
        perspective_transform = transforms.RandomPerspective(distortion_scale=0.3, p=1.0, interpolation=Image.BILINEAR, fill=255)
        img_perspective = perspective_transform(original_img.copy())
        
        # 4. Gaussian Noise
        img_np = np.array(original_img)
        noise = np.random.normal(0, 15, img_np.shape).astype(np.float32)
        img_noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img_noise_pil = Image.fromarray(img_noisy)
        
        # Vẽ 4 ô lưới Augmentation
        fig, axes = plt.subplots(4, 1, figsize=(10, 16))
        axes[0].imshow(original_img, cmap='gray'); axes[0].set_title("1. Original Image (Ảnh gốc)"); axes[0].axis('off')
        axes[1].imshow(img_elastic, cmap='gray'); axes[1].set_title("2. Elastic Distortion (Giấy cong rách/nhăn)"); axes[1].axis('off')
        axes[2].imshow(img_perspective, cmap='gray'); axes[2].set_title("3. Random Perspective (Góc chụp nghiêng)"); axes[2].axis('off')
        axes[3].imshow(img_noise_pil, cmap='gray'); axes[3].set_title("4. Gaussian Noise (Nhiễu hạt)"); axes[3].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "data_augmentation_demo.png"))
        plt.close()
        
        # Vẽ vài file data gốc
        num_samples = min(4, x.size(0))
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
        if num_samples == 1: axes = [axes]
        for i in range(num_samples):
            img_np = x[i, 0].numpy()
            gt_sequence = y[i]
            gt = "".join([pl_module.model.i2w[token.item()] for token in gt_sequence[:-1] if token.item() in pl_module.model.i2w])
            gt = gt.replace("<t>", " ").replace("<b>", " | ").replace("<s>", " ")
            if len(gt) > 80: gt = gt[:80] + "..."
            axes[i].imshow(img_np, cmap='gray')
            axes[i].set_title(f"Data Gốc {i+1}\nLabel: {gt}", fontsize=10)
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "original_data_samples.png"))
        plt.close()
        print(f"[Visualizer] Đã lưu ảnh minh họa vào thư mục {self.output_dir}/")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Lấy loss của batch hiện tại để tính trung bình epoch nếu cần
        loss = outputs['loss'].item() if isinstance(outputs, dict) else outputs.item()
        self.current_train_loss = loss

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        self.epochs.append(epoch)
        
        # Train Loss (lấy xấp xỉ từ batch cuối hoặc callback_metrics)
        t_loss = trainer.callback_metrics.get('loss', self.current_train_loss)
        if isinstance(t_loss, torch.Tensor): t_loss = t_loss.item()
        self.train_losses.append(t_loss)
        
        # Val Loss (SMT không trả val_loss mặc định, ta dùng SER/LER)
        # SMT tính CER, SER, LER. NED (Normalized Edit Distance) thường tương đương với SER (Symbol Error Rate)
        val_ser = trainer.callback_metrics.get('val_SER', 0)
        val_ler = trainer.callback_metrics.get('val_LER', 0)
        
        if isinstance(val_ser, torch.Tensor): val_ser = val_ser.item()
        if isinstance(val_ler, torch.Tensor): val_ler = val_ler.item()
            
        self.val_ser.append(val_ser)
        self.val_ned.append(val_ler) # LER (Line Error Rate) đóng vai trò tham chiếu thêm
        
        # Vẽ biểu đồ Train/Val Loss (Ta vẽ Train Loss) và Val SER/NED
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(self.epochs, self.train_losses, 'b-', marker='o', label='Train Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(self.epochs, self.val_ser, 'r-', marker='s', label='Val SER')
        ax2.plot(self.epochs, self.val_ned, 'g-', marker='^', label='Val NED (LER)')
        ax2.set_title('Validation Error Rates')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Error Rate (%)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_val_metrics.png"))
        plt.close()

    def on_test_epoch_end(self, trainer, pl_module):
        """
        Thực hiện ở cuối quá trình Test (sau khi train xong):
        1. Báo cáo test SER / NED cuối cùng.
        2. Vẽ 1 Attention Map làm ví dụ.
        """
        print("\n" + "="*50)
        print("[Visualizer] BÁO CÁO KẾT QUẢ TEST CUỐI CÙNG")
        
        test_ser = trainer.callback_metrics.get('test_SER', 0)
        test_ler = trainer.callback_metrics.get('test_LER', 0)
        if isinstance(test_ser, torch.Tensor): test_ser = test_ser.item()
        if isinstance(test_ler, torch.Tensor): test_ler = test_ler.item()
            
        print(f"-> Test SER: {test_ser:.2f} %")
        print(f"-> Test NED (LER): {test_ler:.2f} %")
        print("="*50 + "\n")
        
        # 2. Extract 1 Attention Map từ Test Dataloader
        print("[Visualizer] Đang trích xuất 1 Attention Map làm ví dụ...")
        test_loader = trainer.datamodule.test_dataloader()
        batch = next(iter(test_loader))
        x, _, _ = batch
        
        img_tensor = x[0:1].to(pl_module.device)
        img_np = img_tensor[0, 0].cpu().numpy()
        
        pl_module.eval()
        with torch.no_grad():
            predictions, output = pl_module.model.predict(img_tensor, convert_to_str=True, return_weights=True)
            
            if len(predictions) > 0 and output.cross_attentions is not None:
                token_idx = len(predictions) // 2
                target_token = predictions[token_idx]
                
                last_layer_cross_attn = output.cross_attentions[-1]
                attn_weights = last_layer_cross_attn[0, :, token_idx, :].mean(dim=0)
                
                encoder_features = pl_module.model.forward_encoder(img_tensor)
                _, _, feat_h, feat_w = encoder_features.shape
                
                attn_map = attn_weights.view(feat_h, feat_w).cpu().numpy()
                img_h, img_w = img_np.shape[:2]
                attn_map_resized = cv2.resize(attn_map, (img_w, img_h))
                attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min() + 1e-8)
                
                plt.figure(figsize=(12, 6))
                plt.imshow(img_np, cmap='gray')
                plt.imshow(attn_map_resized, cmap='jet', alpha=0.5)
                plt.title(f"Ví dụ Attention Map cho token: '{target_token}'", fontsize=14)
                plt.axis('off')
                
                plt.savefig(os.path.join(self.output_dir, "example_attention_map_test.png"))
                plt.close()
                print(f"[Visualizer] Đã lưu ví dụ Attention Map tại {self.output_dir}/example_attention_map_test.png")
