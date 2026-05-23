import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        self.val_ned = []
        self.epochs = []
        self.current_train_loss = 0.0

    # ------------------------------------------------------------------
    # 1. Data Augmentation Demo (chỉ chạy 1 lần khi bắt đầu train)
    # ------------------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        print("\n[Visualizer] Đang kết xuất ảnh Data Augmentation và Data Gốc...")
        from data import batch_preparation_img2seq
        import torch.utils.data

        temp_loader = torch.utils.data.DataLoader(
            trainer.datamodule.train_set,
            batch_size=4,
            num_workers=0,
            collate_fn=batch_preparation_img2seq,
        )
        batch = next(iter(temp_loader))
        x, _, y = batch  # x shape: (B, C, H, W)

        # Lấy ảnh đầu tiên để minh họa Augmentation
        img_tensor_0 = x[0]
        original_img = transforms.ToPILImage()(img_tensor_0)

        # 1. Elastic Distortion
        elastic_transform = transforms.RandomApply(
            [ElasticDistortion(grid=(2, 2), magnitude=(10, 10), min_sep=(1, 1))], p=1.0
        )
        img_elastic = elastic_transform(original_img.copy())

        # 2. Random Perspective
        perspective_transform = transforms.RandomPerspective(
            distortion_scale=0.3, p=1.0, interpolation=Image.BILINEAR, fill=255
        )
        img_perspective = perspective_transform(original_img.copy())

        # 3. Gaussian Noise
        img_np = np.array(original_img)
        noise = np.random.normal(0, 15, img_np.shape).astype(np.float32)
        img_noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img_noise_pil = Image.fromarray(img_noisy)

        # Vẽ 4 ô lưới Augmentation
        fig, axes = plt.subplots(4, 1, figsize=(10, 16))
        axes[0].imshow(original_img, cmap="gray")
        axes[0].set_title("1. Original Image (Ảnh gốc)")
        axes[0].axis("off")
        axes[1].imshow(img_elastic, cmap="gray")
        axes[1].set_title("2. Elastic Distortion (Giấy cong rách/nhăn)")
        axes[1].axis("off")
        axes[2].imshow(img_perspective, cmap="gray")
        axes[2].set_title("3. Random Perspective (Góc chụp nghiêng)")
        axes[2].axis("off")
        axes[3].imshow(img_noise_pil, cmap="gray")
        axes[3].set_title("4. Gaussian Noise (Nhiễu hạt)")
        axes[3].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "data_augmentation_demo.png"), dpi=150)
        plt.close()

        # Vẽ vài file data gốc kèm label
        num_samples = min(4, x.size(0))
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]
        for i in range(num_samples):
            img_show = x[i, 0].numpy()
            gt_tokens = [
                pl_module.model.i2w[token.item()]
                for token in y[i]
                if token.item() in pl_module.model.i2w
            ]
            gt_str = " ".join(gt_tokens).replace("<t>", " ").replace("<b>", " | ").replace("<s>", " ")
            if len(gt_str) > 80:
                gt_str = gt_str[:80] + "..."
            axes[i].imshow(img_show, cmap="gray")
            axes[i].set_title(f"Data Gốc {i+1}\nLabel: {gt_str}", fontsize=9)
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "original_data_samples.png"), dpi=150)
        plt.close()
        print(f"[Visualizer] Đã lưu ảnh minh họa vào thư mục '{self.output_dir}/'")

    # ------------------------------------------------------------------
    # 2. Ghi nhận train loss mỗi batch
    # ------------------------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
        self.current_train_loss = loss

    # ------------------------------------------------------------------
    # 3. Biểu đồ metrics mỗi validation epoch
    # ------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        self.epochs.append(epoch)

        t_loss = trainer.callback_metrics.get("loss", self.current_train_loss)
        if isinstance(t_loss, torch.Tensor):
            t_loss = t_loss.item()
        self.train_losses.append(t_loss)

        val_ser = trainer.callback_metrics.get("val_SER", 0)
        val_ler = trainer.callback_metrics.get("val_LER", 0)
        if isinstance(val_ser, torch.Tensor):
            val_ser = val_ser.item()
        if isinstance(val_ler, torch.Tensor):
            val_ler = val_ler.item()
        self.val_ser.append(val_ser)
        self.val_ned.append(val_ler)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(self.epochs, self.train_losses, "b-o", label="Train Loss")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(self.epochs, self.val_ser, "r-s", label="Val SER")
        ax2.plot(self.epochs, self.val_ned, "g-^", label="Val NED (LER)")
        ax2.set_title("Validation Error Rates")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Error Rate (%)")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_val_metrics.png"), dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # 4. Test report + 1 Attention Map (chạy sau khi train xong)
    # ------------------------------------------------------------------
    def on_test_epoch_end(self, trainer, pl_module):
        print("\n" + "=" * 50)
        print("[Visualizer] BÁO CÁO KẾT QUẢ TEST CUỐI CÙNG")

        test_ser = trainer.callback_metrics.get("test_SER", 0)
        test_ler = trainer.callback_metrics.get("test_LER", 0)
        if isinstance(test_ser, torch.Tensor):
            test_ser = test_ser.item()
        if isinstance(test_ler, torch.Tensor):
            test_ler = test_ler.item()

        print(f"-> Test SER : {test_ser:.4f}")
        print(f"-> Test NED (LER): {test_ler:.4f}")
        print("=" * 50 + "\n")

        # --- Trích xuất 1 Attention Map ---
        print("[Visualizer] Đang trích xuất Attention Map ví dụ...")
        try:
            self._generate_attention_map(trainer, pl_module)
        except Exception as e:
            print(f"[Visualizer] Không thể tạo Attention Map: {e}")

    # ------------------------------------------------------------------
    # Helper: tạo 1 attention map đúng cách
    # ------------------------------------------------------------------
    def _generate_attention_map(self, trainer, pl_module):
        import re
        from data import batch_preparation_img2seq
        import torch.utils.data

        # ── Kern pitch token detector ──────────────────────────────────
        # Token kern dạng: "4c#", "8.E-J", "2AA", "16bb-LJJ" v.v.
        # Pitch letter: A-G (upper=thấp, lower=cao); không phải rest/meta
        PITCH_RE = re.compile(r'^[\d.]*[A-Ga-g][A-Ga-g#\-]*')

        def is_pitch_token(s: str) -> bool:
            s = s.strip()
            if not s or s[0] in ('*', '=', '!', '<'):
                return False
            if re.match(r'^[\d.]*r', s):   # rest token
                return False
            return bool(PITCH_RE.match(s))

        # ── Load 1 mẫu từ test set ────────────────────────────────────
        temp_loader = torch.utils.data.DataLoader(
            trainer.datamodule.test_set,
            batch_size=1,
            num_workers=0,
            collate_fn=batch_preparation_img2seq,
        )
        x, _, _ = next(iter(temp_loader))
        img_tensor = x[0:1].to(pl_module.device)       # (1, 1, H, W)
        img_np = img_tensor[0, 0].cpu().float().numpy() # (H, W)

        # ── Inference với cross-attention ─────────────────────────────
        # predict() trả về (list[token_id], SMTOutput)
        # SMTOutput.cross_attentions = list[Tensor(1, heads, seq, enc_flat)]
        # phần tử cuối cùng trong vòng lặp chứa attention của TOÀN bộ seq
        pl_module.eval()
        with torch.no_grad():
            token_ids, final_output = pl_module.model.predict(
                img_tensor, convert_to_str=False, return_weights=True
            )

        # final_output là SMTOutput — truy cập qua attribute, không phải dict
        cross_attns = final_output.cross_attentions   # list[Tensor] hoặc None
        if not cross_attns or len(cross_attns) == 0:
            print("[Visualizer] Model không trả về cross_attentions — bỏ qua.")
            return

        # ── Tìm pitch token đầu tiên trong sequence ───────────────────
        pitch_idx, pitch_str = None, None
        last_cross = cross_attns[-1]            # layer sâu nhất: (1, H, T, E)
        seq_len = last_cross.shape[2]

        for i, tid in enumerate(token_ids):
            if i >= seq_len:
                break
            tok = pl_module.model.i2w.get(tid, '')
            if is_pitch_token(tok):
                pitch_idx, pitch_str = i, tok
                break

        if pitch_idx is None:                   # fallback: token đầu tiên hợp lệ
            pitch_idx = min(seq_len // 3, seq_len - 1)
            pitch_str = pl_module.model.i2w.get(
                token_ids[pitch_idx] if pitch_idx < len(token_ids) else 0, '?'
            )
            print(f"[Visualizer] Không tìm được pitch token → fallback idx={pitch_idx} '{pitch_str}'")
        else:
            print(f"[Visualizer] Pitch token: idx={pitch_idx}, token='{pitch_str}'")

        # ── Trích xuất attention vector → 2D map ──────────────────────
        # Trung bình qua tất cả heads: (enc_flat,)
        attn_vec = last_cross[0, :, pitch_idx, :].mean(dim=0).cpu().float().numpy()

        # ConvNext 3 stages → stride 16 mỗi chiều
        stride = pl_module.model.height_reduction   # == width_reduction == 16
        feat_h = img_tensor.shape[2] // stride
        feat_w = img_tensor.shape[3] // stride

        if attn_vec.shape[0] == feat_h * feat_w:
            attn_2d = attn_vec.reshape(feat_h, feat_w)
        else:
            # Fallback: phân tích nhân tử gần đúng
            enc_flat = attn_vec.shape[0]
            h = max(h for h in range(1, int(enc_flat**0.5) + 1) if enc_flat % h == 0)
            attn_2d = attn_vec.reshape(h, enc_flat // h)
            print(f"[Visualizer] enc_flat={enc_flat} ≠ {feat_h*feat_w}, reshape {h}×{enc_flat//h}")

        # Resize về kích thước ảnh + normalize [0,1]
        img_h, img_w = img_np.shape
        attn_up = cv2.resize(attn_2d, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        lo, hi = attn_up.min(), attn_up.max()
        attn_norm = (attn_up - lo) / (hi - lo + 1e-8)

        # ── Vẽ kết quả ────────────────────────────────────────────────
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

        ax_l.imshow(img_np, cmap='gray', aspect='auto')
        ax_l.set_title('Ảnh bản nhạc gốc', fontsize=12, fontweight='bold')
        ax_l.axis('off')

        ax_r.imshow(img_np, cmap='gray', aspect='auto')
        heat = ax_r.imshow(attn_norm, cmap='jet', alpha=0.55,
                           vmin=0.0, vmax=1.0, aspect='auto')
        ax_r.set_title(
            f'Cross-Attention Map\nDecoder "nhìn" vào đâu khi sinh token: "{pitch_str}"',
            fontsize=11, fontweight='bold',
        )
        ax_r.axis('off')

        cbar = plt.colorbar(heat, ax=ax_r, fraction=0.035, pad=0.02)
        cbar.set_label('Attention Weight', fontsize=9)

        n_layers = len(cross_attns)
        n_heads = last_cross.shape[1]
        fig.suptitle(
            f'Sheet Music Transformer — Attention Map\n'
            f'(Layer {n_layers}/{n_layers}, avg {n_heads} heads)',
            fontsize=12, fontweight='bold',
        )

        save_path = os.path.join(self.output_dir, 'example_attention_map.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualizer] Đã lưu Attention Map tại '{save_path}'")
