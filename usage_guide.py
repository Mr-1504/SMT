import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import datasets
from smt_model import SMTModelForCausalLM
from data_augmentation.data_augmentation import convert_img_to_tensor

def plot_samples(dataset_name="antoniorv6/camera_grandstaff", num_samples=4):
    print(f"Downloading/Loading dataset: {dataset_name}...")
    ds = datasets.load_dataset(dataset_name, split='test')
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    for i in range(num_samples):
        sample = ds[i]
        img = sample['image']
        transcription = sample['transcription']
        
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        
        # Format transcription for title (shorten if too long)
        title_text = transcription.replace('\n', ' | ')
        if len(title_text) > 80:
            title_text = title_text[:80] + "..."
            
        ax.set_title(f"Sample {i+1}:\n{title_text}", fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    output_path = "dataset_samples.png"
    plt.savefig(output_path)
    print(f"Saved dataset samples to {output_path}")

def extract_attention_map(model_name="antoniorv6/smt-camera-grandstaff", dataset_name="antoniorv6/camera_grandstaff"):
    print(f"Loading model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SMTModelForCausalLM.from_pretrained(model_name).to(device)
    
    print(f"Loading 1 sample from {dataset_name}...")
    ds = datasets.load_dataset(dataset_name, split='test')
    sample = ds[0]
    
    img = np.array(sample['image'])
    img_tensor = convert_img_to_tensor(img).unsqueeze(0).to(device)
    
    print("Running prediction with return_weights=True...")
    # predict returns: text_sequence, output
    predictions, output = model.predict(img_tensor, convert_to_str=True, return_weights=True)
    
    # We will pick a token to visualize. For example, the 5th token.
    token_idx = 4 if len(predictions) > 4 else len(predictions) - 1
    target_token = predictions[token_idx]
    
    print(f"Predicted sequence length: {len(predictions)}")
    print(f"Visualizing attention for token: '{target_token}' at index {token_idx}")
    
    # Cross attentions shape per layer: (batch_size, num_heads, tgt_len, src_len)
    # output.cross_attentions is a list of tensors for each decoder layer
    # We will average the attention across all heads in the last layer
    last_layer_cross_attn = output.cross_attentions[-1] # shape: (1, 4, tgt_len, src_len)
    
    # Take attention for the specific token, average across heads
    attn_weights = last_layer_cross_attn[0, :, token_idx, :].mean(dim=0) # shape: (src_len,)
    
    # The source sequence is the flattened 2D features from the ConvNeXt encoder.
    # The width and height reduction in SMT is 2**(3+1) = 16 (since conv_next_stages=3)
    # Let's get the feature map spatial dimensions.
    encoder_features = model.forward_encoder(img_tensor) # shape: (1, dim, h, w)
    _, _, feat_h, feat_w = encoder_features.shape
    
    # Reshape attention weights to 2D
    attn_map = attn_weights.view(feat_h, feat_w).detach().cpu().numpy()
    
    # Resize attention map to match original image size
    img_h, img_w = img.shape[:2]
    attn_map_resized = cv2.resize(attn_map, (img_w, img_h))
    
    # Normalize attention map for visualization
    attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min())
    
    # Overlay on original image
    plt.figure(figsize=(12, 6))
    plt.imshow(img, cmap='gray')
    plt.imshow(attn_map_resized, cmap='jet', alpha=0.5)
    plt.title(f"Attention Map for token: '{target_token}'")
    plt.axis('off')
    
    output_path = f"attention_map_token_{token_idx}.png"
    plt.savefig(output_path)
    print(f"Saved attention map to {output_path}")

def plot_train_test_eval_placeholder():
    print("Since we don't have access to the original training logs (WandB/CSV),")
    print("here is how you would plot the evaluation metrics using matplotlib.")
    epochs = np.arange(1, 11)
    train_loss = np.exp(-epochs/3) + 0.1 * np.random.rand(10)
    val_cer = 20 * np.exp(-epochs/4) + 2 * np.random.rand(10)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', marker='o', label='Train Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_cer, 'r-', marker='s', label='Val CER')
    plt.title('Validation Character Error Rate (CER)')
    plt.xlabel('Epochs')
    plt.ylabel('CER (%)')
    plt.legend()
    
    output_path = "mock_evaluation_chart.png"
    plt.savefig(output_path)
    print(f"Saved mock evaluation chart to {output_path}")

if __name__ == "__main__":
    plot_train_test_eval_placeholder()
    plot_samples()
    extract_attention_map()

