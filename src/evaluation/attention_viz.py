import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.vit import ViT,ViTBaseConfig
from data.dataset import get_val_dataloader
from torchvision import datasets, transforms

def visualize_attention(model, image, device, save_path=None):
    
    model.eval()
    
    image_tensor = image.to(device)     # B,C,H,W
    
    with torch.no_grad():
        output = model(image_tensor)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map (placeholder - implement based on your model)
    # This is simplified - you'll need to capture attention weights
    axes[1].imshow(img_np)
    axes[1].set_title('Attention Visualization')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()



def main():
    # Load config and model
    config = ViTBaseConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ViT(config).to(device)
    
    # Load trained weights
    checkpoint = torch.load('/home/samitha/Projects/nanoViT/src/log/swa_model_29to_last.pt')
    model.load_state_dict(checkpoint)
    
    val_dataloader = get_val_dataloader("/home/samitha/Projects/datasets/imagenet100",batch_size=1)
    
    # Visualize a few examples
    save_dir = Path('experiments/attention_viz/')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    
    val_data = iter(val_dataloader)
    for i in range(4):
        image, label = next(val_data)    # B,C,H,W
        
        visualize_attention(
            model, image, device,
            save_path=save_dir / f'attention_{i}.png'
        )

if __name__ == '__main__':
    main()