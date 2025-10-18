import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from eval_vit import ViT, ViTBaseConfig


def visualize_positional_embeddings(model, save_path=None):
    """
    Visualize positional embeddings as a grid of patches.
    """
    # Get positional embeddings (exclude CLS token)
    pos_embed = model.embd.pos_embd.weight.data[1:].cpu()  # Shape: (num_patches, embed_dim)
    
    num_patches = pos_embed.shape[0]
    grid_size = int(np.sqrt(num_patches))
    
    # Reshape to grid
    pos_embed_grid = pos_embed.reshape(grid_size, grid_size, -1)  # (H, W, embed_dim)
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'ViT Positional Embeddings\nGrid: {grid_size}x{grid_size}', 
                 fontsize=16, fontweight='bold')
    
    # Plot each patch's embedding as a small heatmap
    for i in range(grid_size):
        for j in range(grid_size):
            # Get embedding for this patch
            embed = pos_embed_grid[i, j].numpy()
            
            # Reshape to 2D for visualization (arbitrary but consistent)
            h = int(np.sqrt(len(embed)))
            embed_2d = embed[:h*h].reshape(h, h)
            
            # Plot
            axes[i, j].imshow(embed_2d, cmap='viridis', aspect='auto')
            axes[i, j].axis('off')
            
            # Add patch number
            axes[i, j].text(0.5, -0.1, f'{i*grid_size + j + 1}', 
                          ha='center', va='top', 
                          transform=axes[i, j].transAxes, 
                          fontsize=8)
    
    # Add labels
    fig.text(0.5, 0.02, 'Input patch column', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Input patch row', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def visualize_positional_similarity(model, save_path=None):
    """
    Visualize cosine similarity between positional embeddings.
    """
    # Get positional embeddings (exclude CLS token)
    pos_embed = model.embd.pos_embd.weight.data[1:].cpu()
    
    # Compute cosine similarity
    pos_embed_norm = pos_embed / pos_embed.norm(dim=1, keepdim=True)
    similarity = pos_embed_norm @ pos_embed_norm.T
    
    num_patches = pos_embed.shape[0]
    grid_size = int(np.sqrt(num_patches))
    
    # Plot similarity matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(similarity.numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(f'Positional Embedding Cosine Similarity\n{grid_size}x{grid_size} patches', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Patch index', fontsize=12)
    ax.set_ylabel('Patch index', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Cosine similarity')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def main():
    config = ViTBaseConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ViT(config).to(device)
    
    checkpoint = torch.load(
        "src/log/model_300000.pt",
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model'])
    
    save_dir = Path('experiments/pos_embed_viz/')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize positional embeddings
    visualize_positional_embeddings(
        model, 
        save_path=save_dir / 'pos_embeddings.png'
    )
    
    # Visualize similarity between positions
    visualize_positional_similarity(
        model,
        save_path=save_dir / 'pos_similarity.png'
    )


if __name__ == '__main__':
    main()