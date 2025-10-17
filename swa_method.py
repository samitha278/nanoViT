import torch
import copy
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)
    

def create_swa_model(checkpoint_paths, model_template, device='cuda'):
    """
    Create a Stochastic Weight Averaging model from multiple checkpoints.
    This averages the weights of multiple models to create a single, often better model.
    
    Args:
        checkpoint_paths: List of paths to checkpoint files
        model_template: An instance of your model (architecture)
        device: Device to load models on
    
    Returns:
        model with averaged weights
    """
    print(f"Creating SWA model from {len(checkpoint_paths)} checkpoints...")
    
    # Initialize averaged model
    swa_model = copy.deepcopy(model_template)
    swa_state_dict = swa_model.state_dict()
    
    # Initialize all weights to zero
    for key in swa_state_dict.keys():
        swa_state_dict[key] = torch.zeros_like(swa_state_dict[key])
    
    # Sum all model weights
    n_models = 0
    for ckpt_path in checkpoint_paths:
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint['model']
            
            for key in swa_state_dict.keys():
                swa_state_dict[key] += state_dict[key]
            
            n_models += 1
            print(f"  Added: {Path(ckpt_path).name}")
        except Exception as e:
            print(f"  Skipped {Path(ckpt_path).name}: {e}")
    
    # Average the weights
    for key in swa_state_dict.keys():
        swa_state_dict[key] = swa_state_dict[key] / n_models
    
    # Load averaged weights
    swa_model.load_state_dict(swa_state_dict)
    
    print(f"SWA model created from {n_models} checkpoints")
    return swa_model






from src.vit import ViT,ViTBaseConfig
from src.data.dataset import get_val_dataloader

# Initialize model template
model_template = ViT(ViTBaseConfig()).to('cuda')
val_loader = get_val_dataloader("/home/samitha/Projects/datasets/imagenet100")



from pathlib import Path

checkpoint_dir = Path('/home/samitha/Projects/nanoViT/src/log/')
all_checkpoints = sorted(checkpoint_dir.glob('*.pt'))
selected_checkpoints = all_checkpoints[11:22]


swa_model = create_swa_model(selected_checkpoints,model_template,device)

# Save
output_path = '/home/samitha/Projects/nanoViT/src/log/swa_model.pt'
torch.save(swa_model.state_dict(), output_path)
print(f"\n SWA model saved to: {output_path}")



# Evaluate
swa_model.eval()

from src.evaluation import evaluate

results = evaluate(swa_model,val_loader,device)

print(results)