import torch


from src.vit import ViT, ViTBaseConfig
from src.evaluation import evaluate


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)

# Load model
config = ViTBaseConfig()
model = ViT(config)

checkpoint = torch.load('/home/samitha/Projects/nanoViT/src/log/model_300000.pt',map_location='cuda') 
model.load_state_dict(checkpoint['model'])
model.to(device)

# Load val data
from src.data.dataset import get_val_dataloader
val_loader = get_val_dataloader("/home/samitha/Projects/datasets/imagenet100")

model.eval()
results = evaluate(model, val_loader, device)

print(results)