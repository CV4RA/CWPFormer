import torch
import os

def save_checkpoint(model, optimizer, epoch, save_dir='checkpoints', filename='model.pth'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"Model checkpoint saved at {save_path}")

def load_checkpoint(model, optimizer, load_dir='checkpoints', filename='model.pth'):
    load_path = os.path.join(load_dir, filename)
    if os.path.isfile(load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Model loaded from {load_path}, starting from epoch {epoch + 1}")
        return epoch
    else:
        print(f"No checkpoint found at {load_path}")
        return 0
