import torch
from model import CWPFormer
from data_loader import get_dataset
from saver import load_checkpoint

def test():
    
    _, test_loader = get_dataset(batch_size=32)

    model = CWPFormer(num_classes=10)
    optimizer = torch.optim.AdamW(model.parameters())

    load_checkpoint(model, optimizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    test()
