import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import CWPFormer
from data_loader import get_dataset
from utils import accuracy
from saver import save_checkpoint, load_checkpoint
from config import Config
from losses import get_loss_fn
from logger import setup_logger

logger = setup_logger(Config.LOG_DIR)

train_loader, val_loader = get_dataset(batch_size=Config.BATCH_SIZE)

model = CWPFormer(num_classes=Config.NUM_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = get_loss_fn("cross_entropy")
optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

start_epoch = load_checkpoint(model, optimizer, Config.CHECKPOINT_DIR)

def train():
    model.train()
    for epoch in range(start_epoch, Config.EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

        save_checkpoint(model, optimizer, epoch, save_dir=Config.CHECKPOINT_DIR)

        validate()

def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f'Validation Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    train()
