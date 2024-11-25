import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, accuracy_score, cohen_kappa_score
from tqdm import tqdm
from MobileViT_v3_ref_CBAM import MobileViTv3_v2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageFolder("train", data_transforms["train"])
val_dataset = datasets.ImageFolder("val", data_transforms["val"])
test_dataset = datasets.ImageFolder("test", data_transforms["test"])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

image_size = (224, 224)
num_classes = 3
width_multiplier = 1
model = MobileViTv3_v2(image_size, width_multiplier, num_classes)

model.load_state_dict(torch.load("model.pt", map_location='cpu'), strict=False)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0002)

scaler = GradScaler()

start_epoch = 0
checkpoint_path = "checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch, grad_clip_value=1.0):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.float() / len(loader.dataset)


    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)

    return epoch_loss, epoch_acc.item()



@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.float() / len(loader.dataset)

    return epoch_loss, epoch_acc.item()

train_losses = []
val_losses = []
train_accs = []
val_accs = []

best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(start_epoch, 300):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "/美度模型/AesViT-Net/2cbam_refbackbone_best.pt")

    print(f"Epoch [{epoch + 1}/300] "
          f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

model.load_state_dict(best_model_wts)

y_true, y_pred, y_prob_all = [], [], []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_prob = torch.softmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob_all.append(y_prob.cpu().numpy())

y_prob_all = np.concatenate(y_prob_all, axis=0)

y_true = np.array(y_true)

conf_mat = confusion_matrix(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
accuracy = accuracy_score(y_true, y_pred)
auc_roc = roc_auc_score(y_true, y_prob_all, multi_class="ovr")

print(f"Confusion Matrix:\n{conf_mat}")
print(f"Kappa: {kappa:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_pr_curve(y_true, y_prob_all):
    from sklearn.metrics import precision_recall_curve

    precision = dict()
    recall = dict()
    for i in range(y_prob_all.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_prob_all[:, i])

    plt.figure(figsize=(8, 6))
    colors = ['darkorange', 'green', 'blue']
    for i, color in zip(range(y_prob_all.shape[1]), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'Precision-Recall curve of class {i}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(conf_mat):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(conf_mat)), range(len(conf_mat)))
    plt.yticks(range(len(conf_mat)), range(len(conf_mat)))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_roc_curve(y_true, y_prob_all):
    num_classes = y_prob_all.shape[1]
    from sklearn.metrics import roc_curve, auc

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_prob_all[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = ['darkorange', 'green', 'blue']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def extract_features(model, loader, device):
    model.eval()
    features = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model.get_features(inputs)
            features.append(outputs.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def plot_tsne_visualization(vis_x, y_true):
    plt.figure(figsize=(10, 8))
    plt.scatter(vis_x[:, 0], vis_x[:, 1], c=y_true, cmap='viridis', s=10)
    plt.colorbar()
    plt.title('t-SNE Visualization')
    plt.show()


plot_training_curves(train_losses, val_losses, train_accs, val_accs)
plot_confusion_matrix(conf_mat)
plot_roc_curve(y_true, y_prob_all)
plot_pr_curve(y_true, y_prob_all)

tsne = TSNE(n_components=2)
features = extract_features(model, test_loader, device)
vis_x = tsne.fit_transform(features)
plot_tsne_visualization(vis_x, y_true)
