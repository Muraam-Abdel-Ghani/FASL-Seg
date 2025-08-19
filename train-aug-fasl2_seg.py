import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from dataset import EndoVisDataset
from fasl_seg_model import FASLSegModel
from transformers import SegformerImageProcessor
from metrics.dice_score import dice_score
from metrics.miou_score import miou_score
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from PIL import Image

torch.manual_seed(72638)
torch.cuda.manual_seed(72638)
torch.backends.cudnn.deterministic = True


import shutil
import os

def remove_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f'Removed: {file_path}')
        except Exception as e:
            print(f'Error removing {file_path}: {e}')

# Define Combined Loss (Tversky + CrossEntropy)
class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, smooth=1, tversky_alpha=0.7, tversky_beta=0.3):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.tversky_loss = TverskyLoss(smooth=smooth, alpha=tversky_alpha, beta=tversky_beta)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        # print(f'in combined: {outputs.shape}, {targets.shape}')
        tversky_loss = self.tversky_loss(outputs, targets)
        # print(f'combined: tversky loss: {tversky_loss.shape}')
        ce_loss = self.cross_entropy_loss(outputs, targets)
        return self.alpha * tversky_loss + (1 - self.alpha) * ce_loss

# Define Tversky Loss
class TverskyLoss(torch.nn.Module):
    def __init__(self, smooth=1, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        # print(targets)
        # print(torch.unique(targets))
        # print(f'in tversky: {outputs.shape}, {targets.shape}')
        outputs = torch.softmax(outputs, dim=1)
        # print(f'outputs softmax: ------------------')
        # print(outputs)
        num_classes = outputs.size(1)
        targets_one_hot = torch.eye(num_classes, device=targets.device)[targets].permute(0, 3, 1, 2)
        # print(f'targets_one_hot:..............')
        # print(targets_one_hot)
        # print(f'targets_one_hot: {targets_one_hot.shape}, outputs:{outputs.shape}')
        outputs_flat = outputs.reshape(-1)
        targets_flat = targets_one_hot.reshape(-1)
        
        true_pos = torch.sum(outputs_flat * targets_flat)
        false_neg = torch.sum(targets_flat * (1 - outputs_flat))
        false_pos = torch.sum((1 - targets_flat) * outputs_flat)
        
        # print(f'true_pos: {true_pos.shape}, false_neg: {false_neg.shape}, false_pos: {false_pos.shape}')
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        return 1 - tversky_index
    
def compute_iou_per_class(pred, labels, num_classes=12):
    """
    Computes IoU for each class and returns:
      - iou_list: list of IoU for each class [class0, class1, ...]
      - mean_iou: average IoU across all classes
    Args:
        outputs: (B, C, H, W) raw logits
        labels:  (B, H, W) integer class indices
    """
    # Convert to predicted classes
    # pred = outputs.argmax(dim=1)  # shape (B, H, W)
    
    iou_list = []
    for cls_id in range(num_classes):
        intersection = ((pred == cls_id) & (labels == cls_id)).sum().item()
        union = ((pred == cls_id) | (labels == cls_id)).sum().item()
        if union == 0:
            iou = 1.0  # If no pixels in union, consider IoU = 1.0 to avoid penalizing empty classes
        else:
            iou = intersection / union
        iou_list.append(iou)

    mean_iou = sum(iou_list) / len(iou_list)
    return iou_list, mean_iou


# Define instrument mappings (background included)
# EndoVis2018 tools
# id2color = {
#     0: 0,
#     1: 1,
#     2: 2,
#     3: 3,
#     4: 4,
#     5: 5,
#     6: 6,
#     7: 7,
# }

# id2label={
#     0: "background",
#     1: "Bipolar_Forceps",
#     2: "Prograsp_Forceps",
#     3: "Large_Needle_Driver",
#     4: "Monopolar_Curved_Scissors",
#     5: "Ultrasound_Probe",
#     6: "Suction_Instrument",
#     7: "Clip_Applier"
# }

# classes = [1, 2, 3, 4, 5, 6, 7]  # Only instrument classes

#EndoVis2018 parts

id2color = {
    0: [0,0,0],
    1: [0,255,0], #"instrument-shaft"
    2: [0,255,255], #"instrument-clasper"
    3: [125,255,12], #"instrument-wrist"
    4: [255,55,0], #"kidney-parenchyma"
    5: [24,55,125], #"covered-kidney"
    6: [187,155,25], #"thread"
    7: [ 0,255,125],#"clamps"
    8: [255,255,125],#"suturing-needle"
    9: [123,15,175], #"suction-instrument"
    10: [124,155,5], #"small-intestine"
    11: [12,255,141] , #"ultrasound-probe"
}

id2label={
    0: "background-tissue",
    1: "instrument-shaft",
    2: "instrument-clasper",
    3: "instrument-wrist",
    4: "kidney-parenchyma",
    5: "covered-kidney",
    6: "thread",
    7: "clamps",
    8: "suturing-needle",
    9:"suction-instrument",
    10: "small-intestine",
    11: "ultrasound-probe"
}

classes = [1, 2, 3, 4,5,6, 7,8,9,10,11] 

#EndoVis2017 tools
# id2color = {
#     0: 0,
#     1: 1,
#     2: 2,
#     3: 3,
#     4: 4,
#     5: 5,
#     6: 6,
#     7: 7,
# }

# id2label={
#     0: "Background",
#     1: "Bipolar Forceps",
#     2: "Prograsp Forceps",
#     3: "Large Needle Driver",
#     4: "Vessel Sealer",
#     5: "Grasping Retractor",
#     6: "Monopolar Curved Scissors",
#     7: "Ultrasound Probe",
# }

# classes = [1, 2, 3, 4, 5, 6, 7]  # Only instrument classes

#EndoVis2017 parts
# id2color_instruments = {
#     0: 0,  # background
#     1: 10, # shaft
#     2: 20,  # wrist
#     3: 30, # claspers
#     4: 40, # probe
# }

# id2label={
#     0: "background",
#     1: "shaft",
#     2: "wrist",
#     3: "claspers",
#     4: "probe"
# }

# instrument_ids = [1, 2, 3, 4]  # Only instrument classes

# Define parameters
image_size = (512, 512)
batch_size = 4

num_classes = len(classes) + 1  # Including background class
validation_split = 0.2 # 20% of training data for validation



# Initialize the feature extractor
extractor = SegformerImageProcessor(size=image_size)

# Data augmentation transforms for training (geometric only)
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0))
])

# Create dataset (Set  has_greyscale_labels to specify if labels are RGB or not)
# Specify dataset directory path for training data
# Specify sequences to use for training
# If data is all in the train folder without sequences:
# Specify root as the main dataset folder, then pass "train" as the sequence ["train"]

# train_parts
dataset = EndoVisDataset(
    root_dir="/notebooks/endovis2018/train/",
    extractor=extractor,
    sequences=["seq_1", "seq_2", "seq_3", "seq_4", "seq_5", "seq_6", "seq_7", "seq_9", "seq_10", "seq_11", "seq_12", "seq_13", "seq_14", "seq_15", "seq_16"],
    id2color=id2color,
    has_greyscale_labels = False, 
    instrument_ids=classes,
    image_size=image_size
)

# train_tools
# dataset = EndoVisDataset(
#     root_dir="/notebooks/endovis2018(tools)/",
#     extractor=extractor,
#     sequences=["train"],
#     id2color=id2color,
#     has_greyscale_labels = False, 
#     instrument_ids=classes,
#     image_size=image_size
# )
# val_dataset = EndoVisDataset(
#     root_dir="/notebooks/endovis2018/train/",
#     extractor=extractor,
#     sequences=["seq_1", "seq_2"],
#     id2color=id2color,
#     has_greyscale_labels = False, 
#     instrument_ids=classes,
#     image_size=image_size
# )

# train 2017
# train_dataset = EndoVisDataset(
#     root_dir="/notebooks/endovis2017(tools)/train/",
#     extractor=extractor,
#     sequences=["fold1", "fold2", "fold3"],
#     id2color=id2color,
#     has_greyscale_labels = True, 
#     instrument_ids=classes,
#     image_size=image_size
# )
# val_dataset = EndoVisDataset(
#     root_dir="/notebooks/endovis2017(tools)/train/",
#     extractor=extractor,
#     sequences=["fold0"],
#     id2color=id2color,
#     has_greyscale_labels = True, 
#     instrument_ids=classes,
#     image_size=image_size
# )

# Split dataset into training and validation
dataset_size = len(dataset)

val_size = int(validation_split * len(dataset))
train_size =  len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_miou = 0.0
start_epoch=0
# num_epochs = 100
num_epochs = 100

# checkpoint = torch.load(f"/notebooks/FASL2-Seg/models/checkpoints/checkpoint_epoch_{start_epoch}.pth")

model = FASLSegModel(num_classes=num_classes)

# If using only model weights checkpoint:
# model.load_state_dict(torch.load(f"/notebooks/FASL2-Seg/models/ablation/checkpoints/checkpoint_epoch_{start_epoch}.pth"))

# If using optimizer and model weights checkpoint:
# model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)

# experiment_name="xFB_xHLF_LLFAttn01_interpolation"

# Set up optimizer and loss function (Combined Loss)

optimizer = optim.Adam(model.parameters(), lr=0.00001)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

criterion = CombinedLoss(alpha=0.7)

# Initialize mixed precision scaler
scaler = GradScaler()

# Augment both image and labels consistently
def apply_augmentation(inputs, labels):
    # Augment the images and labels simultaneously using torchvision.transforms.functional
    if torch.rand(1).item() > 0.5:
        inputs = TF.hflip(inputs)
        labels = TF.hflip(labels)
    if torch.rand(1).item() > 0.5:
        inputs = TF.vflip(inputs)
        labels = TF.vflip(labels)
    # Apply other augmentations (like cropping) if necessary, similar approach
    return inputs, labels


def resize_predictions(predictions, target_size=(1024, 1280), mode='nearest'):
    """
    Resizes the predictions tensor to the desired spatial size.
    
    Args:
        predictions (torch.Tensor): Input tensor of shape [batch, height, width].
        target_size (tuple): Target size as (height, width).
        mode (str): Interpolation mode ('nearest' for class labels, 'bilinear' for continuous values).
    
    Returns:
        torch.Tensor: Resized tensor of shape [batch, target_height, target_width].
    """
    # Ensure input is a floating-point tensor for interpolation (except for 'nearest')
    if mode != 'nearest' and predictions.dtype != torch.float32:
        predictions = predictions.float()
    elif mode=='nearest' and predictions.dtype==torch.long:
        predictions = predictions.float()

    # Add channel dimension if not already present
    if len(predictions.shape) == 3:  # Shape is [batch, height, width]
        predictions = predictions.unsqueeze(1)  # Add channel dimension, becomes [batch, 1, height, width]

    # Resize using the specified interpolation mode (nearest mode since interpolating class labels)
    resized_predictions = F.interpolate(predictions, size=target_size, mode='nearest')

    # Remove channel dimension (if it was added)
    if resized_predictions.shape[1] == 1:  # Shape is [batch, 1, height, width]
        resized_predictions = resized_predictions.squeeze(1)  # Remove channel dimension, becomes [batch, height, width]

    return resized_predictions

# Training loop
def train_one_epoch(model, train_loader, optimizer, criterion, device,scaler):
    model.train()
    running_loss = 0.0
    total_dice_score = 0.0
    total_iou_score = 0.0
          
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device).long()

        # Here obtaining the original labels that were not resized from the dataset batch dictionary *********************For Resized*****************
        # Skip if not wanting to measure performance on resized labels ****************For Resized****************
        original_labels = batch['original_labels'].to(device).long()

        optimizer.zero_grad()

        # Apply augmentations to both inputs and labels
        augmented_inputs, augmented_labels = apply_augmentation(inputs, labels)

        # Forward pass with mixed precision
        with autocast():
            outputs = model(augmented_inputs)
            # print(f'outputs: {outputs.shape}, labels: {augmented_labels.shape}')
            
            loss = criterion(outputs, augmented_labels)
            # print(f'{loss.shape}')

        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()


        preds = torch.argmax(outputs, dim=1)
        
        # If not wanting to measure performance on resized predictions, comment next line, and pass preds and labels to the dice score and miou score methods for computing performance.*************For Resized**************
        resized_preds =  resize_predictions(preds)
        
        total_dice_score += dice_score(resized_preds, original_labels, num_classes)
        total_iou_score += miou_score(resized_preds, original_labels, num_classes)
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')

    avg_running_loss = running_loss / len(train_loader)
    avg_dice = total_dice_score / len(train_loader)
    avg_iou = total_iou_score / len(train_loader)
    
    return avg_running_loss, avg_dice, avg_iou

# Validation loop (unchanged)
def validate(model, val_loader, criterion, device, scaler):
    model.eval()
    running_loss = 0.0
    total_dice_score = 0.0
    total_iou_score = 0.0
    class_iou_sums = np.zeros(num_classes, dtype=np.float32)
    class_dice_sums = np.zeros(num_classes, dtype=np.float32)
    
    steps = 0

    # For storing IoU per class sums
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device).long()

            
            # Here obtaining the original labels that were not resized from the dataset batch dictionary **********For Resized****************
            # Skip if not wanting to measure performance on resized labels *************For Resized****************
            original_labels=batch['original_labels'].to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            

             # If not wanting to measure performance on resized predictions, comment next line, and pass preds and labels to the dice score and miou score methods for computing performance.***************For Resized*****************
            resized_preds =  resize_predictions(preds)

            per_class_dice, batch_dice_score = dice_score(resized_preds, original_labels, num_classes, return_per_class=True)
            per_class_miou, batch_iou_score = miou_score(resized_preds, original_labels, num_classes, return_per_class=True)
            
            class_iou_sums += per_class_miou
            class_dice_sums += per_class_dice
            total_dice_score += batch_dice_score
            total_iou_score += batch_iou_score
            
            steps += 1
        
            if batch_idx % 10 == 0:
                print(f"Validation Batch {batch_idx}/{len(val_loader)}, Loss: {loss.item()}")

    avg_val_loss = running_loss / len(val_loader)
    avg_val_dice = total_dice_score / len(val_loader)
    avg_val_miou = total_iou_score / len(val_loader)
    
    avg_iou_per_class = class_iou_sums / steps  # array of shape [num_classes]
    avg_dice_per_class = class_dice_sums / steps  # array of shape [num_classes]
    

    return avg_val_loss, avg_val_dice, avg_val_miou,avg_iou_per_class, avg_dice_per_class

# Main training loop
for epoch in range(start_epoch, num_epochs):

    print(f'Endovis18 Parts')
    print(f'Epoch {epoch+1}/{num_epochs}')

    train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)

   
    print(f'Epoch [{epoch+1}/{num_epochs}] - Training Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, mIoU: {train_iou:.4f}')

    # Validate the model
    val_loss, val_dice, val_iou, avg_iou_per_class, avg_dice_per_class = validate(model, val_loader, criterion, device,scaler)
    print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, mIoU: {val_iou:.4f}')

    print(f"  Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}")
    # print(f"  Train IoU/class: {[round(v, 3) for v in unet_iou_train_per_class]}")
    print(f"  Val   Loss: {val_loss:.4f},   Val mIoU: {val_iou:.4f}")
    print(f"  Val   IoU/class: {[round(v, 3) for v in avg_iou_per_class]}")
    print(f"  Val   Dice/class: {[round(v, 3) for v in avg_dice_per_class]}")

# For this you need a checkpoints folder in your models folder
# We save the model file and the optimizer state as well.

    remove_files_in_folder(f"/notebooks/FASL2-Seg/models/checkpoints")
    torch.save({ 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()} , 
               f"/notebooks/FASL2-Seg/models/checkpoints/checkpoint_epoch_{epoch+1}.pth")
    print(f"New checkpoint saved epoch:{epoch+1}")

    # Save the best model
    #  The model is saved using the epoch number and validation iou in the name. It will save every new best without replacing previous best. Remove these details from the name if not interested in this information.
    if val_iou > best_miou:
        best_miou = val_iou
        torch.save({ 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()} ,
                   f"/notebooks/FASL2-Seg/models/best_seg_18_parts_epoch_{epoch+1}_miou_{val_iou:.4f}.pth")
        print(f"New best model saved with mIoU: {best_miou:.4f}")

# Save the last model
torch.save({ 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()} ,
           f"/notebooks/FASL2-Seg/models/last_seg_18_parts_epoch_{epoch+1}_miou_{val_iou:.4f}.pth")

