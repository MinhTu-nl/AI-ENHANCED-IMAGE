import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm  # For progress bar

# Import from lowlightenhancement.py
from enhancement import enhance_net_nopool, L_spa, L_exp, L_color, L_TV_R, weights_init

# --- Configuration ---
# (Keep your configuration variables here)
DATASET_PATHS = [
    './data/DarkPair/low',
    './data/LOL_dataset/our485/low',
    './data/LOL_dataset/eval15/low'
]

IMAGE_SIZE = 256
BATCH_SIZE = 8 # Adjust based on GPU memory
EPOCHS = 100 # Number of training epochs
LEARNING_RATE = 1e-4
EXP_PATCH_SIZE = 16 # Patch size for Exposure Loss
EXP_MEAN_VAL = 0.6 # Target exposure level (adjust as needed)
WEIGHT_SPA = 1.0 # Weight for Spatial Consistency Loss
WEIGHT_EXP = 10.0 # Weight for Exposure Loss
WEIGHT_COL = 5.0 # Weight for Color Constancy Loss
WEIGHT_TV_R = 200.0 # Weight for Illumination Smoothness Loss (TV on 'r')

MODEL_SAVE_PATH = './models/'
SAVE_EVERY_EPOCH = 10 # Save model every N epochs

# --- Dataset Class ---
class LowLightDataset(Dataset):
    # (Keep the dataset class definition exactly as before)
    def __init__(self, image_paths, transform=None):
        self.image_files = []
        print("Loading image paths...")
        for path in image_paths:
            if not os.path.isdir(path):
                print(f"Warning: Path not found or not a directory: {path}")
                continue
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff') # Added more extensions
            try:
                for filename in os.listdir(path):
                    if filename.lower().endswith(valid_extensions):
                        # Check if it's actually a file (sometimes folders might have extensions)
                        if os.path.isfile(os.path.join(path, filename)):
                            self.image_files.append(os.path.join(path, filename))
            except OSError as e:
                print(f"Error accessing path {path}: {e}")
                continue # Skip this path if there's an OS error
        print(f"Found {len(self.image_files)} images.")
        if not self.image_files:
            raise ValueError("No images found in the specified dataset paths. Check paths and permissions.")
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            # Use 'RGBA' and convert to 'RGB' to handle potential transparency issues
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading or processing image {img_path}: {e}")
            # Return a dummy tensor or skip this sample if errors persist
            # Creating a dummy tensor matching the expected output size
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))


# --- Main Execution Block ---
if __name__ == "__main__": # <--- ADD THIS GUARD

    # Create model save directory if it doesn't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # --- Training Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor() # Scales images to [0, 1]
    ])

    # Dataset and DataLoader
    dataset = LowLightDataset(DATASET_PATHS, transform=transform)

    # Check if dataset is empty after potential loading errors
    if len(dataset) == 0:
        print("Error: Dataset is empty. Check image paths and loading errors.")
        exit() # Exit if no data could be loaded


    # Set num_workers based on OS. Often 0 works best on Windows.
    # You can try increasing it if 0 is too slow, but start with 0 if issues persist.
    num_workers = 0 if os.name == 'nt' else 4 # 'nt' is the name for Windows OS
    print(f"Using num_workers: {num_workers}")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    # Model
    model = enhance_net_nopool().to(device)
    model.apply(weights_init) # Initialize weights

    # Loss Functions
    loss_spa = L_spa(device=device)
    loss_exp = L_exp(EXP_PATCH_SIZE, EXP_MEAN_VAL, device=device)
    loss_col = L_color()
    loss_tv_r = L_TV_R(TVLoss_weight=1) # Weight applied later

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, low_light_imgs in enumerate(pbar):
            # Check if the batch contains dummy tensors (from loading errors)
            # This check might be basic; more robust error handling could be added
            if torch.equal(low_light_imgs, torch.zeros((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))):
                print(f"Warning: Skipping batch {batch_idx} due to potential loading errors.")
                continue

            low_light_imgs = low_light_imgs.to(device)

            optimizer.zero_grad()

            # Forward pass
            enhanced_img_1, enhanced_img_final, r_maps = model(low_light_imgs)

            # Calculate Losses (Using final enhanced image and original low-light image)
            l_spa = loss_spa(enhanced_img_final, low_light_imgs)
            l_exp = loss_exp(enhanced_img_final)
            l_col = loss_col(enhanced_img_final)
            l_tv_r = loss_tv_r(r_maps) # TV loss on the curve parameters 'r'

            # Total Loss (Weighted sum)
            current_loss = WEIGHT_SPA * l_spa + \
                           WEIGHT_EXP * l_exp + \
                           WEIGHT_COL * l_col + \
                           WEIGHT_TV_R * l_tv_r

            # Backward pass and optimization
            current_loss.backward()
            optimizer.step()

            total_loss += current_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{current_loss.item():.4f}",
                'L_spa': f"{l_spa.item():.4f}",
                'L_exp': f"{l_exp.item():.4f}",
                'L_col': f"{l_col.item():.4f}",
                'L_tv_r': f"{l_tv_r.item():.4f}"
            })

        if len(dataloader) > 0: # Avoid division by zero if dataloader is empty
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} - No batches were processed.")


        # Save model checkpoint
        if (epoch + 1) % SAVE_EVERY_EPOCH == 0 or (epoch + 1) == EPOCHS:
            save_path = os.path.join(MODEL_SAVE_PATH, f"enhance_net_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print("Training Finished.")