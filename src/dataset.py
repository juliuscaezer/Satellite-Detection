import os 
import glob
from PIL import Image 
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision import transforms

class LEVIRCDDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.a_paths = sorted(glob.glob((os.path.join(self.root_dir, 'A', '*.png'))))
        self.b_paths = sorted(glob.glob((os.path.join(self.root_dir, 'B', '*.png'))))
        self.label_paths = sorted(glob.glob(os.path.join(self.root_dir, 'label', '*.png')))
        self.transform = transform 
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.a_paths)
    
    def __getitem__(self, idx):
        a_img = Image.open(self.a_paths[idx]).convert('RGB')
        b_img = Image.open(self.b_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('L')  # grayscale mask

        if self.transform:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)
            label = self.transform(label)

        a_img = self.to_tensor(a_img)
        b_img = self.to_tensor(b_img)
        label = self.to_tensor(label)
        label = (label > 0.5).float()

        return a_img, b_img, label
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    transform = transforms.ToTensor()

    dataset = LEVIRCDDataset(root_dir='./data/LEVIR-CD', split='train', transform=transform)
    a_img, b_img, label = dataset[0]

    print(f"A image shape: {a_img.shape}")  # [3, H, W]
    print(f"B image shape: {b_img.shape}")  # [3, H, W]
    print(f"Label shape: {label.shape}")    # [1, H, W]

    # Visualize
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].imshow(a_img.permute(1, 2, 0))  # convert C,H,W to H,W,C
    axs[0].set_title("Image A (Before)")
    axs[0].axis('off')

    axs[1].imshow(b_img.permute(1, 2, 0))
    axs[1].set_title("Image B (After)")
    axs[1].axis('off')

    axs[2].imshow(label.squeeze(0), cmap='gray')
    axs[2].set_title("Change Mask")
    axs[2].axis('off')

    plt.show()

