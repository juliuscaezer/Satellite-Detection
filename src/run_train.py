# File: run_train.py

from train import train, validate
import matplotlib.pyplot as plt
import torch 
device = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_sample(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for a_img, b_img, label in dataloader:
            a_img = a_img.to(device)
            b_img = b_img.to(device)
            label = label.to(device)

            # Concatenate A and B for input
            input_img = torch.cat([a_img, b_img], dim=1)
            output = model(input_img)
            output = torch.sigmoid(output)  # If using BCEWithLogitsLoss
            output = (output > 0.5).float()

            # Convert to CPU numpy arrays for plotting
            before = a_img[0].cpu().permute(1, 2, 0)
            after = b_img[0].cpu().permute(1, 2, 0)
            gt = label[0].cpu().squeeze()
            pred = output[0].cpu().squeeze()

            # Plot
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(before)
            axs[0].set_title("Before (A)")
            axs[1].imshow(after)
            axs[1].set_title("After (B)")
            axs[2].imshow(gt, cmap='gray')
            axs[2].set_title("Ground Truth")
            axs[3].imshow(pred, cmap='gray')
            axs[3].set_title("Prediction")
            for ax in axs:
                ax.axis('off')
            plt.show()
            break  # Visualize only one batch

if __name__ == "__main__":
    model, val_loader = train(
        data_dir='data/LEVIR_CD',  # adjust this path
        epochs=10,
        batch_size=4,
        lr=1e-3
    )

    visualize_sample(model, val_loader, device)

