import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import logging
import asyncio
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def compare_validation_images(model: tf.keras.Model, 
                                    validation_dataset: tf.data.Dataset, 
                                    num_samples: int = 5):
    """
    Compare original, masked (input), predicted, and true (unmasked) images from the validation dataset.
    
    Args:
    model (tf.keras.Model): Trained model for prediction
    validation_dataset (tf.data.Dataset): Validation dataset
    num_samples (int): Number of samples to display
    """
    for i, (masked_batch, true_batch) in enumerate(validation_dataset.take(num_samples)):
        # Get prediction
        predicted_batch = await asyncio.to_thread(lambda: model(masked_batch).numpy())

        # Process each image in the batch
        for j in range(masked_batch.shape[0]):
            masked_image = masked_batch[j].numpy()
            true_image = true_batch[j].numpy()
            predicted_image = predicted_batch[j]

            # Create a figure with 4 subplots
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f'Sample {i*masked_batch.shape[0]+j+1}')

            # Original (masked) image
            axs[0].imshow(masked_image)
            axs[0].set_title('Masked Input')
            axs[0].axis('off')

            # Predicted image
            axs[1].imshow(predicted_image)
            axs[1].set_title('Predicted')
            axs[1].axis('off')

            # True (unmasked) image
            axs[2].imshow(true_image)
            axs[2].set_title('Ground Truth')
            axs[2].axis('off')

            # Difference image
            diff = np.abs(true_image - predicted_image)
            axs[3].imshow(diff, cmap='hot')
            axs[3].set_title('Difference')
            axs[3].axis('off')

            plt.tight_layout()
            await asyncio.to_thread(plt.savefig, f'./results/comparison_{i*masked_batch.shape[0]+j+1}.png')
            plt.close(fig)

            # Calculate and print metrics
            mse = np.mean((true_image - predicted_image) ** 2)
            psnr = await asyncio.to_thread(calculate_psnr, true_image, predicted_image)
            ssim = await asyncio.to_thread(calculate_ssim, true_image, predicted_image)
            logger.info(f"Sample {i*masked_batch.shape[0]+j+1} Metrics:")
            logger.info(f"MSE: {mse:.4f}")
            logger.info(f"PSNR: {psnr:.4f}")
            logger.info(f"SSIM: {ssim:.4f}")
            logger.info("\n")

        if (i+1) * masked_batch.shape[0] >= num_samples:
            break

    logger.info(f"Saved {num_samples} comparison images.")

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM) between two images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# Usage example
# This function should be called from an async context
# await compare_validation_images(model, validation_dataset, num_samples=5)