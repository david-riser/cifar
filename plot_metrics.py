import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    args = parser.parse_args()

    # Load metrics
    metrics = pd.read_csv(args.metrics)
    epochs = np.arange(1,len(metrics) + 1)
    
    # Draw Figures
    plt.figure(figsize=(12,10))

    plt.subplot(2,1,1)
    plt.plot(epochs, metrics['train_loss'], label='Train')
    plt.plot(epochs, metrics['valid_loss'], label='Test')
    plt.grid(alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    
    plt.subplot(2,1,2)
    plt.plot(epochs, metrics['train_top1'], label='Train (Top 1)')
    plt.plot(epochs, metrics['valid_top1'], label='Test (Top 1)')
    plt.plot(epochs, metrics['train_top5'], label='Train (Top 5)')
    plt.plot(epochs, metrics['valid_top5'], label='Test (Top 5)')
    plt.grid(alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(frameon=False)
    plt.savefig(args.output_dir + 'metrics.png', bbox_inches='tight')
