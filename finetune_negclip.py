giimport warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import open_clip
from tqdm import tqdm
import random
import numpy as np
import json
import clip
from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores

CACHE_DIR = "./model_cache"
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


def config():
    parser = argparse.ArgumentParser()
    # Default to 'cuda' if available
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use: 'cpu', 'cuda' if available, or 'mps' if available on MacOS")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--eval-batch-size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model-name", default="NegCLIP", type=str)
    parser.add_argument("--dataset", default="Controlled_Images_A", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--weight-decay", default=0.2, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--download", action="store_true", help="Download the dataset if it doesn't exist")
    parser.add_argument("--output-dir", default="./outputs", type=str)
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for analysis")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and only evaluate")
    return parser.parse_args()


def get_device(device_arg):
    """
    Safe device selection for different platforms
    """
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print(f"Requested device '{device_arg}' not available, using CPU instead")
        return torch.device("cpu")


def load_negclip_model(device_str, root_dir=CACHE_DIR):
    """
    Load the NegCLIP model for fine-tuning, safely handling device mapping
    """
    from model_zoo.clip_models import CLIPWrapper  # Assuming you have this in your model_zoo
    # Create the directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)
    # Get appropriate device
    device = get_device(device_str)
    path = os.path.join(root_dir, "negclip.pth")
    if not os.path.exists(path):
        print("Downloading the NegCLIP model...")
        import gdown
        gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
    # Load the model weights to CPU first to avoid device issues
    print("Loading NegCLIP weights...")
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    # Create model on CPU first
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None, device="cpu")
    model.load_state_dict(state_dict, strict=False)

    # Freeze most of the model to prevent catastrophic forgetting
    print("Freezing most model parameters...")
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Only train the last layer of the text encoder
    print("Enabling training for specific layers...")
    for name, param in model.named_parameters():
        if 'ln_final' in name or 'text_projection' in name:
            param.requires_grad = True
            print(f"Training enabled for: {name}")

    # Move model to the target device after loading weights
    model = model.to(device)
    model = model.train()
    print(f"Model loaded and moved to {device}")
    clip_model = CLIPWrapper(model, device)
    return clip_model, image_preprocess, device

def train_negclip_on_controlled_images(model, train_loader, optimizer, device, epochs, output_dir, args):
    """
    Fine-tune the NegCLIP model on the Controlled_Images dataset.
    For each image, the correct caption is at index 0 out of 4 caption options.
    """

    def contrastive_loss(logits, targets, margin=0.2):
        """
        Contrastive loss for 1D logits tensor (for batch size of 1)

        Args:
            logits: tensor of shape [num_options]
            targets: tensor of shape [1] containing index of positive example
            margin: minimum margin between positive and negative scores
        """
        positive_idx = targets[0]
        positive_score = logits[positive_idx]

        margins = margin - (positive_score - logits)
        margins[positive_idx] = 0
        margins = torch.clamp(margins, min=0)

        return margins.sum()

    for epoch in range(epochs):
        model.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            # In Controlled_Images, we have multiple images and for each image, we have 4 caption options
            # The correct caption is always at index 0
            image = batch["image_options"][0]
            batch_size = 1
            total += batch_size

            image_features = model.model.encode_image(image.to(device))
            image_features = F.normalize(image_features, dim=1)
            caption_options = batch["caption_options"]

            caption_tokenized = torch.cat([clip.tokenize(c) for c in caption_options])
            text_features = model.model.encode_text(caption_tokenized.to(device))
            text_features = F.normalize(text_features, dim=1)

            logits = 100 * (image_features @ text_features.T)

            # Correct caption is always at index 0
            targets = torch.zeros(batch_size, dtype=torch.long, device=device)
            loss = contrastive_loss(logits[0], targets)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'accuracy': 100 * correct / total
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, f"checkpoint_epoch_{epoch + 1}.pt")

    print("Training completed!")

    return model


def evaluate_model(model, test_loader, device, dataset, args):
    """
    Evaluate the model on the test dataset
    """
    model.model.eval()  # Set the model to evaluation mode
    print("Evaluating model...")

    scores = model.get_retrieval_scores_batched(test_loader)
    result_records = dataset.evaluate_scores(scores)

    for record in result_records:
        record.update({"Model": f"{args.model_name}_finetuned", "Dataset": args.dataset, "Seed": args.seed})

    output_file = os.path.join(args.output_dir, f"{args.dataset}_results_finetuned.csv")
    df = pd.DataFrame(result_records)
    print(f"Saving results to {output_file}")
    if os.path.exists(output_file):
        all_df = pd.read_csv(output_file, index_col=0)
        all_df = pd.concat([all_df, df])
        all_df.to_csv(output_file)
    else:
        df.to_csv(output_file)

    if args.save_scores:
        save_scores(scores, args)

    # Print the results for easy comparison
    print("\nEvaluation Results:")
    for record in result_records:
        print(f"Preposition: {record['Preposition']}, Accuracy: {record['Accuracy']:.4f}")

    return result_records


def main():
    args = config()
    seed_all(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the NegCLIP model
    model, image_preprocess, device = load_negclip_model(args.device)

    train_dataset = get_dataset(
        args.dataset,
        image_preprocess=image_preprocess,
        download=args.download,
        split="train"
    )

    # Set up dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Use batch size of 1 for this dataset structure
        shuffle=False,
        num_workers=args.num_workers
    )

    collate_fn = _default_collate if image_preprocess is None else None

    test_dataset = get_dataset(
        args.dataset,
        image_preprocess=image_preprocess,
        download=False,
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    evaluate_model(model, test_loader, device, test_dataset, args)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print(f"Starting training for {args.epochs} epochs...")
    model = train_negclip_on_controlled_images(
        model,
        train_loader,
        optimizer,
        device,
        args.epochs,
        args.output_dir,
        args
    )

    evaluate_model(model, test_loader, device, test_dataset, args)

if __name__ == "__main__":
    main()