import warnings

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
#multiprocessing.set_start_method('spawn', force=True)


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
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--weight-decay", default=0.2, type=float)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--download", action="store_true", default=True, help="Download the dataset if it doesn't exist")
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



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import numpy as np
import os


class EnsembleSpatialModel(nn.Module):
    """
    Ensemble model for spatial relationships that uses specialized heads
    for different spatial relationship types.
    """

    def __init__(self, clip_model, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.clip_model = clip_model
        self.device = device

        embed_dim = clip_model.visual.output_dim

        # Create specialized heads for each relationship type
        # Each head projects from the embedding dimension to a scalar score
        self.heads = nn.ModuleDict({
            "left_of": nn.Linear(embed_dim, 1).to(device),
            "on": nn.Linear(embed_dim, 1).to(device),
            "right_of": nn.Linear(embed_dim, 1).to(device),
            "under": nn.Linear(embed_dim, 1).to(device)
        })

        # Create shared projection layer for all relationships
        self.shared_projection = nn.Linear(embed_dim, embed_dim).to(device)

        # Initialize weights
        for head in self.heads.values():
            head.weight.data.normal_(mean=0.0, std=0.02)
            head.bias.data.zero_()

        self.shared_projection.weight.data.normal_(mean=0.0, std=0.02)
        self.shared_projection.bias.data.zero_()

    def encode_image(self, image):
        return self.clip_model.encode_image(image)

    def encode_text(self, text):
        return self.clip_model.encode_text(text)

    def forward(self, image, caption_options):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)

        # Apply shared projection to image features
        image_features = self.shared_projection(image_features)

        indices_by_relation = self._classify_captions_by_relation(caption_options)
        final_scores = torch.zeros(len(caption_options), device=self.device)

        for relation, indices in indices_by_relation.items():
            if not indices:
                continue

            rel_captions = [caption_options[i] for i in indices]

            rel_tokens = torch.cat([clip.tokenize(c) for c in rel_captions])
            rel_tokens = rel_tokens.to(self.device)
            rel_features = self.clip_model.encode_text(rel_tokens)

            image_features_norm = F.normalize(image_features, dim=1)
            rel_features_norm   = F.normalize(rel_features, dim=1)

            similarity = image_features_norm @ rel_features_norm.T

            head_scores = self.heads[relation](rel_features)

            combined_scores = similarity + head_scores

            for i, idx in enumerate(indices):
                final_scores[idx] = combined_scores[0, i]

        return final_scores

    def predict(self, image, text_options, temperature=1.0):
        text_tokens = torch.cat([clip.tokenize(c) for c in text_options]).to(self.device)
        text_by_relation = self._classify_captions_by_relation(text_options, text_tokens)
        scores_by_relation = self.forward(image, text_by_relation)
        final_scores = torch.zeros(len(text_options), device=self.device)

        for relation, indices in text_by_relation.items():
            if len(indices) > 0:
                rel_scores = scores_by_relation[relation]

                for i, idx in enumerate(indices):
                    final_scores[idx] = rel_scores[0, i]

        probs = F.softmax(final_scores / temperature, dim=0)

        return probs

    def _classify_captions_by_relation(self, text_options):
        """
        Classify caption options by spatial relationship type and return indices

        Args:
            text_options: List of caption text strings

        Returns:
            Dictionary mapping relationship types to lists of indices
        """
        # Initialize empty lists for each relationship
        indices_by_relation = {
            "left_of": [],
            "on": [],
            "right_of": [],
            "under": []
        }

        # Classify each caption
        for i, caption in enumerate(text_options):
            for relation in indices_by_relation.keys():
                if relation.replace('_', ' ') in caption[0]:
                    indices_by_relation[relation].append(i)
                    break

        return indices_by_relation


def train_ensemble_model(
        model,
        train_loader,
        num_epochs=5,
        learning_rate=1e-4,
        weight_decay=1e-4,
        save_dir="./ensemble_model/",
        device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the ensemble spatial relationship model
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Set up optimizer
    # Only optimize the heads and shared projection, not the base CLIP model
    params_to_optimize = list(model.model.heads.parameters()) + list(model.model.shared_projection.parameters())
    optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.5, verbose=True)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            image = batch["image_options"][0].to(device)  # Use the first image option
            caption_options = batch["caption_options"]

            indices_by_relation = model.model._classify_captions_by_relation(caption_options)

            if not any(indices for indices in indices_by_relation.values()):
                continue

            final_scores = model.model.forward(image, caption_options)

            # Target is always index 0 (the correct caption)
            targets = torch.zeros(1, dtype=torch.long, device=device)
            loss = F.cross_entropy(final_scores.unsqueeze(0), targets)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(final_scores, 0)
            correct += (predicted == 0).sum().item()  # Correct index is 0
            total += 1

            running_loss += loss.item()

            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'accuracy': 100 * correct / total
            })

        scheduler.step(100 * correct / total)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
            }, os.path.join(save_dir, "best_model.pt"))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': epoch_accuracy,
        }, os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

    print(f"Training complete. Best accuracy: {best_accuracy:.2f}%")
    return model


def direct_evaluate_ensemble(ensemble_model, test_loader, device):
    """
    Directly evaluate the ensemble model without going through wrapper
    """
    ensemble_model.model.eval()
    correct = {
        "left_of": 0,
        "on": 0,
        "right_of": 0,
        "under": 0
    }
    total = {
        "left_of": 0,
        "on": 0,
        "right_of": 0,
        "under": 0
    }

    with torch.no_grad():
        for batch in tqdm(test_loader):
            image = batch["image_options"][0].to(device)
            caption_options = batch["caption_options"]

            final_scores = ensemble_model.model.forward(image, caption_options)

            _, predicted = torch.max(final_scores, 0)

            # Check if correct (correct caption is at index 0)
            is_correct = (predicted == 0).item()

            correct_caption = caption_options[0]
            for rel in ["left_of", "on", "right_of", "under"]:
                if rel.replace('_', ' ') in correct_caption[0]:
                    total[rel] += 1
                    if is_correct:
                        correct[rel] += 1
                    break

    print("\nDirect Evaluation Results:")
    for rel in ["left_of", "on", "right_of", "under"]:
        if total[rel] > 0:
            accuracy = correct[rel] / total[rel]
            print(f"Preposition: {rel}, Accuracy: {accuracy:.4f} ({correct[rel]}/{total[rel]})")

    return correct, total

def evaluate_ensemble_model(
        model,
        eval_loader,
        device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate the ensemble model on a dataset
    """
    model.eval()
    correct = 0
    total = 0

    relation_correct = {
        "left_of": 0,
        "on": 0,
        "right_of": 0,
        "under": 0
    }
    relation_total = {
        "left_of": 0,
        "on": 0,
        "right_of": 0,
        "under": 0
    }

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Extract batch data
            image = batch["image_options"][0].to(device)
            caption_options = batch["caption_options"]

            # Get prediction
            probs = model.predict(image, caption_options)
            _, predicted = torch.max(probs, 0)

            # Update overall accuracy
            correct += (predicted == 0).sum().item()  # Correct index is 0
            total += 1

            # Determine the relationship type
            correct_caption = caption_options[0]
            for relation in relation_correct.keys():
                if f"is {relation}" in correct_caption:
                    relation_total[relation] += 1
                    if predicted == 0:
                        relation_correct[relation] += 1
                    break

    accuracy = 100 * correct / total

    relation_accuracy = {}
    for relation in relation_correct.keys():
        if relation_total[relation] > 0:
            relation_accuracy[relation] = 100 * relation_correct[relation] / relation_total[relation]
        else:
            relation_accuracy[relation] = 0.0

    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("Accuracy by Relationship Type:")
    for relation, acc in relation_accuracy.items():
        print(f"  {relation}: {acc:.2f}% ({relation_correct[relation]}/{relation_total[relation]})")

    return accuracy, relation_accuracy


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

    output_file = os.path.join(args.output_dir, f"{args.dataset}_results.csv")
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

    print("\nEvaluation Results:")
    for record in result_records:
        print(f"Preposition: {record['Preposition']}, Accuracy: {record['Accuracy']:.4f}")

    return result_records

def main():
    args = config()
    seed_all(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    from model_zoo.clip_models import CLIPWrapper

    # Load the NegCLIP model
    model, image_preprocess, device = load_negclip_model(args.device)

    # Create the ensemble model
    ensemble_model = EnsembleSpatialModel(model.model, device=device)

    # We'll keep the original model reference for evaluation after training
    original_model = model

    # Create a wrapped version for the ensemble model
    wrapped_ensemble_model = CLIPWrapper(ensemble_model, device)

    # Load datasets
    train_dataset = get_dataset(
        args.dataset,
        image_preprocess=image_preprocess,
        download=args.download,
        split="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,  # Enable shuffling for better training
        num_workers=args.num_workers
    )

    # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn
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

    print("Evaluating original model...")
   # evaluate_model(original_model, test_loader, device, test_dataset, args)

    if not args.evaluate_only:
        params_to_optimize = list(ensemble_model.heads.parameters()) + list(ensemble_model.shared_projection.parameters())
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        initial_weights = {}
        for rel in ["left_of", "on", "right_of", "under"]:
            initial_weights[rel] = wrapped_ensemble_model.model.heads[rel].weight.clone().detach().cpu().numpy()

        print("Training ensemble model...")
        trained_wrapped_ensemble_model = train_ensemble_model(
            wrapped_ensemble_model,
            train_loader,
            num_epochs=20,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            save_dir="./ensemble_model/"
        )

        torch.save(ensemble_model.state_dict(), os.path.join(args.output_dir, "trained_ensemble_model.pt"))

        print("Ensemble model trained and saved.")

    print("Evaluating ensemble model...")

    final_weights = {}
    for rel in ["left_of", "on", "right_of", "under"]:
        final_weights[rel] = trained_wrapped_ensemble_model.model.heads[rel].weight.clone().detach().cpu().numpy()

    for rel in ["left_of", "on", "right_of", "under"]:
        weight_diff = np.abs(final_weights[rel] - initial_weights[rel]).mean()
        print(f"Average weight change for {rel}: {weight_diff}")

    direct_evaluate_ensemble(trained_wrapped_ensemble_model, test_loader, device)

if __name__ == "__main__":
    main()