"""
Train MLP fusion model on cached RAVDESS features.

Loads pre-extracted face + audio probability vectors from .npz files,
trains a small MLP with 5-fold cross-validation, saves best model.

Usage:
    PYTHONPATH=. venv/bin/python tests/train_mlp_fusion.py

Requires:
    data/ravdess_train_features.npz (from tests/extract_features.py)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mlp_fusion import FusionMLP, MLPFusion
from core.fusion import SHARED_EMOTIONS


# ============================================================================
# Configuration
# ============================================================================

TRAIN_FEATURES = "data/ravdess_train_features.npz"
MODEL_OUTPUT = "models/mlp_fusion.pt"
NUM_FOLDS = 5
EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 32
LR = 0.001
HIDDEN1 = 32
HIDDEN2 = 16
DROPOUT = 0.3


# ============================================================================
# Training
# ============================================================================

def train_one_fold(model, train_loader, val_loader, epochs, patience, lr):
    """Train model for one fold, return best validation accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        val_acc = val_correct / val_total if val_total > 0 else 0
        train_acc = train_correct / train_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Restore best state
    if best_state:
        model.load_state_dict(best_state)

    return best_val_acc


def main():
    # Load features
    if not os.path.exists(TRAIN_FEATURES):
        print(f"Training features not found: {TRAIN_FEATURES}")
        print("Run 'PYTHONPATH=. venv/bin/python tests/extract_features.py' first.")
        sys.exit(1)

    data = np.load(TRAIN_FEATURES, allow_pickle=True)
    face_vecs = data['face_vecs']       # (N, 7)
    audio_vecs = data['audio_vecs']     # (N, 7)
    labels = data['labels']             # (N,) int64
    actor_ids = data['actor_ids']       # (N,) int32
    label_names = data['label_names']   # (N,) str

    # Concatenate face + audio → 14-dim input
    X = np.concatenate([face_vecs, audio_vecs], axis=1).astype(np.float32)
    y = labels.astype(np.int64)
    groups = actor_ids  # For group-based CV (split by actor)

    print(f"Training data: {X.shape[0]} samples, {X.shape[1]}-dim input")
    print(f"Label distribution:")
    for i, em in enumerate(SHARED_EMOTIONS):
        count = np.sum(y == i)
        print(f"  {em}: {count}")
    print(f"Actors: {sorted(set(actor_ids))}")

    # 5-fold cross-validation by actor (speaker-independent within train set)
    print(f"\n{'='*60}")
    print(f"{NUM_FOLDS}-fold cross-validation (grouped by actor)")
    print(f"{'='*60}")

    splitter = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_accs = []
    best_overall_acc = 0.0
    best_overall_state = None

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups)):
        X_train = torch.tensor(X[train_idx])
        y_train = torch.tensor(y[train_idx])
        X_val = torch.tensor(X[val_idx])
        y_val = torch.tensor(y[val_idx])

        train_actors = sorted(set(actor_ids[train_idx]))
        val_actors = sorted(set(actor_ids[val_idx]))

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=BATCH_SIZE)

        model = FusionMLP(
            input_dim=14, hidden1=HIDDEN1, hidden2=HIDDEN2,
            num_classes=len(SHARED_EMOTIONS), dropout=DROPOUT)

        val_acc = train_one_fold(model, train_loader, val_loader, EPOCHS, PATIENCE, LR)
        fold_accs.append(val_acc)

        print(f"  Fold {fold+1}: val_acc={val_acc:.1%} "
              f"(train actors: {train_actors}, val actors: {val_actors})")

        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_overall_state = {k: v.clone() for k, v in model.state_dict().items()}

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\nCV Results: {mean_acc:.1%} ± {std_acc:.1%}")

    # Train final model on all training data
    print(f"\n{'='*60}")
    print("Training final model on all training data...")
    print(f"{'='*60}")

    X_all = torch.tensor(X)
    y_all = torch.tensor(y)
    all_loader = DataLoader(
        TensorDataset(X_all, y_all),
        batch_size=BATCH_SIZE, shuffle=True)

    # Dummy val loader (same as train — just for the training loop interface)
    final_model = FusionMLP(
        input_dim=14, hidden1=HIDDEN1, hidden2=HIDDEN2,
        num_classes=len(SHARED_EMOTIONS), dropout=DROPOUT)

    # Train for fixed epochs (no early stopping on final)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        final_model.train()
        total_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in all_loader:
            optimizer.zero_grad()
            logits = final_model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/total:.4f} "
                  f"acc={correct/total:.1%}")

    # Save
    os.makedirs(os.path.dirname(MODEL_OUTPUT) or '.', exist_ok=True)
    mlp_fusion = MLPFusion()
    mlp_fusion.model = final_model
    mlp_fusion.save(MODEL_OUTPUT)

    # Final train accuracy
    final_model.eval()
    with torch.no_grad():
        logits = final_model(X_all)
        train_acc = (logits.argmax(1) == y_all).sum().item() / len(y_all)
    print(f"\nFinal train accuracy: {train_acc:.1%}")
    print(f"Model saved to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
