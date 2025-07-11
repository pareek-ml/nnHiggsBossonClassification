#!/usr/bin/env python
"""
@File    :   train.py
@Modified:   2025/07/10 22:28:34
@Author  :   Pareek-Yash
@License :   MIT
@Version :   1.0
@Desc    :   None
"""
# -*- coding: utf-8 -*-
import time

import numpy as np
import rich
import torch
import typer
from openml import datasets
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, random_split

from logger import logger
from model import HiggsNet


def getdevice():
    """
    Get the device to use for training.
    Returns:
        torch.device: The device to use (CUDA if available, MPS if Apple Silicon, otherwise CPU).
    """
    if torch.backends.mps.is_available():
        rich.print("[bold green]Using MPS (Apple Silicon) for training[/bold green]")
        return torch.device("mps")
    elif torch.cuda.is_available():
        rich.print("[bold green]Using CUDA for training[/bold green]")
        return torch.device("cuda")
    else:
        rich.print("[bold green]Using CPU for training[/bold green]")
        return torch.device("cpu")


def load_higgs(openml_id=44129, scaler=None):
    """
    Load the Higgs Boson dataset from OpenML and preprocess it.
    Args:
        openml_id (int): The OpenML dataset ID for the Higgs Boson dataset.
        scaler (Optional[StandardScaler]): An optional scaler to use for feature scaling.

    Returns:
        Tuple[TensorDataset, StandardScaler]: The preprocessed dataset and the scaler used.
    """
    raw_dataset = datasets.get_dataset(openml_id, download_data=True)
    dataset = raw_dataset.get_data()
    scaler = scaler or StandardScaler()
    inputs = scaler.fit_transform(
        dataset[0].drop(columns=["target"]).astype(np.float32)
    )
    y = dataset[0]["target"].astype(np.float32)  # Convert target to float32
    tensor_ds = TensorDataset(torch.tensor(inputs), torch.tensor(y))
    return tensor_ds, scaler


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    scaler,
    device,
    accumulation_steps=1,
    scheduler=None,
    use_amp=False,
):
    """
    Train the model for one epoch.
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (nn.Module): The loss function to use.
        scaler (GradScaler): The gradient scaler for mixed precision training.
        device (torch.device): The device to use for training.
        accumulation_steps (int): Number of steps to accumulate gradients before updating.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        use_amp (bool): Whether to use automatic mixed precision (AMP) for training.
    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        with autocast(device_type="mps", enabled=use_amp):
            logits = model(inputs)
            loss = (
                criterion(logits, targets) / accumulation_steps
            )  # Scale loss for accumulation

        (scaler.scale(loss) if use_amp else loss).backward()
        if (i + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler:
                scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    """
    Evaluate the model on the given data loader.
    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for the evaluation data.
        device (torch.device): The device to use for evaluation.
        use_amp (bool): Whether to use automatic mixed precision (AMP) for evaluation.
    Returns:
        Tuple[float, float]: The ROC AUC score and average precision score.
    """
    model.eval()
    y_true, y_prob = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast(device_type="mps", enabled=use_amp):
            logits = model(xb)

        y_true.append(yb)  # keep on CPU
        y_prob.append(torch.sigmoid(logits).cpu())

    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()
    return (roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob))


def train_model(
    lr: float = 0.001,
    epochs: int = 10,
    model_pth: str = "./model.pt",
    pkl_path: str = "./data.pkl",
):
    """
    Train the Higgs Boson classification model.
    Args:
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        model_pth (str): Path to save the trained model.
        pkl_path (str): Path to save the dataset as a pickle file.
    Returns:
        None
    """
    tensor_ds, scaler = load_higgs()
    generator1 = torch.Generator().manual_seed(42)
    train, test = random_split(tensor_ds, [0.8, 0.2], generator=generator1)
    train_loader = DataLoader(train, batch_size=8192, shuffle=True, num_workers=4)
    test_loader = DataLoader(test, batch_size=8192, shuffle=False, num_workers=4)
    data_batch, labels_batch = next(iter(train_loader))
    rich.print(
        f"[blue]Data batch shape: {data_batch.shape}, Labels batch shape: {labels_batch.shape}[/blue]"
    )
    rich.print(
        f"[green]Number of training samples: {len(train)}, Number of test samples: {len(test)}[/green]"
    )

    pos_weight = (len(train) - train.dataset.tensors[1].sum()) / train.dataset.tensors[
        1
    ].sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    device = getdevice()
    model = HiggsNet(
        input_dim=train.dataset.tensors[0].shape[1],
        hidden_dim=512,
        num_layers=6,
        dropout=0.2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="linear",
    )
    scaler = GradScaler()
    best_auc, best_epoch = 0.0, -1
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            accumulation_steps=1,
            scheduler=scheduler,
            use_amp=True,
        )
        val_auc, val_ap = evaluate(model, test_loader, device, use_amp=True)
        t1 = time.time()
        dt = t1 - t0
        logger.info(
            f"Epoch {epoch:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val AP: {val_ap:.4f} | "
            f"Time: {dt:.2f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
        if epoch - best_epoch > 3:
            rich.print("[yellow]Early stopping...[/yellow]")
            break

    rich.print(f"\n[green]Best val AUC: {best_auc:.4f}  (epoch {best_epoch})[/green]")

    rich.print(f"[blue]Saving scaler to {pkl_path}[/blue]")
    torch.save(scaler, pkl_path)
    rich.print(f"[blue]Saving model to {model_pth}[/blue]")
    torch.save(model.state_dict(), model_pth)


def main(
    debug: bool = typer.Option(
        False, "--debug", "-d", is_flag=True, help="Enable debug mode"
    ),
    lr: float = typer.Option(1e-3, "--learning-rate", "-lr", help="Learning rate"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    model_pth: str = typer.Option(
        "../models/model.pt", "--model-path", "-mp", help="Path to save/load model"
    ),
    pkl_path: str = typer.Option(
        "../models/data.pkl", "--pickle-path", "-pp", help="Path to pickle file"
    ),
):
    """A CLI application that accepts various parameters for training a machine learning model."""
    if debug:
        rich.print(
            "[bold red]Debug mode is enabled! Remove -d flag for production[/bold red]"
        )
        logger.debug("Debug mode is enabled")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Number of epochs: {epochs}")
    logger.info(f"Model path: {model_pth}")
    logger.info(f"Pickle path: {pkl_path}")
    train_model(lr=lr, epochs=epochs, model_pth=model_pth, pkl_path=pkl_path)


if __name__ == "__main__":
    typer.run(main)
