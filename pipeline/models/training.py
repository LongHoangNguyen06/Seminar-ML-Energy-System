import os
import pickle

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR as Scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from pipeline.models.dataset import TimeSeriesDataset
from pipeline.models.transformer import TimeSeriesTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training loop
def train(model, train_loader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    scheduler.step()  # Update the learning rate
    progress_bar.close()
    return total_loss / len(train_loader)


# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    progress_bar.close()
    return total_loss / len(val_loader)


def train_loop(
    hyperparameters, df, train_id, merge_train_val: bool = False, log_wandb: bool = True
):
    """
    Main training loop for the TimeSeriesTransformer model.
    Args:
    hyperparameters : DotMap
        Configuration object.
    df : pd.DataFrame
        Dataframe containing the time series data.
    train_id : int
        Unique identifier for the training run.
    merge_train_val : bool
        Whether to merge the training and validation sets.
    Returns:
        best_val_loss : float
            Best validation loss achieved during training.
    """
    experiment_path = os.path.join(hyperparameters.model.save_path, f"run_{train_id}")
    model_path = os.path.join(experiment_path, "model.pth")
    hyperparameters_path = os.path.join(experiment_path, "hyperparameters.pth")
    os.makedirs(experiment_path, exist_ok=True)
    pickle.dump(hyperparameters, open(hyperparameters_path, "wb"))
    # Initialize data
    if merge_train_val:
        train_df = df[df["train"] | df["val"]].reset_index(drop=True)
        val_df = df[df["val"]].reset_index(drop=True)
    else:
        train_df = df[df["train"]].reset_index(drop=True)
        val_df = df[df["val"]].reset_index(drop=True)

    train_dataset = TimeSeriesDataset(train_df, hyperparameters=hyperparameters)
    val_dataset = TimeSeriesDataset(val_df, hyperparameters=hyperparameters)

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparameters.train.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=hyperparameters.train.batch_size, shuffle=False
    )

    # Initialize model
    model = TimeSeriesTransformer(hyperparameters=hyperparameters).to(device)
    optimizer = Adam(model.parameters(), lr=hyperparameters.train.lr)
    criterion = hyperparameters.train.loss()

    scheduler = Scheduler(
        optimizer,
        T_max=hyperparameters.train.epochs,
        eta_min=hyperparameters.train.min_lr,
    )

    # Main training loop
    num_epochs = hyperparameters.train.epochs
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, scheduler)
        val_loss = validate(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

        if log_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                }
            )
        else:
            print("wandb is not initialized, skipping log.")
    return best_val_loss
