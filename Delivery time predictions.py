import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


os.chdir(r'C:\Users\Doru\Desktop\Pandas')

data=pd.read_csv('data_with_features.csv')
print(data.head())


def rush_hour_feature(hours_tensor, weekends_tensor):
    """
    Engineers a new binary feature indicating if a delivery is in a weekday rush hour.

    Args:
        hours_tensor (torch.Tensor): A tensor of delivery times of day.
        weekends_tensor (torch.Tensor): A tensor indicating if a delivery is on a weekend.

    Returns:
        torch.Tensor: A tensor of 0s and 1s indicating weekday rush hour.
    """

    is_morning_rush = (hours_tensor >= 8.0) & (hours_tensor < 10)
    is_evening_rush = (hours_tensor >= 16) & (hours_tensor < 19)
    is_weekday = (weekends_tensor == 0)

    is_rush_hour_mask = torch.tensor(is_morning_rush | is_evening_rush & is_weekday).float()

    return is_rush_hour_mask.float()

test_df = data.head(5).copy()

raw_test_tensor = torch.tensor(test_df.values, dtype=torch.float32)
raw_test_tensor = torch.tensor(test_df.values, dtype=torch.float32)



def prepare_data(df):
    """
    Converts a pandas DataFrame into prepared PyTorch tensors for modeling.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the raw delivery data.

    Returns:
        prepared_features (torch.Tensor): The final 2D feature tensor for the model.
        prepared_targets (torch.Tensor): The final 2D target tensor.
        results_dict (dict): A dictionary of intermediate tensors for testing purposes.
    """


    all_values = df.values
    full_tensor = torch.from_numpy(all_values).float()
    raw_distances = full_tensor[:, 0]
    raw_hours = full_tensor[:, 1]
    raw_weekends = full_tensor[:, 2]
    raw_targets = full_tensor[:, 3]
    is_rush_hour_feature = rush_hour_feature(raw_hours, raw_weekends)
    distances_col = raw_distances.unsqueeze(1)
    hours_col = raw_hours.unsqueeze(1)
    weekends_col = raw_weekends.unsqueeze(1)
    rush_hour_col = is_rush_hour_feature.unsqueeze(1)
    dist_mean, dist_std = distances_col.mean(), distances_col.std()
    hours_mean, hours_std = hours_col.mean(), hours_col.std()

    distances_norm = (distances_col - dist_mean) / dist_std
    hours_norm = (hours_col - hours_mean) / hours_std

    prepared_features = torch.cat([
        distances_norm,
        hours_norm,
        weekends_col,
        rush_hour_col
    ], dim=1)
    prepared_targets = raw_targets.unsqueeze(1)


    results_dict = {
        'full_tensor': full_tensor,
        'raw_distances': raw_distances,
        'raw_hours': raw_hours,
        'raw_weekends': raw_weekends,
        'raw_targets': raw_targets,
        'distances_col': distances_col,
        'hours_col': hours_col,
        'weekends_col': weekends_col,
        'rush_hour_col': rush_hour_col
    }

    return prepared_features, prepared_targets, results_dict


def init_model():
    """
    Initializes the neural network model, optimizer, and loss function.

    Returns:
        model (nn.Sequential): The initialized PyTorch sequential model.
        optimizer (torch.optim.Optimizer): The initialized optimizer for training.
        loss_function: The initialized loss function.
    """


    model = nn.Sequential(nn.Linear(4, 64),
                          nn.ReLU(),
                          nn.Linear(64, 32),
                          nn.ReLU(),
                          nn.Linear(32, 1),
                          )

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    loss_function = nn.MSELoss()

    return model, optimizer, loss_function


def train_model(features, targets, epochs, verbose=True):
    """
    Trains the model using the provided data for a number of epochs.

    Args:
        features (torch.Tensor): The input features for training.
        targets (torch.Tensor): The target values for training.
        epochs (int): The number of training epochs.
        verbose (bool): If True, prints training progress. Defaults to True.

    Returns:
        model (nn.Sequential): The trained model.
        losses (list): A list of loss values recorded every 5000 epochs.
    """
    losses = []

    model, optimizer, loss_function = init_model()

    for epoch in range(epochs):
        outputs = model(features)
        loss = loss_function(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5000 == 0:
            losses.append(loss.item())
            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model, losses






