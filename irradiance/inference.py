import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def ipredict(model, dataset, return_images=False, batch_size=2, num_workers=None):
    """Predict irradiance for a given set of npy image stacks using a generator.

    Parameters
    ----------
    chk_path: model save point.
    dataset: pytorch dataset for streaming the input data.
    return_images: set True to return input images.
    batch_size: number of samples to process in parallel.
    num_workers: number of workers for data preprocessing (default cpu_count / 2).

    Returns
    -------
    predicted irradiance as numpy array and corresponding image if return_images==True
    """
    # use model after training or load weights and drop into the production system
    model.eval()
    # load data
    num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(model.device)
            pred_irradiance = model.forward_unnormalize(imgs)
            for pred, img in zip(pred_irradiance, imgs):
                if return_images:
                    yield pred.cpu(), img.cpu()
                else:
                    yield pred.cpu()

def ipredict_uncertainty(model, dataset, return_images=False, forward_passes=100, batch_size=2, num_workers = None):
    """Predict irradiance and uncertainty for a given set of npy image stacks using a generator.

    Parameters
    ----------
    chk_path: model save point.
    dataset: pytorch dataset for streaming the input data.
    return_images: set True to return input images.
    batch_size: number of samples to process in parallel.
    num_workers: number of workers for data preprocessing (default cpu_count / 2).

    Returns
    -------
    predicted irradiance as numpy array and corresponding image if return_images==True
    """
    # load data
    num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(model.device)
            dropout_predictions = []
            for _ in range(forward_passes):
                model.eval()
                enable_dropout(model)
                pred_irradiance = model.forward_unnormalize(imgs)
                dropout_predictions += [pred_irradiance.cpu()]

            # shape (forward_passes, n_samples, n_classes)
            dropout_predictions = torch.stack(dropout_predictions).numpy()

            mean_batch = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

            # Calculating std across multiple MCD forward passes
            std_batch = np.std(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

            epsilon = sys.float_info.min
            # Calculating entropy across multiple MCD forward passes
            entropy = -np.sum(mean_batch * np.log(mean_batch + epsilon), axis=-1)  # shape (n_samples,)

            # Calculating mutual information across multiple MCD forward passes
            mutual_info_batch = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                                  axis=-1), axis=0)  # shape (n_samples,)

            # iterate through batch
            for img, mean, std, mutual_info in zip(imgs, mean_batch, std_batch, mutual_info_batch):
                if return_images:
                    yield (mean, std, mutual_info), img.cpu().numpy()
                else:
                    yield (mean, std, mutual_info)


def ipredict_ensembles(models, dataset, return_images=False, batch_size=2, num_workers = None):
    """Predict irradiance and uncertainty for a given set of npy image stacks using a generator using ensembles.

    Parameters
    ----------
    chk_paths: list of models save points.
    dataset: pytorch dataset for streaming the input data.
    return_images: set True to return input images.
    batch_size: number of samples to process in parallel.
    num_workers: number of workers for data preprocessing (default cpu_count / 2).

    Returns
    -------
    predicted irradiance as numpy array and corresponding image if return_images==True
    """
     # load data
    num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(model.device)
            ensemble_predictions = []
            for model in models:
                # use model after training or load weights and drop into the production system
                model.eval()
                # pred_irradiance = model.forward(imgs)
                pred_irradiance = model.forward_unnormalize(imgs)
                ensemble_predictions += [pred_irradiance.cpu()]   

            ensemble_predictions = torch.stack(ensemble_predictions).numpy() # shape (forward_passes, n_samples, n_classes)
            ensemble_predictions = np.mean(ensemble_predictions, axis=0)  # shape (n_samples, n_classes)

            # Calculating variance across multiple MCD forward passes
            ensemble_uncertainty = np.std(ensemble_predictions, axis=0)  # shape (n_samples, n_classes)

            # iterate through batch
            for img, mean, variance in zip(imgs, ensemble_predictions, ensemble_uncertainty):
                if return_images:
                    yield (mean, variance), img.cpu().numpy()
                else:
                    yield (mean, variance)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
