import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.vision.datasets.folder import ImageFolder

def get_dataloaders(
            train_dir,
            var_dir,
            train_transform = None,
            val_transform = None,
            split = (0.5, 0.5),
            batch_size = 32):
    """
    This function returns the train, val and test dataloaders:
    """
    # create the datasets
    train_ds = ImageFolder(root = train_dir, transform = train_transform)
    val_ds   = ImageFolder(root = val_dir, transform = val_transform)
    # now we want to split the val_ds in validation and test
    lengths = np.array(split) * len(val_ds)
    lengths = lengths.astype(int)
    left = len(val_ds) - lengths.sum()
    # we need to add the different due to float approx to int
    lenghts[-1] += left
    
    val_ds, test_ds = random_split(val_ds, lengths.tolist())
    
    logging.info(f'{:10}{:10}'.format("Dataset", "N. Samples"))
    logging.info(f'{:10}={:10}'.format("Train", len(train_ds)))
    logging.info(f'{:10}={:10}'.format("Validation", len(val_ds)))
    logging.info(f'{:10}={:10}'.format("Test", len(test_ds)))
    
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    val_dl   = DataLoader(val_ds, batch_size = batch_size, shuffle = False)
    test_dl  = DataLoader(test_ds, batch_size = batch_size, shuffle = False)
    
    return train_dl, val_dl, test_dl
    
    