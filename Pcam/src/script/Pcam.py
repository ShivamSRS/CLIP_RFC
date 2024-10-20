from torch.utils.data import Dataset
from torchvision import datasets
import h5py
import matplotlib as plt
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Pcam(Dataset):
    def __init__(self, path, transform=None, percent_data=1.0, seed=None):
        """
        Args:
            path (str): Base file path for the dataset (without '_x.h5' or '_y.h5').
            transform (callable, optional): Transform to be applied on an image.
            percent_data (float): Fraction of data to keep (0 < percent_data <= 1).
            seed (int, optional): Seed for random number generator to ensure reproducibility.
        """
        self.file_path = path
        self.dataset_x = None
        self.dataset_y = None
        self.transform = transform
        self.percent_data = percent_data  # Fraction of data to use
        self.seed = seed

        print(f"Initializing Pcam dataset with percent_data={self.percent_data*100} %")

        # Set the random seed for reproducibility if provided
        if self.seed is not None:
            np.random.seed(self.seed)

        # Read the dataset lengths
        with h5py.File(self.file_path + '_x.h5', 'r') as file_x:
            self.dataset_x_len = len(file_x['x'])

        with h5py.File(self.file_path + '_y.h5', 'r') as file_y:
            self.dataset_y_len = len(file_y['y'])

        assert self.dataset_x_len == self.dataset_y_len, "X and Y datasets have different lengths"

        # Validate percent_data
        if not 0 < self.percent_data <= 1:
            raise ValueError("percent_data must be between 0 and 1 (exclusive)")

        # Generate a list of indices to use based on percent_data
        self.all_indices = np.arange(self.dataset_x_len)
        # Randomly shuffle the indices
        np.random.shuffle(self.all_indices)
        # Calculate the number of samples to keep
        num_samples = int(self.percent_data * self.dataset_x_len)
        num_samples = max(num_samples, 1)  # Ensure at least one sample is selected
        # Keep only the first num_samples indices
        self.selected_indices = self.all_indices[:num_samples]

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        if self.dataset_x is None:
            self.dataset_x = h5py.File(self.file_path + '_x.h5', 'r')['x']
        if self.dataset_y is None:
            self.dataset_y = h5py.File(self.file_path + '_y.h5', 'r')['y']

        # Map the dataset index to the selected indices
        real_index = self.selected_indices[index]

        # Load image and label
        image = Image.fromarray(self.dataset_x[real_index])
        label = self.dataset_y[real_index].item()

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label, real_index

    def visualize(self, index):
        if self.dataset_x is None:
            self.dataset_x = h5py.File(self.file_path + '_x.h5', 'r')['x']
        if self.dataset_y is None:
            self.dataset_y = h5py.File(self.file_path + '_y.h5', 'r')['y']

        real_index = self.selected_indices[index]
        image = self.dataset_x[real_index]

        plt.imshow(image)
        plt.title(f"Sample Index: {index}, Real Index: {real_index}")
        plt.show()


# class Pcam(Dataset):
#     def __init__(self, path, transform=None):
#         self.file_path = path
#         self.dataset_x = None
#         self.dataset_y = None
#         self.transform = transform
#         print(self.file_path)

#         # Going to read the X part of the dataset - it's a different file
#         with h5py.File(self.file_path + '_x.h5', 'r') as file_x:
#             print(self.file_path)
#             self.dataset_x_len = len(file_x['x'])

#         # Going to read the y part of the dataset - it's a different file
#         with h5py.File(self.file_path + '_y.h5', 'r') as file_y:
#             self.dataset_y_len = len(file_y['y'])

#     def __getitem__(self, index):
#         if self.dataset_x is None:
#             self.dataset_x = h5py.File(self.file_path + '_x.h5', 'r')['x']
#         if self.dataset_y is None:
#             self.dataset_y = h5py.File(self.file_path + '_y.h5', 'r')['y']

#         image = Image.fromarray(self.dataset_x[index])
#         label = self.dataset_y[index].item()
#         if self.transform:
#             image = self.transform(image)
#         return image, label, index

#     def __len__(self):
#         assert self.dataset_x_len == self.dataset_y_len
#         return self.dataset_x_len

#     def visualize(self, index):
#         if self.dataset_x is None:
#             self.dataset_x = h5py.File(self.file_path + '_x.h5', 'r')['x']
#         if self.dataset_y is None:
#             self.dataset_y = h5py.File(self.file_path + '_y.h5', 'r')['y']
#         image = self.dataset_x[index]
#         #         if self.transform:
#         #             image = self.transform(image)
#         #         print('Label: ', self.dataset_y[index].item())
#         #         return self.dataset_y[index].item()
#         plt.imshow(image)
#         plt.show
