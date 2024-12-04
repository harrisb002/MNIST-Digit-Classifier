# dataset.py

import csv
import numpy as np
import os

class DatasetLoader:
    """
    Class for loading and handling datasets.
    """

    @staticmethod
    def read_file(file_path):
        """
        Reads a CSV file and returns the data.

        Params:
        - file_path (str): Path to the CSV file.

        Returns:
        - list: List of rows from the CSV file.
        """
        dataset = []
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                dataset.append(row)
        return dataset

    @staticmethod
    def get_images_and_labels(dataset):
        """
        Processes the dataset and extracts images and labels.

        Params:
        - dataset (list): List of rows from the CSV file.

        Returns:
        - list: List of (label, image) tuples.
        """
        data = []
        for row in dataset:
            label = int(row[0])
            image = np.array(row[1:], dtype=np.uint8).reshape(28, 28)
            data.append((label, image))
        return data

    @staticmethod
    def extract_folder(directory):
        """
        Reads all CSV files in a directory and returns combined data.

        Params:
        - directory (str): Path to the directory.

        Returns:
        - list: Combined dataset from all files.
        """
        filenames = sorted(os.listdir(directory))
        data_files = []
        for file in filenames:
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and file.endswith('.csv'):
                data_files.append(file_path)

        dataset = []
        for file in data_files:
            file_data = DatasetLoader.read_file(file)
            dataset.extend(DatasetLoader.get_images_and_labels(file_data))
        return dataset
