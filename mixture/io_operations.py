import os.path
import pickle
from abc import ABC, abstractmethod


class DataLoader(ABC):

    @abstractmethod
    def load_object(self, file_path):
        pass

    @abstractmethod
    def load_objects(self, file_path, no_objects):
        pass


class DataSaver(ABC):

    @abstractmethod
    def save_object(self, file_path):
        pass

    @abstractmethod
    def save_objects(self, file_path, objects):
        pass


class PickleDataLoader(DataLoader):

    def load_object(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def load_objects(self, file_path, no_objects):

        """
        :param file_path: where to save the object to
        :param no_objects: how many objects to save
        :return: the loaded objects
        """

        params_list = [None for _ in range(no_objects)]
        with open(file_path, 'rb') as f:
            saved_data = pickle.load(f)
            for i in range(no_objects):
                params_list[i] = saved_data[i]

        return params_list


class PickleDataSaver(DataSaver):

    def save_object(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(f)

    def save_objects(self, file_path, params_list):
        with open(file_path, 'wb') as f:
            pickle.dump({str(i): params_list[i] for i in range(len(params_list))}, f)

        return params_list

# Loader and saver ??
class Loader:

    def __init__(self, file_path, r=False, w=False):
        self.file_path = file_path
        self.r = r
        self.w = w

    def read(self):
        pass

    def write(self):
        pass


