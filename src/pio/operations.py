import os.path
import pickle
from abc import ABC, abstractmethod


class DataLoader(ABC):

    @abstractmethod
    def read_object(self, file_path, no_objects):
        pass


class DataSaver(ABC):

    @abstractmethod
    def write_object(self, file_path, objects):
        pass


class PickleReader(DataLoader):

    def read_object(self, file_path, no_objects):

        """
        :param file_path: where to save the object to
        :param no_objects: how many objects to save
        :return: the loaded objects
        """

        if no_objects < 1:
            raise FileNotFoundError("Pass at least one object to be read")

        if no_objects == 1:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data
        else:
            params_list = []
            with open(file_path, 'rb') as f:
                saved_data = pickle.load(f)
                for value in saved_data.values():
                    params_list.append(value)

                return params_list


class PickleWriter(DataSaver):

    def write_object(self, file_path, data):

        if data is None:
            raise FileNotFoundError("Pass at least one object to be written")

        if len(data) == 1:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump({str(i): data[i] for i in range(len(data))}, f)


class Loader:

    def __init__(self):
        self.writer = PickleWriter()
        self.reader = PickleReader()

    def fully_load(self, file_path, function, no_outputs, *params_list):

        """
        :param file_path: where to save the file
        :param function: the function that processes the params_list argument
        :param no_outputs: expected outputs
        :param params_list: parameters of the function
        :return: the result of the function if the path doesn't exist, otherwise the saved file
        """

        if not os.path.exists(file_path):
            print(f"Computing and saving..")
            data = function(*params_list)  # returns a tuple
            print(f"data = {data}")
            self.write(file_path, data)
            return data
        else:
            print(f"Only loading")
            return self.read(file_path, no_outputs)

    def read(self, file_path, no_outputs):
        return self.reader.read_object(file_path, no_outputs)

    def write(self, file_path, data):
        self.writer.write_object(file_path, data)