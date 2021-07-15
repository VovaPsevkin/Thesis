import os
import time
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from colorama import Fore

from memory_profiler import profile


class Preprocessing(object):
    """This class implement data preprocessing given a folder.
    Reads combinations into a large dictionary object.
    Filter the dictionary and create a new one based on the combination
     of impressions in at least two people.
    Reads the diagnosis from a file, Positive or Negative (CMV).

    The goal is to create a dictionary and labels that the statistical model will use
    ===================================================================================

    Attributes:
        combinations (dict): The storage dictionary which data is being grouped.
        id2patient (dict): The storage dictionary containing patient information and serial number.
        size (int): Total number of patients, amount of files in a folder.
        cmv (numpy.array): An array of patient tags that contains zero or one

    """

    # Set class variables to economize storage, economize memory
    __slots__ = ('combinations', '__id2patient', '__size', 'cmv')

    def __init__(self):
        """Constructor initialize all Args by default values

        """

        self.combinations = {}
        self.__id2patient = {}
        self.__size = None
        self.cmv = None

    def __personal_information(self, directory):
        """Creating a dictionary that retains patient information allows you
        to maintain order by providing an index, which is a patient ID within the array
        Creates the id2patient dictionary and size,

         Args:
          directory (str): Path to folder where all files are located.

        After this methods id2patient look like: {(name, label, path_of_file): index}
                           size look like: int number

        """
        #assert directory == '/home/vovaps/Projects/EmersonSoftwareFramework/Data/PreprocessedTrain/', f"{Fore.RED}Not same directory{Fore.RESET}"
        directory = Path(directory)
        self.__size = len(list(directory.glob('*')))
        for index, item in tqdm(enumerate(directory.glob('*')), total=self.__size, desc="Maintain patient order",
                                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            attr = item.stem.split('_')
            attr.append(item.as_posix())
            self.__id2patient[tuple(attr)] = index

    def __create_combinations(self):
        """The dictionary to which data is loaded from all files contains all combinations,
        reads files from another dictionary.
        Initializes each element in the dictionary by key (combination) and the value by  zeros array
        in size of the patient.
        A key word already appears in the dictionary, so one patient's index is gathered.

        """
        print(f"\n{Fore.LIGHTBLUE_EX}Generate a quantity of {Fore.LIGHTMAGENTA_EX}instances combination{Fore.RESET}")
        start_time = time.time()


        for personal_info, ind in tqdm(self.__id2patient.items(), total=self.__size, desc='Create Combinations',
                                       bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            _, _, path = personal_info
            for element in pd.read_csv(path, usecols=[2], dtype={'combined': '|S45'})['combined']:
                if element not in self.combinations:
                    self.combinations[element] = np.zeros(self.__size, dtype=np.int8)
                # TODO: bag of words, I change to equal 1 instead of count amount of impressions
                #  per person (combinations[element][item[1]] += 1) in order to skip the binary dictionary step
                #  A place to combine "embedding layer"
                self.combinations[element][ind] = 1
        print(f"{Fore.LIGHTBLUE_EX}Amount of combinations: {Fore.RED}{len(self.combinations)}{Fore.RESET}")
        print(
            f"{Fore.LIGHTBLUE_EX}Generate a quantity of instances combinations, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")

    def __filter_dict(self):
        """Initial filtering of combinations dictionary, filtering is done by the number of
        impressions of a key if it is less than two The combination comes out of a dictionary.
        """
        print(
            f"{Fore.LIGHTBLUE_EX}Filter dictionary by combining multiple instances, {Fore.LIGHTMAGENTA_EX}at least two{Fore.RESET}")
        start_time = time.time()
        for k in tqdm(list(self.combinations), total=len(self.combinations), desc='Filter dictionary',
                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            if np.count_nonzero(self.combinations[k]) < 2:
                del self.combinations[k]
        print(
            f"{Fore.LIGHTBLUE_EX}The amount of combination after the filtering operation: {Fore.RED}{len(self.combinations)}{Fore.RESET}")

        print(
            f"{Fore.LIGHTBLUE_EX}Filter dictionary by combining multiple instances at least two, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")
        #
        # print(
        #     f"{Fore.LIGHTBLUE_EX}Save filtered dictionary, {Fore.LIGHTMAGENTA_EX}at least two{Fore.RESET}")
        #
        # start_time = time.time()
        # directory = 'objects'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # name = ''.join(['filtred_more_twenty', '.pkl'])
        # path = os.path.join(directory, name)
        # with open(path, 'wb') as f:
        #     pickle.dump(self.combinations, f, pickle.HIGHEST_PROTOCOL)
        #
        # print(
        #     f"{Fore.LIGHTBLUE_EX}Save filtered dictionary, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")

    def __create_cmv_vector(self):
        """"Create a vector of tags זzero is negative and one is positive
        """
        # FIXME the loop and condition
        self.cmv = np.zeros(self.__size, dtype=np.int8)
        for item, index in self.__id2patient.items():
            if item[1] == 'positive':
                self.cmv[index] = 1

    def activate_mechanism(self, directory):
        """A central function that activates the step-by-step mechanism
        """
        # TODO: Maybe convert all methods to private
        self.__personal_information(directory)
        self.__create_cmv_vector()
        self.__create_combinations()
        self.__filter_dict()
