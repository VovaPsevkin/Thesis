from os import getcwd
import sys
from pathlib import Path
import time
import csv
import argparse
import traceback

from scipy import stats, special
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from sklearn import metrics
import pickle
from tqdm import tqdm
from colorama import Fore, Style
from joblib import load
from RegularEmerson.preprocessing import Preprocessing


class EmersonSecondVersion(Preprocessing):
    """This class implement Ryan Emerson's article.
    implement step by step statistical algorithm with beta distribution.
    The goal is to classify a compositions and labels and compute auc score.
    ========================================================================

    Attributes:
        combinations (dict): The storage dictionary which data is being grouped.
        cmv (numpy.array): An array of patient tags that contains zero or one

    """
    __slots__ = ('filtered_p_value_combinations', 'cmv', 'label_combinations', 'negative_train', 'positive_train',
                 'test_patient_info', 'test_patient_pos_neg', 'negative_test', 'positive_test', 'cmv_test',
                 'train_auc', 'test_auc', 'n_jobs', 'p_value', 'N0', 'N1', 'c_scalar', 'fpr', 'tpr')

    # global variable static - All the time the code is running
    # Retrieved from Emerson's article
    alpha1, alpha0 = 4.0, 26.7
    beta1, beta0 = 51820, 2814963.8

    def __init__(self, n_jobs, p_value):
        super().__init__()

        self.n_jobs = n_jobs
        self.p_value = p_value
        self.filtered_p_value_combinations = {}
        self.label_combinations = {}
        self.negative_train = self.positive_train = None
        self.test_patient_info = {}
        self.test_patient_pos_neg = {}
        self.negative_test, self.positive_test, = [], []
        self.cmv_test = None
        self.train_auc = self.test_auc = None
        self.N0 = self.N1 = 0
        self.c_scalar = 0
        self.fpr, self.tpr = 0, 0

    @staticmethod
    def check_p_value(param1, param2, p_value):
        """
         despacher function that calculate fisher extract test
         create matrix and compute pvalue by the book
         condition less p_value argument

        Args:
          directory (str): Path to folder where all files are located.

        """
        first = np.sum((param1 == 0) & (param2[1] == 0))
        second = np.sum((param1 == 0) & (param2[1] == 1))
        third = np.sum((param1 == 1) & (param2[1] == 0))
        fourth = np.sum((param1 == 1) & (param2[1] == 1))
        # TODO:  alternative = 'greater'
        _, pvalue = stats.fisher_exact([[first, second], [third, fourth]])
        if pvalue <= p_value:
            return param2[0], param2[1]

    @staticmethod
    def fishers_exact_test_sort(combinations, cmv, jobs, p_value):
        # todo: documentations describe about function
        print(
            f"{Fore.LIGHTBLUE_EX}Filter dictionary by condition of {Fore.LIGHTMAGENTA_EX}p value,{Fore.RED} Fisher's exact test{Fore.RESET}")
        start_time = time.time()
        temp = Parallel(n_jobs=jobs)(
            delayed(EmersonSecondVersion.check_p_value)(cmv, item, p_value) for item in
            tqdm(combinations.items(), desc="Multiprocessing filter dictionary by p value",
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)))
        temp = list(filter(None, temp))

        filtered_p_value_combinations = dict(temp)

        print(
            f"{Fore.LIGHTBLUE_EX}The amount of combination after the filtering operation: {Fore.RED}{len(filtered_p_value_combinations)}{Fore.RESET}")
        print(
            f"{Fore.LIGHTBLUE_EX}Filter dictionary by condition of p value, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")

        # with open('filtered_p_value_emerson.pickle', 'wb') as handle:
        #     pickle.dump(filtered_p_value_combinations, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return filtered_p_value_combinations

    def determine_label(self):
        # TODO: describe function
        """
        To train
        Math formula   a (a + b + c + d) / (a + b) * (a + c)
        """

        for key in tqdm(self.filtered_p_value_combinations, desc='Determine a cmv according to a mathematical formula',
                        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET),
                        total=len(self.filtered_p_value_combinations)):
            d = np.sum((self.cmv == 0) & (self.filtered_p_value_combinations[key] == 0))
            c = np.sum((self.cmv == 0) & (self.filtered_p_value_combinations[key] == 1))
            b = np.sum((self.cmv == 1) & (self.filtered_p_value_combinations[key] == 0))
            a = np.sum((self.cmv == 1) & (self.filtered_p_value_combinations[key] == 1))

            numerator = a * (a + b + c + d)
            denominator = (a + b) * (a + c)
            # Fixme: if else conditions shourtcut
            if numerator / denominator > 1:
                self.label_combinations[key] = 1
            else:
                self.label_combinations[key] = 0

    def count_of_pos_neg(self):
        # TODO: describe function
        # FIXME: array and lists
        temp = []
        for key in tqdm(self.label_combinations, desc="Count the number of group combinations impressions in Test",
                        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET),
                        total=len(self.label_combinations)):
            item = np.concatenate((self.filtered_p_value_combinations[key], [self.label_combinations[key]]))
            temp.append(item)

        arr = np.array(temp)
        brr = arr[arr[:, -1] == 0].sum(0)
        crr = arr[arr[:, -1] == 1].sum(0)

        self.negative_train, self.positive_train = brr[:-1], crr[:-1]

    def prepare_combinations(self, data_path):
        # TODO: define docstring
        """

        """
        directory = Path(data_path)
        size = len(list(directory.glob('*')))

        patient_dict = {}
        for path in tqdm(directory.glob('*'), total=size, desc='Create combinations of test data',
                         bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            patient_name, _, label = path.stem.split('_')
            data = pd.read_csv(path, usecols=[2], dtype={'combined': '|S45'})['combined']
            patient_dict[(patient_name, label, path.as_posix())] = data.tolist()

        for key in tqdm(patient_dict, total=size,
                        desc='Filter combinations of test data by spesific p value that calculated in train combinations',
                        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            filter_list = []
            value = patient_dict[key]
            for element in value:
                if element in self.filtered_p_value_combinations:
                    filter_list.append(element)
            self.test_patient_info[key] = list(set(filter_list))
        print("Combinations after filtering in 154 clusters", self.test_patient_info)

    def data_preparation(self):

        for i, key in enumerate(self.test_patient_info):
            temp = [self.label_combinations[value] for value in self.test_patient_info[key]]
            self.test_patient_pos_neg[key] = np.array(temp)

        self.cmv_test = np.zeros(len(self.test_patient_pos_neg), dtype=np.int8)
        for ind, key1 in enumerate(self.test_patient_pos_neg):
            if key1[1] == 'positive':
                self.cmv_test[ind] = 1

            self.negative_test.append(sum(self.test_patient_pos_neg[key1] == 0))
            self.positive_test.append(sum(self.test_patient_pos_neg[key1] == 1))

        self.negative_test, self.positive_test = np.array(self.negative_test, dtype=np.int16), np.array(
            self.positive_test, dtype=np.int16)

    def count_positive_negative_patients(self):
        self.N0 = sum(self.cmv == 0)
        self.N1 = sum(self.cmv == 1)

    def compute_scalar(self):
        self.c_scalar = np.log(1 + self.N1) - np.log(1 + self.N0) + np.log(
            special.beta(EmersonSecondVersion.alpha0, EmersonSecondVersion.beta0)) - np.log(
            special.beta(EmersonSecondVersion.alpha1, EmersonSecondVersion.beta1))

    def fit(self, data_path):
        try:
            # TODO: describe functions
            self.activate_mechanism(data_path)

            self.filtered_p_value_combinations = self.fishers_exact_test_sort(self.combinations, \
                                                                              self.cmv, self.n_jobs, self.p_value)

            self.count_positive_negative_patients()
            self.determine_label()

            if len(self.label_combinations) == 0:
                raise ValueError(
                    f"Not found combinations between patients that passed p value Fisher\'s test, empty dictionary: {self.label_combinations}")

            self.count_of_pos_neg()
            self.compute_scalar()


        except ValueError as e:
            _, compositions = Path(data_path).stem.split('_')
            write2file(compositions, auc=0, p_value=self.p_value, comment=f"{e}")
            sys.exit(f"{Fore.YELLOW}{e}{Fore.RESET}")

    def predict(self, data_path):
        self.prepare_combinations(data_path)
        self.data_preparation()

        total_pos_and_neg_per_person = self.negative_test + self.positive_test

        scores = np.zeros(total_pos_and_neg_per_person.shape, dtype=np.float64)
        y_pred = np.zeros(total_pos_and_neg_per_person.shape, dtype=np.int8)
        for i in tqdm(range(len(total_pos_and_neg_per_person)), desc="Prediction Score",
                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            k_, n_ = self.positive_test[i], total_pos_and_neg_per_person[i]
            # c_scalar = -10
            statement = self.c_scalar + np.log(
                special.beta(k_ + EmersonSecondVersion.alpha1, n_ - k_ + EmersonSecondVersion.beta1)) - np.log(
                special.beta(k_ + EmersonSecondVersion.alpha0, n_ - k_ + EmersonSecondVersion.beta0))
            scores[i] = statement
            if statement > 0:
                y_pred[i] = 1

        self.fpr, self.tpr, thresholds = metrics.roc_curve(self.cmv_test, scores)
        self.test_auc = metrics.auc(self.fpr, self.tpr)


def roc_curve_visualization(obj):
    sns.set_context("poster")
    fig = plt.figure(figsize=(10, 10))
    sns.lineplot(obj.fpr, obj.tpr, label='AUC = %0.3f' % obj.test_auc, estimator=None, lw=1)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tick_params(top=None, bottom='off', left='off', right=None, labelleft='off', labelbottom='on')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    image_path = Path(
        '/home/vovaps/Projects/EmersonSoftwareFramework/roc_curve_emerson_400_first.png').absolute()
    plt.savefig(image_path.as_posix())
    # fig.savefig(image_path, dpi=fig.dpi)


def write2file(compositions, auc, p_value, comment='', name=f'Amount of combinations and result.txt'):
    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["{}".format(compositions), "{}".format(auc), "{}".format(p_value), "{}".format(comment)])


if __name__ == '__main__':
    try:

        # create default paths to pickle's
        ROOT = Path(getcwd()).parent
        # ROOT = Path('/home/vovaps/Projects/EmersonSoftwareFramework')
        DATA = ROOT / 'Data'
        TRAIN = DATA / 'PreprocessedTrain_641'
        TEST = DATA / 'PreprocessedTest'

        parser = argparse.ArgumentParser(
            description=f"""python3 emerson_version_one.py \
            --train_data_path=/home/vovaps/Projects/EmersonSoftwareFramework/Data/PreprocessedTrain/ \
            --test_data_path=/home/vovaps/Projects/EmersonSoftwareFramework/Data/PreprocessedTest/ \
            --n_jobs=35 \
            --p_value=0.0001""")

        parser.add_argument("--train_data_path", type=str, default=TRAIN.as_posix(), nargs='?',
                            help="Input train directory path of files with all patients csv\'s")

        parser.add_argument("--test_data_path", type=str, default=TEST.as_posix(), nargs='?',
                            help="Input test directory path of files with all patients csv\'s")

        parser.add_argument("--n_jobs", type=int, default=45, nargs='?',
                            help="Amount of processes corresponding to Fisher exact test")

        parser.add_argument("--p_value", type=float, default=1E-4, nargs='?',
                            help="The probability of getting a result is the same as the sample \
                             statistics obtained, to Fisher exact test")

        args = parser.parse_args()

        emerson_v2 = EmersonSecondVersion(args.n_jobs, args.p_value)
        emerson_v2.fit(args.train_data_path)
        emerson_v2.predict(args.test_data_path)

        print(f"{Fore.LIGHTBLUE_EX}Test AUC score: {Fore.RED}{emerson_v2.test_auc}{Fore.RESET}")

        _, compositions = Path(args.train_data_path).stem.split('_')
        write2file(compositions, auc=emerson_v2.test_auc, p_value=args.p_value, comment=f"Work without Exceptions")

        roc_curve_visualization(emerson_v2)
        print(Style.RESET_ALL)

    except Exception as e:
        print(e)
        print(traceback.format_exc())
