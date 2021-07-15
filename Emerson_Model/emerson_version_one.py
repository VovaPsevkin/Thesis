import argparse
import time
import traceback
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from colorama import Fore, Back, Style
from joblib import Parallel, delayed
from os import getcwd
from scipy import stats
from scipy.optimize import minimize
from sklearn import metrics
from tqdm import tqdm

from RegularEmerson.preprocessing import Preprocessing


class EmersonFirstVersion(Preprocessing):
    # todo: documentations describe about class
    __slots__ = ('filtered_p_value_combinations', 'cmv', 'label_combinations', 'negative_train', 'positive_train',
                 'test_patient_info', 'test_patient_pos_neg', 'negative_test', 'positive_test', 'cmv_test', 'A', 'B',
                 'train_auc', 'test_auc', 'n_jobs', 'p_value')

    def __init__(self, n_jobs, p_value):
        super().__init__()

        self.n_jobs = n_jobs
        self.p_value = p_value
        self.filtered_p_value_combinations = {}
        self.label_combinations = {}
        self.negative_train = self.positive_train = None
        self.test_patient_info = {}
        self.test_patient_pos_neg = {}
        self.negative_test, self.positive_test, self.cmv_test = [], [], []
        self.A = self.B = self.train_auc = self.test_auc = None

    @staticmethod
    def check_p_value(param1, param2, p_value):
        # todo: documentations describe about static function
        first = np.sum((param1 == 0) & (param2[1] == 0))
        second = np.sum((param1 == 0) & (param2[1] == 1))
        third = np.sum((param1 == 1) & (param2[1] == 0))
        fourth = np.sum((param1 == 1) & (param2[1] == 1))
        _, pvalue = stats.fisher_exact([[first, second], [third, fourth]])
        if pvalue < p_value:
            return param2[0], param2[1]

    @staticmethod
    def fishers_exact_test_sort(combinations, cmv, jobs, p_value):
        # todo: documentations describe about function
        print(
            f"\n{Fore.LIGHTBLUE_EX}Filter dictionary by condition of {Fore.LIGHTMAGENTA_EX}p value,{Fore.RED} Fisher's exact test{Fore.RESET}")
        start_time = time.time()
        temp = Parallel(n_jobs=jobs)(
            delayed(EmersonFirstVersion.check_p_value)(cmv, item, p_value) for item in
            tqdm(combinations.items(), desc="Multiprocessing filter dictionary by p value",
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)))
        temp = list(filter(None, temp))

        filtered_p_value_combinations = dict(temp)

        print(
            f"{Fore.LIGHTBLUE_EX}The amount of combination after the filtering operation: {Fore.RED}{len(filtered_p_value_combinations)}{Fore.RESET}")
        print(
            f"{Fore.LIGHTBLUE_EX}Filter dictionary by condition of p value, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")

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

    def data_preparation(self):

        for i, key in enumerate(self.test_patient_info):
            temp = [self.label_combinations[value] for value in self.test_patient_info[key]]
            self.test_patient_pos_neg[key] = np.array(temp)

        for key1 in self.test_patient_pos_neg:
            if key1[1] == 'negative':
                self.cmv_test.append(0)
            else:
                self.cmv_test.append(1)
            self.negative_test.append(sum(self.test_patient_pos_neg[key1] == 0))
            self.negative_test.append(sum(self.test_patient_pos_neg[key1] == 1))

    @staticmethod
    def compute_rho(t, x, y):
        a, b = t
        numerator = a + x
        denominator = a + b + x + y
        rho = numerator / denominator
        return rho

    @staticmethod
    def auc(t, x, y, cmv):
        y_true = np.squeeze(cmv)
        pred = EmersonFirstVersion.compute_rho(t, x, y)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, pred)
        result = metrics.auc(fpr, tpr)
        return -result

    @staticmethod
    def optimization(x, y, cmv, x0):
        try:
            res = minimize(EmersonFirstVersion.auc, x0=x0, method='Nelder-Mead', args=(x, y, cmv),
                           options={'xatol': 1e-8, 'disp': False})
            current_time_auc = abs(res.fun)
            A, B = x0
            return A, B, current_time_auc
        except ValueError:
            A, B = x0
            return A, B, 0

    def learn(self):

        print(
            f"\n{Fore.LIGHTBLUE_EX}Start optimization process of {Fore.LIGHTMAGENTA_EX}parameters A, B - {Fore.RED}Prior{Fore.RESET}")
        start_time = time.time()

        params = np.linspace(0.03, 0.95, num=500)

        temp = Parallel(n_jobs=30)(
            delayed(EmersonFirstVersion.optimization)(self.negative_train, self.positive_train, self.cmv, [a, b]) for
            a, b in
            tqdm(list(product(params, repeat=2)),
                 desc='Optimization Process',
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)))
        a = filter(None, temp)
        a = sorted(a, key=lambda x: x[2], reverse=True)

        self.A, self.B, self.train_auc = a[0]

        print(
            f"{Fore.LIGHTBLUE_EX}Best Parameters: {Fore.RED}A = {self.A}, B = {self.B}{Fore.RESET}")

        print(
            f"{Fore.LIGHTBLUE_EX}End optimization of prior by posterior, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")

    def fit(self, data_path):
        try:
            # TODO: describe functions
            self.activate_mechanism(data_path)
            self.filtered_p_value_combinations = self.fishers_exact_test_sort(self.combinations, self.cmv, self.n_jobs,
                                                                              self.p_value)
            self.determine_label()

            if len(self.label_combinations) == 0:
                raise ValueError(
                    f"No found combinations between patients that meet p value Fisher\'s condition -- {self.label_combinations}")

            self.count_of_pos_neg()
            self.learn()

        except ValueError as e:
            print(f"{Fore.YELLOW}{e}")

    def predict(self, data_path):
        self.prepare_combinations(data_path)
        self.data_preparation()

        _, _, self.test_auc = EmersonFirstVersion.optimization(x=self.negative_test, y=self.positive_test,
                                                               cmv=self.cmv_test,
                                                               x0=[self.A, self.B])
        self.test_auc = abs(self.test_auc)


if __name__ == '__main__':
    try:
        print(Back.LIGHTWHITE_EX, end='')

        # create default paths to pickle's
        ROOT = Path(getcwd()).parent

        DATA = ROOT / 'Data'
        TRAIN = DATA / 'PreprocessedTrain'
        TEST = DATA / 'PreprocessedTest'

        parser = argparse.ArgumentParser(
            description=f"""python3 emerson_version_one.py \
            --train_data_path=/home/vovaps/Projects/EmersonSoftwareFramework/Data/PreprocessedTrain/ \
            --test_data_path=/home/vovaps/Projects/EmersonSoftwareFramework/Data/PreprocessedTest/ \
            --n_jobs=45 \
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

        # new version
        emerson_v1 = EmersonFirstVersion(args.n_jobs, args.p_value)
        emerson_v1.fit(args.train_data_path)
        emerson_v1.predict(args.test_data_path)

        print(f"{Fore.LIGHTBLUE_EX}Train AUC score: {Fore.RED}{emerson_v1.train_auc}{Fore.RESET}")
        print(f"{Fore.LIGHTBLUE_EX}Test AUC score: {Fore.RED}{emerson_v1.test_auc}{Fore.RESET}")

        print(Style.RESET_ALL)

    except Exception as e:
        print(e)
        print(traceback.format_exc())
