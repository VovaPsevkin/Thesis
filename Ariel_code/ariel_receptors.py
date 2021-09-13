# from preprocessing import Preprocessing
import os
import sys
import time
from pathlib import Path
import pickle
import json

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import torch
import tensorboard

import numpy as np
from scipy import stats
import pandas as pd
import random

from tqdm import tqdm
from colorama import Fore
import seaborn as sns
import matplotlib.pyplot as plt

from neuralnet import neural_network
import nni


class HistoMaker:

    def __init__(self, name, nump):
        self.name = name
        self.combinations = {}
        self.clustercom = {}
        self.__id2patient = {}
        self.score = {}
        self.outlier = {}
        self.size = nump
        self.cmv = None

    def personal_information(self, directory, mix=False):
        """
            creates the dict the connects people to their index
        """
        directory = Path(directory)
        #self.__size = len(list(directory.glob('*')))
        if self.name == "train":
            samples = random.sample(range(641), self.size)
        else:
            samples = range(120)
        count = 0
        for ind, item in tqdm(enumerate(directory.glob('*')), total=self.size, desc="Maintain patient order",
                              bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            if ind not in samples:
                continue
            # attr hold the patient and the path to him
            attr = item.stem.split('_')
            attr.append(item.as_posix())
            self.__id2patient[tuple(attr)] = count
            count+=1

    def create_cmv_vector(self):
        """
        creates the vector that tells us whether person i is pos/neg
        """
        self.cmv = np.zeros(self.size, dtype=np.int8)
        for item, index in self.__id2patient.items():
            if item[1] == 'positive':
                self.cmv[index] = 1

    def create_combinations(self):
        """
        The dictionary to which data is loaded from all files contains all combinations,
        reads files from another dictionary.
        Initializes each element in the dictionary by key (combination) and the value by  zeros array
        in size of the patient.
        A key word already appears in the dictionary, so one patient's index is gathered.
        """
        print(f"\n{Fore.LIGHTBLUE_EX}Generate a quantity of {Fore.LIGHTMAGENTA_EX}instances combination{Fore.RESET}")
        start_time = time.time()
        for personal_info, ind in tqdm(self.__id2patient.items(), total=self.size, desc='Create Combinations',
                                       bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            # reading each person's file to fill the appropriate place in the dict for that person
            if ind >= self.size:
                break
            _, _, path = personal_info
            df = pd.read_csv(path, usecols=['frequency', 'combined', 'autoencoder_projection'])
            v_frac, v_comb, v_vec = df["frequency"], df["combined"], df["autoencoder_projection"]
            for freq, element, vector in zip(v_frac, v_comb, v_vec):
                # "cleaning" the projection vector
                vector = vector[2:-2]
                if '_' in vector:
                    vector = vector.replace("_", "")
                # creating and adding to the dict
                if (element, vector) not in self.combinations:
                    self.combinations[(element, vector)] = (np.zeros(self.size, dtype=np.int8), None)  # np.zeros(self.size, dtype=np.float32))
                self.combinations[(element, vector)][0][ind] += 1
                # self.combinations[(element, vector)][1][ind] += freq
        print(f"{Fore.LIGHTBLUE_EX}Amount of combinations: {Fore.RED}{len(self.combinations)}{Fore.RESET}")
        print(
            f"{Fore.LIGHTBLUE_EX}Generate a quantity of instances combinations, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")

    def filter_dict(self):
        """Initial filtering of combinations dictionary, filtering is done by the number of
                impressions of a key if it is less than two The combination comes out of a dictionary.
                """
        print(
            f"{Fore.LIGHTBLUE_EX}Filter dictionary by combining multiple instances, {Fore.LIGHTMAGENTA_EX}at least seven{Fore.RESET}")
        start_time = time.time()
        print(f"length of combos{len(self.combinations)}")
        for k in tqdm(list(self.combinations.keys()), total=len(self.combinations), desc='Filter dictionary',
                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            if np.count_nonzero(self.combinations[k][0]) < 7:
                del self.combinations[k]
        print(
            f"{Fore.LIGHTBLUE_EX}The amount of combination after the filtering operation: {Fore.RED}{len(self.combinations)}{Fore.RESET}")

        print(
            f"{Fore.LIGHTBLUE_EX}Filter dictionary by combining multiple instances at least seven, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")

    def add_freq(self):
        for personal_info, ind in tqdm(self.__id2patient.items(), total=self.size, desc='Create Combinations',
                                       bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            # reading each person's file to fill the appropriate place in the dict for that person
            if ind >= self.size:
                break
            _, _, path = personal_info
            df = pd.read_csv(path, usecols=['frequency', 'combined', 'autoencoder_projection'])
            v_frac, v_comb, v_vec = df["frequency"], df["combined"], df["autoencoder_projection"]
            for freq, element, vector in zip(v_frac, v_comb, v_vec):
                # "cleaning" the projection vector
                vector = vector[2:-2]
                if '_' in vector:
                    vector = vector.replace("_", "")
                if (element, vector) in self.combinations:
                    if self.combinations[(element,vector)][1] is None:
                        self.combinations[(element, vector)] = list(self.combinations[(element, vector)])
                        self.combinations[(element, vector)][1] = np.zeros(641, dtype=np.float32)
                    self.combinations[(element, vector)][1][ind] += freq
        with open(self.name + str(self.size) + "freq.p", "wb") as f:
            pickle.dump(self.combinations, f)


    def save_data(self, directory):
        """this includes the whole process of storing the basic data
            """
        self.personal_information(directory)
        self.create_combinations()
        self.filter_dict()
        self.create_cmv_vector()
        with open(f"{self.name}_{self.size}_mix.pkl", "wb") as f:
            pickle.dump(self.combinations, f)


    def load_data(self, directory, mix=True):
        with open(self.name + "presonInfo.pkl", 'rb') as f:
            id2patient = pickle.load(f)
            id2patient = dict(random.sample(list(id2patient.items()), self.size))
        #self.personal_information(directory)
        if mix:
            if self.name == "test":
                name = "test120freq.p"
            else:
                name = "train641freq.p"
        else:
            name = self.name + str(self.size) + "freq.p"
        with open(name, 'rb') as f:
            self.combinations = pickle.load(f)
        if mix:
            combos_to_use = {}
            for combo, val in self.combinations.items():
                newval = [np.zeros(self.size, dtype=np.int16), np.zeros(self.size, dtype=np.float32)]
                for count,  id in enumerate(id2patient.values()):
                    newval[0][count] = val[0][id]
                    newval[1][count] = val[1][id]
                self.combinations[combo] = newval
            for count, key in enumerate(id2patient.keys()):
                self.__id2patient[key] = count
        else:
            for key, val in self.combinations.items():
                val[1] = val[1][:self.size]
        self.create_cmv_vector()


    def hist_ploter_bars(self, data, dataName, type):
        """
        plots the histograms for the V,J genes and the proj (also for actuall amino acids if needed)
        :param data: a tuple of the neg and pos histograms
        :param dataName: train or test
        :param type: regular or cluster
        :return: saves the plot
        """
        x_pos = np.arange(len(data[0]))
        # Build the plot
        fig, ax = plt.subplots()
        wid = 0.35
        ax.bar(x_pos, data[1], width=0.2, align='center', alpha=0.5, color="#073192", label="Positive")  # AED6F1
        ax.bar(x_pos + wid / 2, data[0], width=0.2, align='center', alpha=0.5, color="#19A5EC",
               label="Negative")  # 6495ED
        ax.set_ylabel('P', color='red', size=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(i) for i in range(len(data[0]))])
        # ax.set_title('Distribution of Amino over positives and negatives - ' + dataName)
        ax.yaxis.grid(True)
        ax.legend()
        plt.xticks(fontsize=6)
        # Save the figure and show
        plt.tight_layout()
        plt.savefig(type + self.name + "_hist_" + dataName + '.png')
        plt.show()


    def amino_hist(self, type="Regular"):
        """
        creates histograms for v,j,poj and amino acids and sends them to be plotted
        """
        if "Outlier" in type:
            self.outlier_finder(8.5, 150)
            receptors = self.outlier
        else:
            receptors = self.combinations
        v_mat = np.zeros((32, self.size))
        sum = np.zeros(self.size)
        j_mat = np.zeros((2, self.size))
        amino_letters = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V",
                         "W", "Y"]
        amino_to_numbers = {amino_letters[i]: i for i in range(0, 20)}
        vec_arr = [np.zeros(30, dtype=np.float32), np.zeros(30, dtype=np.float32)]
        amino_arr = (np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.float32))
        negsum = np.count_nonzero(1 - self.cmv)
        possum = np.count_nonzero(self.cmv)
        posrec, negrec = 0, 0
        for element, people in receptors.items():
            v, aminoSeq, j = element[0].split("_")
            vec = np.array(element[1].split(","), dtype=np.float32)
            v = int(v[-2:])
            j = int(j[-2:])
            numpos = np.dot(self.cmv, people[0])
            numneg = np.dot(1 - self.cmv, people[0])
            posrec += numpos
            negrec += numneg
            for letter in aminoSeq:
                amino_arr[0][amino_to_numbers[letter]] += numneg
                amino_arr[1][amino_to_numbers[letter]] += numpos
            v_mat[v - 1] += people[0]
            sum += people[0]
            j_mat[j - 1] += people[0]
            vec_arr[0] += vec * numneg
            vec_arr[1] += vec * numpos
        for colv, colj, num in zip(v_mat.T, j_mat.T, sum):
            if num:
                colv /= num
                colj /= num
        #v_mat = np.where(v_mat != 0, np.log2(v_mat), 0)
        #j_mat = np.where(j_mat != 0, np.log2(j_mat), 0)
        v_arr = (np.zeros(32), np.zeros(32))
        j_arr = (np.zeros(2), np.zeros(2))
        for i in range(len(v_arr[0])):
            v_arr[1][i] = np.dot(self.cmv, v_mat[i]) / possum
            v_arr[0][i] = np.dot(1 - self.cmv, v_mat[i]) / negsum
        for i in range(len(j_arr[0])):
            j_arr[1][i] = np.dot(self.cmv, j_mat[i]) / possum
            j_arr[0][i] = np.dot(1 - self.cmv, j_mat[i]) / negsum
        vec_arr[1] /= posrec
        vec_arr[0] /= negrec
        self.hist_ploter_bars(v_arr, "v_log", type)
        self.hist_ploter_bars(j_arr, "j_log", type)
        self.hist_ploter_bars(vec_arr, "aminoProj", type)
        # self.hist_ploter(amino_arr, "amino", type)

    def scatter_score(self):
        """
        creates a score of how much more prevelant positive people are for a certain recpetor
        """
        self.score = {}
        numn = np.count_nonzero(1 - self.cmv)
        nump = np.count_nonzero(self.cmv)
        pos_precent = nump / (numn + nump)
        for element, val in self.combinations.items():
            sumrec = np.count_nonzero(val[0])
            if sumrec < 2:
                self.score[element] = 0
            else:
                sumPos = np.dot(np.sign(val[0]), self.cmv)
                self.score[element] = abs(sumPos - pos_precent * sumrec) * (sumPos - pos_precent * sumrec) / (pos_precent * sumrec)
            if abs(self.score[element]) > 50:
                del self.score[element]

    def dbscan(self):
        """
        clusters receptors by projection
        :return: a dict that sums the counts of all receptors with the same label
        """
        recProj = []
        combo = []
        for element, people in self.combinations.items():
            recProj.append(np.array(element[1].split(",")))
            combo.append(element)
        recProj = np.array(recProj)
        X = recProj
        X = StandardScaler().fit_transform(X)
        X = normalize(X)
        X = pd.DataFrame(X)
        eps = 0.25
        min_samples = 5
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        print("creating labels..")
        for vec, com, label in zip(recProj, combo, labels):
            if label not in self.clustercom.keys():
                self.clustercom[label] = [np.zeros(self.size, dtype=np.int8), None]
            self.clustercom[label][0] += self.combinations[com][0]
        print("done creating labels")

    def dbscan_scatter_score(self):
        """
         creates a score of how much more prevelant positive people are for a certain cluster
        """
        self.dbscan()
        self.score = {}
        numn = np.count_nonzero(1 - self.cmv)
        nump = np.count_nonzero(self.cmv)
        pos_precent = nump / (numn + nump)
        for element, val in self.clustercom.items():
            sumrec = np.count_nonzero(val[0])
            sumPos = np.dot(np.sign(val[0]), self.cmv)
            self.score[element] = abs(sumPos - pos_precent * sumrec) * (sumPos - pos_precent * sumrec) / (
                        pos_precent * sumrec)
            if abs(self.score[element]) > 20:
                # print(element, self.score[element])
                del self.score[element]

    def scatter_plot(self, other, type="Regular"):
        """
        plots the scatter plots for the receptors and clusters
        :param other: the other data to compare to
        :param type: whether its receptors or clusters of the former
        :return: saves the scatter plot
        """
        # creating the scores for the graph
        if "cluster" in type:
            self.dbscan_scatter_score()
            other.dbscan_scatter_score()
            count_v = self.clustercom
        else:
            self.scatter_score()
            other.scatter_score()
            count_v = self.combinations
        # turns the dicts into lists
        x_s, y_s, count_s = [], [], []
        for rec in self.score.keys():
            if rec in other.score.keys():
                x_s.append(self.score[rec])
                y_s.append(other.score[rec])
                count_s.append(np.mean(count_v[rec][0]))
        # the actual plotting
        try:
            data = pd.DataFrame({'X_s': x_s, 'Y_s': y_s, 'Count_s': count_s})
            fig = plt.figure(figsize=(12, 8))
            # sns.scatterplot(x=X_s, y=Y_s, hue=Nfi_s)
            sns.scatterplot(data=data, x="X_s", y="Y_s", hue='Count_s', palette='coolwarm')
            plt.xlabel(self.name.capitalize() + " set", color='red', size=15)
            plt.ylabel(other.name.capitalize() + " set", color='red', size=15)
            plt.legend([])
            fig.legends = []
            plt.grid()
            plt.savefig(f"Compare {type}")
            # plt.show()

        except Exception:
            print("A --- > X", len(x_s))
            print("B --- > Y", len(y_s))
            print("Seaborn doesn\'t work")

    def tcr_count_make_hist(self, ind, dtype):
        dict_hist = {}
        for rec in self.combinations.values():
            if rec[dtype][ind] in dict_hist.keys() and rec[dtype][ind] != 0:
                dict_hist[rec[dtype][ind]] += 1
            elif rec[dtype][ind] != 0:
                dict_hist[rec[dtype][ind]] = 1
        return dict_hist
        hist = np.zeros(np.max(list(dict_hist.keys())) + 1)
        for count, tcr_n in dict_hist.items():
            hist[count] = tcr_n
        return hist

    def tcr_per_count(self, typ="count"):
        d = {"count": 0, "frequency": 1}
        hists = []
        random.seed()
        for i in range(5):
            ind = random.randint(0, self.size)
            while self.cmv[ind] != 1:
                ind = random.randint(0, self.size)
            hists.append(self.tcr_count_make_hist(ind, d[typ]))
        for i in range(5):
            ind = random.randint(0, self.size-1)
            while self.cmv[ind] != 0:
                ind = random.randint(0, self.size)
            hists.append(self.tcr_count_make_hist(ind, d[typ]))
        size = np.max([len(hist) for hist in hists])
        fig, ax = plt.subplots()
        for id, hist in enumerate(hists):
            if id < 5:
                self.plot_hist_line(ax, hist, typ, 1)
            else:
                self.plot_hist_line(ax, hist, typ, 0)
        #ax.set_title(f"TCRs per {typ} for positives and negitives")
        ax.yaxis.grid(True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(typ.capitalize(), color='red', size=15)
        ax.set_ylabel('Number of TCRs', color='red', size=15)
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        ax.legend()
        plt.xticks(fontsize=6)
        # Save the figure and show
        plt.tight_layout()
        plt.savefig(f"TCR_per_{typ}.png")


    def plot_hist_line(self, ax, data, typ ,isPos):
        if 0 in data:
            del data[0]
        y_pos = []
        x_pos = np.sort(list(data.keys()))
        for freq in data.keys():
            y_pos.append(data[freq])
        # Build the plot
        color = "#073192" if isPos else "#19A5EC"
        if typ == "count":
            ax.plot(x_pos, y_pos, color=color)
        else:
            ax.scatter(x_pos, y_pos, color=color, alpha=0.5, s=3.5)

    def avg_tcr_count(self, typ="count"):
        d = {"count": 0, "frequency": 1}
        dtypelst = [np.int32, np.float32]
        sumArr = np.zeros(self.size, dtype=dtypelst[d[typ]])
        sizeArr = np.zeros(self.size, dtype=dtypelst[d[typ]])
        avgArr = np.zeros(self.size, dtype=np.float32)
        for arr in self.combinations.values():
            sumArr += arr[d[typ]][:self.size]
            sizeArr += np.sign(arr[d[typ]][:self.size])
        np.divide(sumArr, sizeArr, out=avgArr)
        labels = ["CMV+" if cmv == 1 else "CMV-" for cmv in self.cmv]
        df = pd.DataFrame({"Label": labels, "Count_avg": avgArr})
        fig, ax = plt.subplots()
        ax = sns.swarmplot(x="Label", y="Count_avg", hue="Label", data=df, palette=["#073192", "#19A5EC"])
        ax.yaxis.label.set_color('red')
        ax.xaxis.label.set_color('red')
        ax.yaxis.grid(True)
        ax.set_yscale("log")
        #ax.set_title(f"Average {typ} of a patient's tcr divided over positive and negative patients", fontsize=10)
        plt.savefig(f"Avg_TCRs_{typ}.png")

    def tcr_per_threshold(self):
        size = 50
        x = np.linspace(5, 10, size)
        y = np.zeros(size)
        for i in range(size-1, -1, -1):
            y[i] = self.outlier_finder(cutoff=x[i])
        fig, ax = plt.subplots()
        ax.plot(x, y)
        #ax.set_title("TCRs per count for positives and negitives")
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        #ax.set_xscale("log")
        #ax.set_yscale("log")
        ax.set_xlabel("Threshold", color='red', size=15)
        ax.set_ylabel('Number of TCRs', color='red', size=15)
        plt.savefig("TCR_per_threshold.png")

    def tcr_avg_swarm(self, cutoff=7, typ="count"):
        typd = {"count": 0, "frequency": 1, "freq": 1}
        self.outlier_finder(cutoff, 150)
        length = len(self.outlier)
        tcr_lst_control = self.randkeys(length)
        tcr_lst_golden = list(self.outlier.keys())
        labels = (["Control"] * length)
        labels.extend(["Golden"] * length)
        tcr_keys = tcr_lst_control.copy()
        tcr_keys.extend(tcr_lst_golden)
        tcr_values = [np.mean(self.combinations[key][typd[typ]]) for key in tcr_keys]
        df = pd.DataFrame({"Data": tcr_values, "Labels": labels})
        fig, ax = plt.subplots()
        ax = sns.swarmplot(x="Labels", y="Data", hue="Labels", data=df, palette=["#121BDC", "#EBC010"])
        ax.yaxis.grid(True)
        ax.yaxis.label.set_color('red')
        ax.xaxis.label.set_color('red')
        # ax.set_title(f"Average {typ} TCR over golden and control TCRs", fontsize=10)
        plt.savefig(f"Avg_{typ}_for golden_TCRs.png")
        print(stats.ttest_ind(tcr_values[:length], tcr_values[length:], equal_var=False))


    """
     the next five functions check for proximity of golden receptors to each other
    """
    def randkeys(self, num):
        lst = []
        for i in range(num):
            lst.append(random.choice(list(self.combinations.keys())))
        return lst


    def outliercheck(self, type):
        self.outlier_finder(6.5, 150)
        if "control" in type:
            reclst = self.randkeys(len(self.outlier))
        else:
            reclst = list(self.outlier.keys())
        print("len: ", len(self.outlier))
        self.minDist(reclst)
        self.neighscore(reclst)

    def recDist(self, x, y):
        return np.linalg.norm(np.array(x[1].split(","), dtype=float)
                              - np.array(y[1].split(","), dtype=float))

    def minDist(self, reclst):
        mindist = {}
        for recx in reclst:
            mindist[recx] = None
            for recy in reclst:
                dist = self.recDist(recx, recy)
                if not mindist[recx] or 0 < dist < mindist[recx]:
                    mindist[recx] = dist
        print("dist: ", np.mean(list(mindist.values())))

    def neighscore(self, reclst):
        score = {}
        for recx in tqdm(reclst, total=len(self.outlier), desc='Find closest receptor',
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            neigbors = sorted(self.score.items(), key=lambda x: self.recDist(recx, x[0]))[1:6]
            score[recx] = abs(sum([x[1] for x in neigbors])/5)
            print(score[recx])
        print("score: ", np.mean(list(score.values())))

    def clump_TCRs(self):
        self.scatter_score()
        self.outlier_finder(cutoff=5.0)
        group_TCR = {}
        reclst = self.outlier.copy()
        while reclst:
            recx = list(reclst.keys())[0]
            group_TCR[recx] = reclst[recx]
            del reclst[recx]
            for i in range(5):
                if reclst:
                    neighbor = min(reclst.keys(), key=lambda x: self.recDist(recx, x))
                    group_TCR[recx] += reclst[neighbor]
                    del reclst[neighbor]
        self.outlier = group_TCR


    def outlier_finder(self, cutoff=8.0, numrec=None):
        #if len(self.outlier) != 0:
        #    return
        self.scatter_score()
        for element, score in self.score.items():
            if abs(score) > cutoff:
                self.outlier[element] = self.combinations[element]
        if numrec is not None and len(self.outlier) < numrec:
            self.outlier_finder(cutoff=cutoff-0.1, numrec=numrec)
        return len(self.outlier)

    def outlier_finder_score(self, cutoff):
        self.scatter_score()
        for element, score in self.score.items():
            if abs(score) > cutoff:
                self.outlier[element] = self.score[element]
        with open("chi_square.pkl", "wb") as f:
            pickle.dump(self.outlier, f)

    def prepToLearning(self, test, numrec, datatype="default"):
        if "train" != self.name:
            print("can\'t use test data as training data")
            return
        self.outlier_finder(numrec=numrec)
        if datatype == "count":
            trainData = np.transpose(np.asmatrix([row[0] for row in self.outlier.values()]))
            testData = np.transpose(np.asmatrix([test.combinations[key][0]
                                                 if key in test.combinations.keys()
                                                 else np.zeros(test.size)
                                                 for key in self.outlier.keys()]))
        elif datatype == "freq":
            trainData = np.transpose(np.asmatrix([row[1] for row in self.outlier.values()]))
            testData = np.transpose(np.asmatrix([test.combinations[key][1]
                                                 if key in test.combinations.keys()
                                                 else np.zeros(test.size)
                                                 for key in self.outlier.keys()]))
        else:
            trainData = np.transpose(np.asmatrix([np.sign(row[0]) for row in self.outlier.values()]))
            testData = np.transpose(np.asmatrix([np.sign(test.combinations[key][0][:])
                                                 if key in test.combinations.keys()
                                                 else np.zeros(test.size)
                                                for key in self.outlier.keys()]))
        return trainData, testData

    def prepToNn(self, test, numrec, trainPrec, datatype):
        matTraind, matTestd = self.prepToLearning(test, numrec, datatype=datatype)
        traind = torch.from_numpy(matTraind.astype(np.float32))
        testd = torch.from_numpy(matTestd.astype(np.float32))
        realTrainSize = int(trainPrec/100 * len(traind))
        realTraind = traind[0:realTrainSize]
        realTrainl = torch.from_numpy(self.cmv[0:realTrainSize].astype(np.int64))
        vald = traind[realTrainSize:]
        vall = torch.from_numpy(self.cmv[realTrainSize:].astype(np.int64))
        return {"train_data": realTraind, "train_labels":realTrainl, "val_data": vald,
                "val_labels": vall, "test_data": testd, "test_labels": test.cmv.astype(np.int64)}



    def findrfparams(self, test):
        n_estimators = [int(x) for x in np.linspace(100, 1000, 100)]
        max_features = ["auto", "sqrt"]
        max_depth = [int(x) for x in np.linspace(10, 110, 11)]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        with open(f"{self.size}rfparameters.txt", "w") as f:
            rf = RandomForestClassifier()
            for numrec in range(50, 275, 25):
                trainData, testData = self.prepToLearning(test, numrec)
                rf_grid = GridSearchCV(estimator=rf, param_grid=param_grid, scoring="roc_auc",
                                       cv=3, n_jobs=-1)
                rf_grid.fit(trainData, self.cmv)
                dict = rf_grid.best_params_
                dict["numrec"] = numrec
                f.write(json.dumps(dict))
                f.write("/n")
                evaluate(rf_grid.best_estimator_, trainData, self.cmv, testData, test.cmv, numrec, self.size, "rf")

    def findxgbparams(self, test):
        param_grid = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                      "max_depth": [3, 4, 5, 6, 8, 10, 12, 15, 18, 20],
                      "min_child_weight": [1, 3, 5, 7],
                      "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
                      "colsample_bytree": [0.3, 0.4, 0.5, 0.7]}
        with open(f"{self.size}xgbparameters.txt", "w") as f:
            xgbc = xgb.XGBClassifier()
            for numrec in range(50, 275, 25):
                trainData, testData = self.prepToLearning(test, numrec)
                xgb_grid = GridSearchCV(estimator=xgbc, param_grid=param_grid, scoring="roc_auc",
                                       cv=3, n_jobs=-1)
                xgb_grid.fit(trainData, self.cmv)
                dict = xgb_grid.best_params_
                dict["numrec"] = numrec
                f.write(json.dumps(dict))
                evaluate(xgb_grid.best_estimator_, trainData, self.cmv, testData, test.cmv, numrec, self.size, "xgb")

    def randomForest(self, test, params, dtype="default"):
        trainParam, testParam = self.prepToLearning(test, params["numrec"], dtype)
        paramscopy = params.copy()
        del paramscopy["numrec"]
        rfc = RandomForestClassifier(**paramscopy)
        rfc.fit(trainParam, self.cmv)
        pred = rfc.predict(testParam)
        # plot_roc_curve(rfc, testParam, test.cmv)
        # plt.savefig(f"roc_test_numpeople{self.size}_{dtype}.png")
        # plot_roc_curve(rfc, trainParam, self.cmv)
        # plt.savefig(f"roc_train_numpeople{self.size}_{dtype}.png")
        return roc_auc_score(test.cmv, pred)

    def xgboost(self, test, params):
        trainData, testData = self.prepToLearning(test, params["numrec"])
        paramscopy = params.copy()
        del paramscopy["numrec"]
        clf = xgb.XGBClassifier(**paramscopy)
        #learn = CalibratedClassifierCV(clf)
        clf.fit(trainData, self.cmv)
        pred = clf.predict(testData)
        # plot_roc_curve(clf, testData, test.cmv)
        # plt.savefig(f"roc_test_nest{nest}_gamma{gamma}_lamda{lamda}_sample{sample}.png")
        # plot_roc_curve(clf, trainData, self.cmv)
        # plt.savefig(f"roc_train_nest{nest}_gamma{gamma}_lamda{lamda}_sample{sample}.png")
        return roc_auc_score(test.cmv, pred)



def evaluate(model, train_features, train_labels, test_features, test_labels, numrec, size, name):
    plot_roc_curve(model, test_features, test_labels)
    plt.savefig(f"roc{name}_test_numrec{numrec}numpeople{size}.png")
    plot_roc_curve(model, train_features, train_labels)
    plt.savefig(f"roc{name}_train_numrec{numrec}numpeople{size}.png")



if __name__ == "__main__":
    test = HistoMaker("test", 120)
    test.load_data("/home/psevkiv/vova/Project/Data/test_data")
    # params = {
    # "batch_size": 32,
    # "hidden_size_01": 64,
    # "hidden_size_02": 16,
    # "lr": 0.0001,
    # "n_epoch": 11,
    # "numrec": 50,
    # "trainprec": 90
    # }
    #params = {"bootstrap": True, "max_depth": 10, "max_features": "auto", "min_samples_leaf": 1,"min_samples_split": 2, "n_estimators": 100}
    params = nni.get_next_parameter()
    model = sys.argv[1]
    for i in range(len(sys.argv) - 2):
        auc = 0
        for j in range(3):
            train = HistoMaker("train",  int(sys.argv[i+2]))
            train.load_data("/home/psevkiv/vova/Project/Data/train_data", mix=True)
            if "test" in model:
                print(train.cmv)

            if "group" in model:
                train.clump_TCRs()

            if "rf" in model:
                auc += train.randomForest(test, params)/3

            if "xgb" in model:
                auc += train.xgboost(test, params)/3

            if "nn" in model:
                data = train.prepToNn(test, params["numrec"], params["trainprec"], "freq")
                auc += neural_network(**data, params=params)/3
        if sys.argv[i+2] != 500:
            nni.report_intermediate_result(auc)
        else:
            nni.report_final_result(auc)


    #
    # def xgboost(self, test, params):
    #     trainData, testData = self.prepToLearning(test, 100)
    #     #dtrain = xgb.DMatrix(trainData, label= self.cmv)
    #     #dtest = xgb.DMatrix(testData)
    #     param = {'max_depth': 8, 'eta': 0.4, "min_child_weight": 3, "n_estimators":nest,
    #              "reg_lambda": lamda, "gamma":gamma, "objective":'binary:logistic', "colsample_bytree": sample}
    #     clf = xgb.XGBClassifier(**params)
    #     #learn = CalibratedClassifierCV(clf)
    #     clf.fit(trainData, self.cmv)
    #     pred = clf.predict(testData)
    #     plot_roc_curve(clf, testData, test.cmv)
    #     plt.savefig(f"roc_test_nest{nest}_gamma{gamma}_lamda{lamda}_sample{sample}.png")
    #     plot_roc_curve(clf, trainData, self.cmv)
    #     plt.savefig(f"roc_train_nest{nest}_gamma{gamma}_lamda{lamda}_sample{sample}.png")
    #     print(nest, gamma, lamda, sample)
    #     #print("area under roc: ", roc_auc_score(test.cmv, pred))
    #     params = [{"bootstrap": True, "max_depth": 10, "max_features": "auto", "min_samples_leaf": 1,
    #                "min_samples_split": 2, "n_estimators": 100},
    #               {"bootstrap": True, "max_depth": 20, "max_features": "sqrt", "min_samples_leaf": 1,
    #                "min_samples_split": 5, "n_estimators": 100},
    #               {"bootstrap": True, "max_depth": 40, "max_features": "auto", "min_samples_leaf": 1,
    #                "min_samples_split": 10, "n_estimators": 100},
    #               {"bootstrap": True, "max_depth": 30, "max_features": "auto", "min_samples_leaf": 1,
    #                "min_samples_split": 10, "n_estimators": 100},
    #               {"bootstrap": True, "max_depth": 50, "max_features": "sqrt", "min_samples_leaf": 1,
    #                "min_samples_split": 10, "n_estimators": 100},
    #               {"bootstrap": True, "max_depth": 80, "max_features": "auto", "min_samples_leaf": 1,
    #                "min_samples_split": 5, "n_estimators": 200}]
    #     params = [{"numrec": 250, "trainprec": 60, "batch_size": 128, "hidden_size_01": 32, "hidden_size_02": 32,
    #                "lr": 0.001, "n_epoch": 11},
    #               {"numrec": 125, "trainprec": 80, "batch_size": 64, "hidden_size_01": 128, "hidden_size_02": 16,
    #                "lr": 0.001, "n_epoch": 15},
    #               {"numrec": 50, "trainprec": 90, "batch_size": 32, "hidden_size_01": 64, "hidden_size_02": 16,
    #                "lr": 0.001, "n_epoch": 15},
    #               {"numrec": 50, "trainprec": 70, "batch_size": 16, "hidden_size_01": 256, "hidden_size_02": 32,
    #                "lr": 0.0001, "n_epoch": 12},
    #               {"numrec": 50, "trainprec": 80, "batch_size": 16, "hidden_size_01": 128, "hidden_size_02": 64,
    #                "lr": 0.0001, "n_epoch": 11},
    #               {"numrec": 150, "trainprec": 70, "batch_size": 16, "hidden_size_01": 128, "hidden_size_02": 32,
    #                "lr": 0.0001, "n_epoch": 7}]
