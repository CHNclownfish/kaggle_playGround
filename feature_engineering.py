import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
import seaborn as sns
from cycler import cycler
#from IPython.display import display
import datetime

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
class feature_analysor:
    def __init__(self, train, test):
        self.train = train
        self.test = test


    # feature info
    def show_info(self):
        print(self.train.info())

    # target_portion
    def target_portion(self):
        p = (self.train.target.value_counts() / len(self.train)).round(2)
        print(p)

    # float feature distribution
    def show_float_feature_distribution(self,float_features):


        # Training histograms
        fig, axs = plt.subplots(4, 4, figsize=(16, 16))
        for f, ax in zip(float_features, axs.ravel()):
            ax.hist(self.train[f], density=True, bins=100)
            ax.set_title(f'Train {f}, std={self.train[f].std():.1f}')
        plt.suptitle('Histograms of the float features', y=0.93, fontsize=20)
        plt.show()
        # Test histograms
        # fig, axs = plt.subplots(4, 4, figsize=(16, 16))
        # for f, ax in zip(float_features, axs.ravel()):
        #     ax.hist(test[f], density=True, bins=100)
        #     ax.set_title(f'Test {f}, std={test[f].std():.1f}')
        # plt.show()

    # float feature correlation
    def correlation_of_float_feature(self, float_features):
        plt.figure(figsize=(12, 12))
        sns.heatmap(self.train[float_features + ['target']].corr(), center=0, annot=True, fmt='.2f')
        plt.show()

    # float feature non linear dependence
    def plot_mutual_info_diagram(self, features, ncols=4, by_quantile=True, mutual_info=True,
                                 title='How the target probability depends on single features'):
        def H(p):
            """Entropy of a binary random variable in nat"""
            return -np.log(p) * p - np.log(1-p) * (1-p)

        nrows = (len(features) + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(16, nrows*4), sharey=True)
        for f, ax in zip(features, axs.ravel()):
            temp = pd.DataFrame({f: self.train[f].values,
                                 'state': self.train.target.values})
            temp = temp.sort_values(f)
            temp.reset_index(inplace=True)
            rolling_mean = temp.state.rolling(15000, center=True, min_periods=1).mean()
            if by_quantile:
                ax.scatter(temp.index, rolling_mean, s=2)
            else:
                ax.scatter(temp[f], rolling_mean, s=2)
            if mutual_info and by_quantile:
                ax.set_xlabel(f'{f} mi={H(temp.state.mean()) - H(rolling_mean[~rolling_mean.isna()].values).mean():.5f}')
            else:
                ax.set_xlabel(f'{f}')
        plt.suptitle(title, y=0.90, fontsize=20)
        plt.show()

    def integer_features(self, int_features):
        figure = plt.figure(figsize=(16, 16))
        # for f, ax in zip(int_features, axs.ravel()):
        for i, f in enumerate(int_features):
            plt.subplot(4, 4, i+1)
            ax = plt.gca()
            vc = self.train[f].value_counts()
            ax.bar(vc.index, vc)
            #ax.hist(train[f], density=False, bins=(train[f].max()-train[f].min()+1))
            ax.set_xlabel(f'Train {f}')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # only integer labels
        plt.suptitle('Histograms of the integer features', y=1.0, fontsize=20)
        figure.tight_layout(h_pad=1.0)
        plt.show()

        # Test histograms
        # fig, axs = plt.subplots(4, 4, figsize=(16, 16))
        # for f, ax in zip(int_features, axs.ravel()):
        #     ax.hist(test[f], density=True, bins=100)
        #     ax.set_title(f'Test {f}, std={test[f].std():.1f}')
        # plt.show()
    def string_feature_size(self):
        print(self.train.f_27.str.len().min(), self.train.f_27.str.len().max(),
              self.test.f_27.str.len().min(), self.test.f_27.str.len().max())


    def count_string_feature_values(self):
        print(self.train.f_27.value_counts())

    def count_strin_feature_values_all_dataset(self):
        print(pd.concat([self.train, self.test]).f_27.value_counts())

    def see_str_pos(self):
        for i in range(10):
            print(f'Position {i}')
            tg = self.train.groupby(self.train.f_27.str.get(i))
            temp = pd.DataFrame({'size': tg.size(), 'probability': tg.target.mean().round(2)})
            print(temp)
            print()

    def see_unique_chr(self):
        unique_characters = self.train.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
        tg = self.train.groupby(unique_characters)
        temp = pd.DataFrame({'size': tg.size(), 'probability': tg.target.mean().round(2)})
        print(temp)

    def string_feature_transformer(self):
        for df in [self.train, self.test]:
            for i in range(10):
                df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
            df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))

    def twoD_feature(self):
        plt.rcParams['axes.facecolor'] = 'k'
        plt.figure(figsize=(11, 5))
        cmap = ListedColormap(["#ffd700", "#0057b8"])
        # target == 0 → yellow; target == 1 → blue

        ax = plt.subplot(1, 3, 1)
        ax.scatter(self.train['f_02'], self.train['f_21'], s=1,
                   c=self.train.target, cmap=cmap)
        ax.set_xlabel('f_02')
        ax.set_ylabel('f_21')
        ax.set_aspect('equal')
        ax0 = ax

        ax = plt.subplot(1, 3, 2, sharex=ax0, sharey=ax0)
        ax.scatter(self.train['f_05'], self.train['f_22'], s=1,
                   c=self.train.target, cmap=cmap)
        ax.set_xlabel('f_05')
        ax.set_ylabel('f_22')
        ax.set_aspect('equal')

        ax = plt.subplot(1, 3, 3, sharex=ax0, sharey=ax0)
        ax.scatter(self.train['f_00'] + self.train['f_01'], self.train['f_26'], s=1,
                   c=self.train.target, cmap=cmap)
        ax.set_xlabel('f_00 + f_01')
        ax.set_ylabel('f_26')
        ax.set_aspect('equal')

        plt.tight_layout(w_pad=1.0)
        plt.savefig('three-projections.png')
        plt.show()
        plt.rcParams['axes.facecolor'] = '#0057b8'

    def combine_features(self):
        for df in [self.train, self.test]:
            df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
            df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
            i_00_01_26 = df.f_00 + df.f_01 + df.f_26
            df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
