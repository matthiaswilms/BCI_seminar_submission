import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.preprocessing import Normalizer, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
from train_svm import get_all_data
if __name__ == "__main__":
    scaler = ""

    # comment whichever one you don't want to use
    # ---- if you just want one datafile ----
    # dataset = "data/dataset20-20-ICA-L3-H50-t1.2.csv"
    # plot_name = dataset
    # data = pd.read_csv(dataset)
    # data = data.to_numpy()
    # ----------------------------------------

    # ---- if you want all datafiles from a folder ----
    folder = 'data/training/C64-S5-L1-H32-t5'
    plot_name = folder
    data = get_all_data(folder, num_files=0)
    # -------------------------------------------------


    # X = data[:, :4]
    X = data[:, :-1]
    # X = X/np.linalg.norm(X)
    # X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    scaler = "RobustScaler"
    y = data[:, -1]
    fig, axs = plt.subplots(4, 2)
    fig.suptitle(plot_name + " " + scaler)

    fig.suptitle('Various feature pairs of the training data plotted (zoomed in)')

    axs[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    axs[0, 0].set_title('variance, range')
    axs[0, 1].scatter(X[:, 0], X[:, 2], c=y, cmap=plt.cm.coolwarm)
    axs[0, 1].set_title('variance, kurtosis')
    axs[1, 0].scatter(X[:, 0], X[:, 3], c=y, cmap=plt.cm.coolwarm)
    axs[1, 0].set_title('variance, entropy')
    axs[1, 1].scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm)
    axs[1, 1].set_title('range, kurtosis')
    axs[2, 0].scatter(X[:, 1], X[:, 3], c=y, cmap=plt.cm.coolwarm)
    axs[2, 0].set_title('range, entropy')
    axs[2, 1].scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.coolwarm)
    axs[2, 1].set_title('kurtosis, entropy')

    axs[3, 0].scatter(np.abs(X[:, 4]), X[:, 3], c=y, cmap=plt.cm.coolwarm)
    axs[3, 0].set_title('avg, entropy')

