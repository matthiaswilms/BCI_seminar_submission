import warnings

import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import svm
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from joblib import dump, load
import csv
import pathlib
from datetime import datetime


def kernels_plot(clf, X, y, X_test=None, y_test=None):
    # You can visualize the SVM wit this, but only with two features
    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of linear SVC ')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    if X_test is not None and y_test is not None:
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=ListedColormap(["black", "orange"]), s=20, edgecolors='k')
    ax.set_ylabel('y label here')
    ax.set_xlabel('x label here')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()


def train_svm(data, model_name, kernel='linear', class_weight=None, log=True):
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    C = 30
    model = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel, class_weight=class_weight, C=C))
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    target_names = ["non-artifactual", "artifactual"]
    if log:
        print("Test data stats:")
        print(classification_report(y_test, y_pred))
        print("Training data stats:")
        print(classification_report(y_train, clf.predict(X_train)))

    model = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel, class_weight=class_weight, C=C))
    clf = model.fit(X, y)
    dump(clf, model_name)

    res = classification_report(y_test, y_pred, target_names=target_names, output_dict=True), \
          classification_report(y_train, clf.predict(X_train), target_names=target_names, output_dict=True)
    return res

def report(data_dir, model_file):
    data = get_all_data(data_dir)
    model = load(model_file)
    X = data[:, :5]
    y = data[:, -1]
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))


def get_all_data(dir, num_files=0):
    """
    Loads and combines all csv files from the given directory.


    :param dir: the directory to load from
    :param num_files: use this to limit the number of files
    :return: the combined data
    """
    data_dir = pathlib.Path(dir)
    if not data_dir.exists():
        warnings.warn(f"Data directory {dir} no found.")
        return
    combined_data = None
    files = list(data_dir.iterdir())
    files.sort()
    if num_files:
        files = files[:num_files]
    for file in files:
        d = pd.read_csv(str(file)).to_numpy()
        if combined_data is None:
            combined_data = d
        else:
            combined_data = np.concatenate((combined_data, d))
    return combined_data


def write_results(result_file, test_stats, train_stats, model_name):
    headers = ["model", "date", "type", "class", "precision", "recall", "f1-score", "support"]
    if not result_file.exists():
        with open(result_file, 'x') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    rows = []
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    row = {'model': model_name, 'type': 'train', 'date': date_str, 'class': 'non-artifactual'}
    row.update(train_stats['non-artifactual'])
    rows.append(row)
    row = {'model': '', 'type': '', 'date': '', 'class': 'artifactual'}
    row.update(train_stats['artifactual'])
    rows.append(row)
    row = {'model': '', 'type': 'test', 'date': '', 'class': 'non-artifactual'}
    row.update(test_stats['non-artifactual'])
    rows.append(row)
    row = {'model': '', 'type': '', 'date': '', 'class': 'artifactual'}
    row.update(test_stats['artifactual'])
    rows.append(row)
    with open(result_file, 'a') as f:
        writer = csv.DictWriter(f, headers)
        writer.writerows(rows)


def train(data_dir, num_files=0, kernel='linear', class_weight=None, log=True, overwrite=False):
    """
    Train a SVM using the data in the given directory and save the stats to a csv file.


    :param data_dir: directory with training data files
    :param num_files: maximum number of training files used
    :param kernel: kernel type, either 'linear', 'rbf' or 'poly'
    :param class_weight: None or 'balanced'
    :param log: print training stats
    :param overwrite: overwrite existing model trained on that data directory
    :return: the name of the model file
    """
    data_path = pathlib.Path(data_dir)
    data = get_all_data(data_dir, num_files)
    if log:
        print("Training data shape: ", data.shape)
    if class_weight:
        model_name = f"{data_path.name}-{kernel}-{class_weight}.joblib"
    else:
        model_name = f"{data_path.name}-{kernel}.joblib"
    model_path = pathlib.Path(model_name)
    if not model_path.exists() or overwrite:
        test_stats, train_stats = train_svm(data, model_name, kernel, class_weight, log)

        # write model stats to file
        output_dir = pathlib.Path('data/output')
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        result_file = output_dir.joinpath('training_stats.csv')
        write_results(result_file, test_stats, train_stats, model_path.stem)

    return model_name


if __name__ == "__main__":
    filename = 'dataset20-20-ICA-L3-H50-t3.csv'
    data = pd.read_csv(f'data/{filename}')
    data = data.to_numpy()
    samples = 5000
    n_points = 400
    X = data[:samples, [0, 1, 2, 3]]  # select at most two if you want to plot
    y = data[:samples, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = make_pipeline(RobustScaler(), svm.SVC(kernel='linear', class_weight=None))
    clf = model.fit(X_train, y_train)
    # kernels_plot(clf, X_train[:n_points], y_train[:n_points], X_test[:n_points], y_test[:n_points])
    # kernels_plot(clf, X_train[:n_points], y_train[:n_points])
    y_pred = clf.predict(X_test)
    print("Test data stats:")
    print(classification_report(y_test, y_pred))
    print("Training data stats:")
    print(classification_report(y_train, clf.predict(X_train)))

    dump(clf, f'{filename.replace(".csv", "")}.joblib')

    # X_train_2 = RobustScaler().fit_transform(X_train)
    # X_test_2 = RobustScaler().fit_transform(X_test)
    # model = svm.SVC(kernel='linear', class_weight='balanced')
    # clf = model.fit(X_train_2, y_train)
    # # kernels_plot(clf, X_train_2[:n_points], y_train[:n_points], X)
    # y_pred = clf.predict(X_test_2)
    # print(classification_report(y_test, y_pred))

    # data = pd.read_csv('data/dataset1.csv')
    # data = data.to_numpy()
    # X = data[:, :4]
    # y = data[:, -1]
    # clf = svm.SVC()
    # clf.fit(X, y)
    # plt.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.coolwarm)
    # ax = plt.gca()
    # plt.show()
