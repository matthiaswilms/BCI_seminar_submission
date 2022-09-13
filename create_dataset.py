from load_and_clip import load_single_file, load_and_clip
from mne.preprocessing import ICA
import mne
import numpy as np
from scipy.stats import kurtosis, differential_entropy, skew
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import glob
import warnings
import pathlib


def get_raw_data(data_eeg_raw, scale=True):
    hydro_montage = mne.channels.make_standard_montage('GSN-HydroCel-128')

    eog_channels = [8, 14, 17, 21, 25, 125, 126, 127, 128]  # channels that are eog as specified in the paper
    eog_channels = [i - 1 for i in eog_channels]

    # setting the channel types for mne
    channel_types = ['eeg' if i not in eog_channels else 'eog' for i in range(128)]
    # channel_types = ['eeg' for i in range(128)]  # setting the channel types for mne
    channel_names = ['E' + str(i + 1) for i in range(128)]

    sfreq = 500  # sampling frequency

    # creating the raw object
    info = mne.create_info(channel_names, sfreq, channel_types)
    if scale:
        data_eeg_raw = data_eeg_raw * 1e-6
    raw = mne.io.RawArray(data_eeg_raw, info)
    # raw.set_montage(hydro_montage)
    # [5,6,10,46,64,65,66,67,89,95]  # these are labeled bad channels in the preprocessed file.
    # more bads [1, 32, 56, 63, 68, 73, 81, 88, 94, 99, 107]
    raw.info['bads'] += ['E' + str(i) for i in [5, 6, 10, 46, 64, 65, 66, 67, 89, 95]+[1, 32, 56, 63, 68, 73, 81, 88, 94, 99, 107]]
    raw.info['bads'].append('E48')
    raw.info['bads'].append('E49')
    raw.info['bads'].append('E55')
    raw.info['bads'].append('E76')
    raw.info['bads'].append('E77')
    raw.info['bads'].append('E78')
    raw.info['bads'].append('E113')
    raw.info['bads'].append('E114')
    raw.info['bads'].append('E119')
    return raw


def perform_ICA(raw, n_components=None):
    filt_raw = raw.copy()  # .filter(l_freq=1., h_freq=128)
    ica = ICA(n_components=n_components, max_iter='auto', random_state=97)
    ica.fit(filt_raw)
    srcs = ica.get_sources(filt_raw)
    return srcs


def perform_FastICA(raw, n_components):
    filt_raw = raw.copy()
    ica = FastICA(n_components=n_components, whiten='unit-variance')
    srcs = ica.fit_transform(filt_raw.get_data().T)
    return srcs.T


def compute_variance_ratio(t_blinks, srcs):
    split_indices = list(sum(t_blinks, ()))
    ratios = []
    for IC in srcs:
        split_arr = np.split(IC, split_indices)
        blink_data = np.array([])
        non_blink_data = np.array([])
        for i in range(0, len(split_arr), 2):
            non_blink_data = np.concatenate((non_blink_data, split_arr[i]))
        for i in range(1, len(split_arr), 2):
            blink_data = np.concatenate((blink_data, split_arr[i]))

        var_blink = np.var(blink_data)
        var_non_blink = np.var(non_blink_data)
        var_ratio = var_blink / var_non_blink
        ratios.append(var_ratio)
        # print(var_ratio)
    return ratios


def compute_variance_ratio_part(part, part_index, part_length, blink_map):
    non_blink_data = []
    blink_data = []
    for i, val in enumerate(part):
        if blink_map[part_index * part_length + i]:
            blink_data.append(val)
        else:
            non_blink_data.append(val)
    var_blink = 0
    if len(blink_data) > 0:
        var_blink = np.var(np.array(blink_data))
    if len(non_blink_data) > 0:
        var_non_blink = np.var(np.array(non_blink_data))
    else:
        return math.inf
    var_ratio = var_blink / var_non_blink
    return var_ratio


def extract_features_(data):
    # return [np.var(data), data.max() - data.min(), kurtosis(data), differential_entropy(data, base=2), np.abs(np.median(data)), np.abs(np.average(data))]
    return [np.var(data), data.max() - data.min(), kurtosis(data), differential_entropy(data, base=2), np.abs(np.average(data))]


# variance, kurtosis, Shannon’s entropy and range of amplitude
def extract_features(srcs):
    data = []
    for IC in srcs:
        # row = [np.var(IC, ddof=1), IC.max() - IC.min(), kurtosis(IC), differential_entropy(IC, base=2)]
        row = extract_features_(IC)
        data.append(row)
    data = np.array(data)
    return data


def label_data(data, variance_ratios, threshold=5.0):
    labels = [ratio > threshold for ratio in variance_ratios]
    labels = np.array(labels)
    labeled_data = np.column_stack((data, labels))
    return labeled_data


def segment_data(length, data):
    part_length = length * 500
    num_parts = math.floor(data.shape[1] / part_length)
    res = []
    for row in data:
        for part in np.split(row[:part_length * num_parts], num_parts):
            res.append(part)
    return np.array(res)


def segment_and_score_data(length, srcs, t_blinks):
    part_length = length * 500

    # data = srcs.get_data()
    data = srcs
    sample_length = data.shape[1]

    num_parts = math.floor(sample_length / part_length)
    clipped_length = part_length * num_parts
    # blink_map = [False] * t_blinks[-1][1]
    blink_map = [False] * sample_length
    for (start, end) in t_blinks:
        for i in range(start, end):
            blink_map[i] = True
    res = []
    variance_ratios = []
    for row_idx, row in enumerate(data):
        split = np.split(row[:clipped_length], num_parts)
        for idx, part in enumerate(split):
            res.append(part)
            variance_ratio = compute_variance_ratio_part(part, idx, part_length, blink_map)
            variance_ratios.append(variance_ratio)
            # if variance_ratio > 3:
            #     print("Artifact detected in row", row_idx)
            # fig, ax = plt.subplots()
            # ax.plot(part)
    return np.array(res), np.array(variance_ratios)


def perform_segmented_ICA(raw, t_blinks, window=5, n_components=15):
    sample_rate = 500
    new_len = len(raw) // (window * sample_rate)
    filt_raw = raw.copy().filter(l_freq=1., h_freq=128)
    blink_map = [False] * t_blinks[-1][1]
    for (start, end) in t_blinks:
        for i in range(start, end):
            blink_map[i] = True

    segments = [filt_raw.copy().crop(tmin=i * window, tmax=((i + 1) * window), include_tmax=False) for i in
                range(0, new_len)]
    res = np.array([]).reshape(0, window * sample_rate)
    variance_ratios = []
    for idx, segment in enumerate(segments):
        ica = ICA(n_components=n_components, max_iter='auto', random_state=97)
        ica.fit(segment)
        srcs = ica.get_sources(segment)
        for src in srcs.get_data():
            variance_ratio = compute_variance_ratio_part(src, idx, window * sample_rate, blink_map)
            variance_ratios.append(variance_ratio)
        res = np.concatenate((res, srcs.get_data()))
    return res, np.array(variance_ratios)


def save_dataset(labeled_data, filename):
    # col_labels = ["variance", "range", "kurtosis", "diff. entropy", "artifactual"]
    col_labels = ["variance", "range", "kurtosis", "diff. entropy", "avg", "artifactual"]
    df = pd.DataFrame(labeled_data, columns=col_labels)
    df.to_csv(f'data/{filename}', index=False)


def create_dataset(L_f, H_f, n_components, segment_len, threshold):
    data_eeg_raw, data_et, t, t_blinks = load_single_file()
    raw = get_raw_data(data_eeg_raw)
    raw.notch_filter(freqs=np.arange(50, 201, 50)).filter(l_freq=L_f, h_freq=H_f)

    # srcs = perform_ICA(raw, n_components)
    srcs = perform_FastICA(raw, n_components)
    segmented_data, variance_ratios = segment_and_score_data(segment_len, srcs, t_blinks)
    data = extract_features(segmented_data)
    labeled_data = label_data(data, variance_ratios, threshold=threshold)
    filename = f"dataset{n_components}-{segment_len}-ICA-L{L_f}-H{H_f}-t{threshold}.csv"
    save_dataset(labeled_data, filename)
    return filename


def process_file(eeg_file, et_file, low_freq, high_freq, n_components, segment_len, threshold):
    """
    Create and return a training dataset from the given EEG and ET file.


    :return: the dataset
    """
    data_eeg_raw, data_et, t, t_blinks = load_and_clip(eeg_file, et_file)
    raw = get_raw_data(data_eeg_raw)
    raw.notch_filter(freqs=np.arange(50, 201, 50)).filter(l_freq=low_freq, h_freq=high_freq)

    # srcs = perform_ICA(raw, n_components)
    srcs = perform_FastICA(raw, n_components)
    segmented_data, variance_ratios = segment_and_score_data(segment_len, srcs, t_blinks)
    data = extract_features(segmented_data)
    labeled_data = label_data(data, variance_ratios, threshold=threshold)
    return labeled_data


def process_file_raw(eeg_file, et_file, low_freq, high_freq, n_components, segment_len, threshold):
    """
    Same as above, but the unmixing matrix is learned from a heavily filtered datasets, but then applied to the
    unfiltered version to reconstruct the sources, which are then used to create the training data.
    """
    data_eeg_raw, data_et, t, t_blinks = load_and_clip(eeg_file, et_file)
    raw = get_raw_data(data_eeg_raw)
    # raw.notch_filter(freqs=np.arange(50, 201, 50)).filter(l_freq=0.1, h_freq=128)
    raw.notch_filter(freqs=np.arange(50, 201, 50)).filter(l_freq=0.5, h_freq=64)
    raw.pick_types(eeg=True)  # ignore eog
    filt_raw = raw.copy().filter(l_freq=low_freq, h_freq=high_freq)

    # srcs = perform_FastICA(raw, n_components)
    ica = FastICA(n_components=n_components, whiten='unit-variance', max_iter=500)
    ica.fit(filt_raw.get_data().T)
    srcs = ica.transform(raw.get_data().T).T

    segmented_data, variance_ratios = segment_and_score_data(segment_len, srcs, t_blinks)
    data = extract_features(segmented_data)
    labeled_data = label_data(data, variance_ratios, threshold=threshold)
    return labeled_data


def process_subjects(subjects, low_freq, high_freq, n_components, segment_len, threshold, force_data_gen=False):
    """
    Creates training datasets from the given or all subjects.


    :param subjects: list of given subjects or None for automatically finding subjects
    :return: the directory the datasets have been saved to
    """
    if not subjects:
        subjects_dir = pathlib.Path('data/subjects')
        subjects = [subject.stem for subject in subjects_dir.iterdir()]
    out = ''
    for subject in subjects:
        out = process_subject(subject, low_freq, high_freq, n_components, segment_len, threshold,
                              force_data_gen)

    return out


def process_subject(subject, low_freq, high_freq, n_components, segment_len, threshold, force_data_gen=False):
    """
    Creates and saves a training dataset from each pair of EEG and ET data of the given subject.


    :return: the directory the datasets have been saved to
    """
    subject_path = 'data/subjects/' + subject
    eeg_files = glob.glob(subject_path + f"/{subject}_SR*_EEG.mat")
    et_files = glob.glob(subject_path + f"/{subject}_SR*_ET.mat")
    eeg_files.sort()
    et_files.sort()
    if not len(et_files) == len(eeg_files):
        # warnings.warn(f"Different number of eeg and et files. Could not process {subject}")
        print(f"Different number of eeg and et files for subject {subject}. Clipping")
        if len(eeg_files) > len(et_files):
            eeg_files = eeg_files[:len(et_files)]
        else:
            et_files = et_files[:len(eeg_files)]
        assert len(eeg_files) == len(et_files)
    # return

    output_dir = pathlib.Path(
        f'data/training/C{n_components}-S{segment_len}-L{low_freq}-H{high_freq}-t{threshold}')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if len(glob.glob(f"{str(output_dir)}/{subject}*")) > 0:
        print(f"Data for subject {subject} already exists.")
        if force_data_gen:
            print(f"Overwriting data for subject {subject}.")
        else:
            return str(output_dir)

    for idx, (eeg_file, et_file) in enumerate(zip(eeg_files, et_files)):
        try:
            labeled_data = process_file_raw(eeg_file, et_file, low_freq, high_freq, n_components, segment_len,
                                            threshold)
            # labeled_data = process_file(eeg_file, et_file, low_freq, high_freq, n_components, segment_len,
            #                             threshold)
        except Exception as e:
            print(f"Could not process {eeg_file} and {et_file}:")
            print(e)
            continue
        output_file_name = subject + str(idx + 1) + '.csv'
        # col_labels = ["variance", "range", "kurtosis", "diff. entropy", "artifactual"]
        col_labels = ["variance", "range", "kurtosis", "diff. entropy","avg", "artifactual"]
        df = pd.DataFrame(labeled_data, columns=col_labels)
        df.to_csv(str(output_dir.joinpath(output_file_name)), index=False)
    print("Finished processing subject ", subject)
    return str(output_dir)


def plot_ICA(eeg_file, et_file, low_freq, high_freq, n_components):
    """
    for manual checking
    """
    data_eeg_raw, data_et, t, t_blinks = load_and_clip(eeg_file, et_file)
    raw = get_raw_data(data_eeg_raw)
    raw.notch_filter(freqs=np.arange(50, 201, 50)).filter(l_freq=low_freq, h_freq=high_freq)

    srcs = perform_ICA(raw, n_components)
    srcs.plot()
    return srcs, t_blinks


def plot_blinks(srcs, t_blinks):
    """
    for manual checking
    """
    IC = srcs.get_data()[0]
    BLINK_WIDTH_FRONT = 70
    BLINK_WIDTH_BACK = BLINK_WIDTH_FRONT

    fig, ax = plt.subplots()
    ax.plot(IC)  # plot the first component which happens to include the blinking artifacts
    for (start, end) in t_blinks:
        plt.axvline(x=start - BLINK_WIDTH_FRONT, color='r', linestyle='dashed')
        plt.axvline(x=end + BLINK_WIDTH_BACK, color='y', linestyle='dashed')


def ICA_comparison(raw, n_components):
    filt_raw = raw.copy().filter(l_freq=2, h_freq=32, picks=['eeg', 'eog'])
    filt_raw.pick_types(eeg=True, eog=True)
    srcs = perform_ICA(filt_raw, n_components)
    ica = FastICA(n_components=n_components)
    X = filt_raw.get_data()
    S_ = ica.fit_transform(X.T)
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(srcs.get_data()[0])
    # axs[1].plot(S_.T[0])
    channel_names = ['ICA' + str(i) for i in range(n_components)]
    info = mne.create_info(channel_names, 500)
    srcs2 = mne.io.RawArray(S_.T, info)
    srcs.plot()
    srcs2.plot()
    return srcs, srcs

if __name__ == "__main__":
    # srcs, t_blinks = plot_ICA('data/subjects/ZAB/ZAB_SR1_EEG.mat', 'data/subjects/ZAB/ZAB_SR1_ET.mat', 3, 50, 20)
    # plot_blinks(srcs, t_blinks)
    data_eeg_raw, data_et, t, t_blinks = load_single_file()
    raw = get_raw_data(data_eeg_raw)
    srcs_mne, srcs_sk = ICA_comparison(raw, 50)
    # process_subject('ZAB', 3, 50, 5, 20, 3)
#     L_f = 3
#     H_f = 50
#     n_ICs = 10
#     segment_len = 20
#     threshold = 3
#     data_eeg_raw, data_et, t, t_blinks = load_single_file()
#     raw = get_raw_data(data_eeg_raw)
#     raw.notch_filter(freqs=np.arange(50, 201, 50)).filter(l_freq=L_f, h_freq=H_f)
#
#     srcs = perform_ICA(raw, n_ICs)
#     segmented_data, variance_ratios = segment_and_score_data(segment_len, srcs, t_blinks)
#     data = extract_features(segmented_data)
#     labeled_data = label_data(data, variance_ratios, threshold=threshold)
#     save_dataset(labeled_data, f"dataset{n_ICs}-{segment_len}-ICA-L{L_f}-H{H_f}-t{threshold}.csv")

# segmented_data, variance_ratios = perform_segmented_ICA(raw, t_blinks, segment_len, n_ICs)
# data = extract_features(segmented_data)
# labeled_data = label_data(data, variance_ratios, threshold=threshold)
# save_dataset(labeled_data, f"dataset{n_ICs}-{segment_len}-L{L_f}-H{H_f}-t{threshold}.csv")
