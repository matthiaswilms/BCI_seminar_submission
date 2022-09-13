import csv

import mne
import sklearn.svm
from joblib import dump, load
import numpy as np
from create_dataset import get_raw_data, perform_FastICA, extract_features_
from load_and_clip import load_and_clip, load_eeg, load_single_file
import math
from scipy.stats import kurtosis, differential_entropy, skew
from mne.preprocessing import ICA
import pathlib
import glob
from sklearn.decomposition import FastICA
from evaluation import evaluate_overcorrection


def get_data():
    data_eeg_raw, data_et, t, t_blinks = load_and_clip('data/subjects/ZAB/ZAB_SR1_EEG.mat',
                                                       'data/subjects/ZAB/ZAB_SR1_ET.mat')
    raw = get_raw_data(data_eeg_raw)
    raw.notch_filter(freqs=np.arange(50, 201, 50)).filter(l_freq=0.5, h_freq=128)
    return raw


def segment(ica, filt_raw, length, model):
    part_length = length * 500

    data = ica.get_sources(filt_raw).get_data()
    sample_length = data.shape[1]

    num_parts = math.floor(sample_length / part_length)
    clipped_length = part_length * num_parts
    exclude = []
    for row_idx, row in enumerate(data):
        n_excluded = 0
        split = np.split(row[:clipped_length], num_parts)
        for idx, part in enumerate(split):
            svm_input = [np.var(part, ddof=1), part.max() - part.min(), kurtosis(part),
                         differential_entropy(part, base=2)]
            # svm_input = extract_features_(part)
            prediction = model.predict([svm_input])
            if prediction == 1.0:
                n_excluded += 1
        if n_excluded > num_parts // 2:
            exclude.append(row_idx)
    ica.exclude = exclude
    return ica


def clean_srcs(srcs, segment_len, model):
    part_length = segment_len * 500

    sample_length = srcs.shape[1]

    num_parts = math.floor(sample_length / part_length)
    clipped_length = part_length * num_parts
    cleaned_srcs = []
    n_removed = 0
    for row in srcs:
        new_row = np.array([])
        split = np.split(row[:clipped_length], num_parts)
        for part in split:
            svm_input = [np.var(part, ddof=1), part.max() - part.min(), kurtosis(part),
                         differential_entropy(part, base=2)]
            # svm_input = [skew(part), part.max() - part.min(), kurtosis(part),
            #              differential_entropy(part, base=2)]
            svm_input = extract_features_(part)
            prediction = model.predict([svm_input])
            if prediction == 1.0:
                new_row = np.concatenate((new_row, np.zeros(part_length)))
                n_removed += 1
            else:
                new_row = np.concatenate((new_row, part))
        cleaned_srcs.append(new_row)
    print("Removed", n_removed, "component parts")
    return np.array(cleaned_srcs)


def perform_ICA(raw, n_components=None):
    filt_raw = raw.copy().filter(l_freq=3, h_freq=50)
    ica = ICA(n_components=n_components, max_iter='auto', random_state=97)
    ica.fit(filt_raw)
    return ica, filt_raw


def clean():
    pass


def clean_subject_data(subject, model_file, n_components, segment_len, low_freq, high_freq, save_output=False):
    """
    Clean all EEG recordings of a subject.


    :param high_freq:
    :param low_freq:
    :param subject: the subject name
    :param model_file: the model to use for classification
    :param save_output: whether to save the cleaned recordings
    :return: two arrays of the cleaned and the original recordings
    """
    subject_path = 'data/subjects/' + subject
    eeg_files = glob.glob(subject_path + f"/{subject}_SR*_EEG.mat")
    et_files = glob.glob(subject_path + f"/{subject}_SR*_ET.mat")
    eeg_files.sort()
    et_files.sort()
    if not len(et_files) == len(eeg_files):
        # warnings.warn(f"Different number of eeg and et files. Could not process {subject}")
        print(f"CLeaning: Different number of eeg and et files for subject {subject}. Clipping")
        if len(eeg_files) > len(et_files):
            eeg_files = eeg_files[:len(et_files)]
        else:
            et_files = et_files[:len(eeg_files)]
        assert len(eeg_files) == len(et_files)

    cleaned_data = []
    original_data = []
    for eeg_file, et_file in zip(eeg_files, et_files):
        cleaned, original = clean_file(eeg_file, et_file, model_file, n_components, segment_len, low_freq, high_freq,
                                       save_output)
        cleaned_data.append(cleaned)
        original_data.append(original)
    return cleaned_data, original_data


def add_SR1_annotations(raw, srx=1):
    """
    Adds vertical bars for the blink onsets and offsets of SR1_ET
    :param raw:
    :return:
    """
    # _, _, _, t_blinks = load_single_file()
    _, _, _, t_blinks = load_and_clip(f'data/subjects/ZAB/ZAB_SR{srx}_EEG.mat', f'data/subjects/ZAB/ZAB_SR{srx}_ET.mat')
    events = np.array([[end, 0, 1] for (start, end) in t_blinks] + [[start, 0, 0] for (start, end) in t_blinks])
    annotations = mne.annotations_from_events(events, 500)
    raw.set_annotations(annotations)
    return raw


def clean_file(eeg_file, et_file, model_file, n_components, segment_len, low_freq, high_freq, save_output=False):
    """
    Clean the specified EEG recording.


    :param eeg_file: the eeg recording
    :param model_file: the model to use for classification
    :param save_output: whether to save the cleaned recording
    :return: the cleaned and the original recording
    """
    model = load(model_file)
    try:
        data_eeg, data_et, t, t_blinks = load_and_clip(eeg_file, et_file)
    except Exception as e:
        print(f"Could not process {eeg_file} and {et_file}:")
        print(e)
        return (None, None)

    raw = get_raw_data(data_eeg)
    raw.notch_filter(freqs=np.arange(50, 201, 50)).filter(l_freq=0.5, h_freq=64)  # basic filtering
    raw.pick_types(eeg=True)
    original = raw.copy()


    filt_raw = raw.filter(l_freq=low_freq, h_freq=high_freq)  # more filtering to learn the unmixing matrix
    ica = FastICA(n_components=n_components, whiten='unit-variance', max_iter=500)

    filt_raw.pick_types(eeg=True)  # ignore eog
    ica.fit(filt_raw.get_data().T)  # learn unmixing matrix
    to_filter = original.copy().pick_types(eeg=True)  # .filter(l_freq=low_freq, h_freq=high_freq)
    srcs = ica.transform(to_filter.get_data().T)  # apply unmixing matrix to less filtered version


    cleaned_srcs = clean_srcs(srcs.T, segment_len, model)  # clean the sources
    cleaned_signal = ica.inverse_transform(cleaned_srcs.T)  # mix the cleaned sources again
    info = mne.create_info(to_filter.info['ch_names'], 500)
    raw = mne.io.RawArray(cleaned_signal.T, info)

    if save_output:
        model_name = pathlib.Path(model_file).stem
        output_dir = pathlib.Path('data/output/').joinpath(model_name)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        out_file_name = pathlib.Path(eeg_file).stem + '_cleaned.fif'
        raw.save(str(output_dir.joinpath(out_file_name)), overwrite=True)
        cleaning_stats = evaluate_overcorrection(raw, original, t_blinks=t_blinks, plot=False)
        stats_file = output_dir.joinpath('cleaning_stats.csv')

        headers = [k for k in cleaning_stats.keys()]
        if not stats_file.exists():
            with open(stats_file, 'x') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        with open(stats_file, 'a') as f:
            writer = csv.DictWriter(f, headers)
            writer.writerow(cleaning_stats)

    return raw, original


def clean_data(model_file, n_components, segment_len):
    # n_components = 20
    # segment_len = 20
    model_name = model_file
    model = load(model_name)
    raw = get_data()
    original = raw.copy()
    ica, filt_raw = perform_ICA(raw, n_components)
    ica = segment(ica, filt_raw, segment_len, model)
    ica.apply(raw)
    return raw, original


def test():
    pass


if __name__ == "__main__":
    clean, original = clean_data('dataset20-20-ICA-L3-H50-t3.joblib', 20, 20)
    original.plot()
    clean.plot()
