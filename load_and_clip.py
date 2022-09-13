import numpy as np
import scipy.io as sio
import math


def load_single_file():
    return load_and_clip('data/subjects/ZAB/ZAB_SR1_EEG.mat', 'data/subjects/ZAB/ZAB_SR1_ET.mat')

def load_eeg(eeg_file):
    c_eeg = sio.loadmat(eeg_file, squeeze_me=True, struct_as_record=False)
    data_eeg = c_eeg['EEG'].data
    return data_eeg

def load_and_clip(eeg_file, et_file):
    # path_subject = 'data/ZAB'
    # Load content of Matlab files
    # c_eeg = sio.loadmat(path_subject + '/ZAB_SR1_EEG.mat', squeeze_me=True, struct_as_record=False)
    # c_eyetracking = sio.loadmat(path_subject + '/ZAB_SR1_ET.mat', squeeze_me=True, struct_as_record=False)

    c_eeg = sio.loadmat(eeg_file, squeeze_me=True, struct_as_record=False)
    c_eyetracking = sio.loadmat(et_file, squeeze_me=True, struct_as_record=False)

    # EEG data
    # (128 channels x time)
    data_eeg = c_eeg['EEG'].data

    # Eyetracking data
    # (3 channels (x, y and area) x time) and corresponding timestamps
    data_et = c_eyetracking['data'].transpose()
    t, data_et = data_et[0], data_et[1:]

    onsets_eeg = []
    ends_eeg = []
    events_eeg = c_eeg['EEG'].event
    for event in events_eeg:
        if event.type.strip() == '10':
            onsets_eeg.append(event.latency)
        elif event.type.strip() == '11':
            ends_eeg.append(event.latency)

    onsets_et = []
    ends_et = []
    events_et = c_eyetracking['event']
    for event in events_et:
        if event[1] == 10:
            onsets_et.append(np.argmin(np.abs(event[0] - t)))
        elif event[1] == 11:
            ends_et.append(np.argmin(np.abs(event[0] - t)))

    # Throw if sampling rates are not approximately equal
    for onset_eeg, end_eeg, onset_et, end_et in zip(onsets_eeg, ends_eeg, onsets_et, ends_et):
        assert abs((end_eeg - onset_eeg) - (end_et - onset_et)) <= 5

    # Clip start of signals to align them
    n_samples_clip = onsets_eeg[0] - onsets_et[0]
    if n_samples_clip < 0:
        data_et = data_et[:, abs(n_samples_clip):]
        t = t[abs(n_samples_clip):]
    elif n_samples_clip > 0:
        data_eeg = data_eeg[:, abs(n_samples_clip):]

    # Clip end of signals to fix them to equal length
    n_samples = min(data_eeg.shape[-1], data_et.shape[-1])
    data_eeg = data_eeg[:, :n_samples]
    data_et = data_et[:, :n_samples]
    t = t[:n_samples]

    # Sampling rate (we assume 500 Hz and check this here)
    sampling_rate = np.round(np.mean(1 / np.diff(t / 1000))).astype('int')
    assert sampling_rate == 500

    # Get time intervals of eye blinks, fixations and saccades (= eye movement)
    t_blinks = [(start, end) for start, end, _ in c_eyetracking['eyeevent'].blinks.data]
    t_fixations = [(start, end) for start, end, _, _, _, _ in c_eyetracking['eyeevent'].fixations.data]
    t_saccades = [(start, end) for start, end, _, _, _, _, _, _, _ in c_eyetracking['eyeevent'].saccades.data]

    # events_et = c_eyetracking['event']
    # t_blinks = [(math.floor(start / 2), math.floor(end / 2)) for (start, end) in
    #             t_blinks]  # divide by 2 because the eye tracking data is measure by time(ms) but the eeg data by samples, 1 sample = 2ms
    # diff = math.floor(events_et[0][0] / 2) - c_eeg['EEG'].event[0].latency  # line up the eeg with the blinks
    # t_blinks = [(start - diff, end - diff) for (start, end) in t_blinks]

    t_blinks = [(np.argmin(np.abs(start - t)), np.argmin(np.abs(end - t))) for start, end, _ in c_eyetracking['eyeevent'].blinks.data]

    blink_expansion = 40
    t_blinks = [(max(start-blink_expansion, 0), min(end+blink_expansion, n_samples)) for (start, end) in t_blinks]  # TODO: experimental

    return data_eeg, data_et, t, t_blinks
