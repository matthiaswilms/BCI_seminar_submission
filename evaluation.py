import mne.io
import numpy as np
import scipy.stats as stats
from load_and_clip import load_and_clip
import matplotlib.pyplot as plt


def evaluate_overcorrection(cleaned, original, t_blinks=None, eeg_file=None, et_file=None, plot=False):
    if t_blinks is None:
        _, _, _, t_blinks = load_and_clip(eeg_file, et_file)

    cleaned_data = cleaned.get_data()
    original_data = original.get_data()
    n_samples = min(cleaned_data.shape[1], original_data.shape[1])


    cleaned_data = cleaned_data[:, :n_samples] * 1e6
    original_data = original_data[:, :n_samples] * 1e6

    blink_map = [False] * n_samples
    for (start, end) in t_blinks:
        for i in range(max(start - 40, 0), min(end + 40, n_samples)):
            blink_map[i] = True
    blink_map_inverse = [not x for x in blink_map]
    cleaned_data_art = cleaned_data[:, blink_map]
    cleaned_data_not_art = cleaned_data[:, blink_map_inverse]
    original_data_art = original_data[:, blink_map]
    original_data_not_art = original_data[:, blink_map_inverse]
    average_cleaned_not_art = np.average(cleaned_data_not_art, axis=0)
    average_original_not_art = np.average(original_data_not_art, axis=0)
    difference_not_art = average_original_not_art - average_cleaned_not_art
    average_cleaned = np.average(cleaned_data, axis=0)
    average_original = np.average(original_data, axis=0)
    difference = average_original - average_cleaned
    if plot:
        fig, axs = plt.subplots()
        # fig, axs = plt.subplots(2)
        # axs[0].set_title("Averaged signals in blink-free samples")
        # axs[1].set_title("Averaged signals over the whole recording")
        # axs[0].set_ylabel("uV")
        # axs[1].set_ylabel("uV")
        # axs[0].plot(average_original_not_art - np.average(average_original_not_art), label='avg original')
        # axs[0].plot(average_cleaned_not_art - np.average(average_cleaned_not_art), label='avg cleaned')
        # axs[0].plot(difference_not_art - np.average(difference_not_art), label='difference')
        # axs[0].legend(loc="upper left")
        #
        # axs[1].plot(average_original - np.average(average_original), label='avg original')
        # axs[1].plot(average_cleaned - np.average(average_cleaned), label='avg cleaned')
        # axs[1].plot(difference - np.average(difference), label='difference')
        # axs[1].legend(loc="upper left")

        axs.set_title("Averaged signals over the whole recording")
        axs.set_ylabel("uV")

        axs.plot(average_original - np.average(average_original), label='avg original')
        axs.plot(average_cleaned - np.average(average_cleaned), label='avg cleaned')
        axs.plot(difference - np.average(difference), label='difference')
        axs.legend(loc="upper left")


        # plt.show()

    distortion_variance = np.var(difference_not_art)
    original_var = np.var(average_original)
    cleaned_var = np.var(average_cleaned)
    distortion_std = np.std(difference_not_art)
    original_std = np.std(average_original)
    cleaned_std = np.std(average_cleaned)

    t_blinks = [(start, end) for (start, end) in t_blinks if end <= n_samples]
    missed_blinks = 0
    for (start, end) in t_blinks:
        blink_avg = (start + end) // 2
        correction_value = np.abs(difference[max(blink_avg - 100, 0): min(blink_avg + 100, n_samples)]).max()
        if correction_value < 2 * distortion_std:
            if plot:
                # axs[1].axvline(x=blink_avg, color='red', linestyle='dashed')
                axs.axvline(x=blink_avg, color='red', linestyle='dashed')
            missed_blinks += 1
        elif plot:
            # axs[1].axvline(x=blink_avg, color='grey', linestyle='dashed')
            axs.axvline(x=blink_avg, color='grey', linestyle='dashed')

    if plot:
        fig.savefig('correction_sample.pdf')
    print("Distortion variance:", distortion_variance)
    print("Original signal variance:", original_var)
    print("Cleaned variance:", cleaned_var)
    print("Distortion std:", distortion_std)
    print("Original signal std:", original_std)
    print("Cleaned std:", cleaned_std)

    # missed_blinks = estimate_missed_blinks(average_original, average_cleaned, t_blinks, distortion_std)

    result = {"distortion_variance": distortion_variance, "original_var": original_var, "cleaned_var": cleaned_var,
              "distortion_std": distortion_std, "original_std": original_std, "cleaned_std": cleaned_std,
              "missed_blinks": missed_blinks, 'total_blinks': len(t_blinks)}
    return result


def estimate_missed_blinks(average_original, average_cleaned, t_blinks, distortion_std):
    correction_signal = average_original - average_cleaned
    blink_corrections = []

    for (start, end) in t_blinks:
        s = (start + end) // 2
        blink_corrections.append(correction_signal[s])
    blink_corrections = np.array(blink_corrections)
    std = np.std(blink_corrections)
    avg = np.average(blink_corrections)
    n_missed = 0
    for idx, v in enumerate(blink_corrections):
        if abs(v) < 2 * distortion_std:
            n_missed += 1
            print("Missed blink:", idx)
    print(f"{std=}, {avg=}, {n_missed=}")
    return n_missed
