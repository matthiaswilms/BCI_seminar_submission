from clean_data import clean_subject_data
from train_svm import train
from create_dataset import process_subjects


def run_pipeline(low_freq, high_freq, n_components, segment_len, threshold, kernel='linear', force_data_gen=False,
                 overwrite_model=False,
                 num_files=0, save_output=False):
    """
    Runs the whole pipeline: training data creation, SVM training, data cleaning.
    The created training data and model is saved for later use.



    :param low_freq:  lower frequency cutoff for filtering before ICA
    :param high_freq: higher frequency cutoff for filtering before ICA
    :param n_components:  number of independent components returned by ICA
    :param segment_len:  length in seconds of segments
    :param threshold:  variance ratio threshold for labeling of artifactual components
    :param kernel: kernel type, either 'linear', 'rbf' or 'poly'
    :param force_data_gen: force generation of datasets even if they already exist
    :param overwrite_model: overwrite existing model trained on that data directory
    :param num_files: number of files in 'training' folder to user for model training
    :param save_output: whether to save the cleaned recording(s)
    :return:
    """

    subjects = ['T']
    print("Processing subjects..")
    data_dir = process_subjects(subjects, low_freq, high_freq, n_components, segment_len, threshold,
                                force_data_gen)
    print("Training model..")
    model_name = train(data_dir, num_files, kernel=kernel, overwrite=overwrite_model)
    print("Cleaning data..")
    # eeg_file = 'data/subjects/ZJM/ZJM_SR2_EEG.mat'
    # et_file = 'data/subjects/ZJM/ZJM_SR2_ET.mat'
    # cleaned, original = clean_file(eeg_file, et_file, model_name, n_components, segment_len,
    #                                low_freq, high_freq, save_output)
    # evaluate_overcorrection(cleaned, original, eeg_file=eeg_file, et_file=et_file, plot=True)
    cleaned, original = clean_subject_data('T_eval', model_name, n_components, segment_len,
                                           low_freq, high_freq, save_output=True)
    return cleaned, original





if __name__ == "__main__":
    low_freq = 1
    high_freq = 32
    n_components = 32
    segment_len = 5
    threshold = 5

    cleaned, original = run_pipeline(low_freq, high_freq, n_components, segment_len, threshold,
                                     force_data_gen=False, num_files=0, save_output=False,
                                     overwrite_model=True, kernel='rbf')

    print("Finished running pipeline")

