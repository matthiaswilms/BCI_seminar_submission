import matplotlib.pyplot as plt


def big_plot():
    fig, axs = plt.subplots(4, figsize=(5, 12))
    frac_col = 'red'
    missed_col = 'blue'
    for i in range(4):
        axs[i].set_ylabel("missed corrections", color=missed_col)
        axs[i].tick_params(axis='y', labelcolor=missed_col)

    # axs[0].set_title("$f_l$")
    axs[0].set_xlabel("$f_l$")
    l_freqs = [0.5, 1, 2, 3, 5]
    n_missed = [13, 13, 17, 19, 37]
    std_ratios = get_std_ratios('C64-S5-L*-H32-t5-rbf')
    a = axs[0].twinx()
    a.set_ylabel('$frac{\sigma_o}{\sigma_d}$', color=frac_col)
    a.tick_params(axis='y', labelcolor=frac_col)
    a.plot(l_freqs, std_ratios, marker='v', color=frac_col, linestyle='--')
    axs[0].plot(l_freqs, n_missed, marker='o', linestyle='--', color=missed_col)

    axs[1].set_xlabel("number of components")
    n_components = [16, 32, 64, 89]
    n_missed = [9, 7, 12, 16]

    distortion = get_std_ratios_list(['C16-S5-L1-H32-t5-rbf', 'C32-S5-L1-H32-t5-rbf','C64-S5-L1-H32-t5-rbf', 'C94-S5-L1-H32-t5-rbf'])
    a = axs[1].twinx()
    a.set_ylabel('$frac{\sigma_o}{\sigma_d}$', color=frac_col)
    a.tick_params(axis='y', labelcolor=frac_col)
    a.plot(n_components, distortion, marker='v', color=frac_col, linestyle='--')
    axs[1].plot(n_components, n_missed, marker='o', linestyle='--', color=missed_col)

    axs[2].set_xlabel("variance ratio thresholds")
    thresholds = [1, 1.1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30]
    n_missed = [11, 11, 12, 12, 12, 12, 14, 21, 33, 42, 51]
    std_ratios = get_std_ratios()

    a = axs[2].twinx()
    a.set_ylabel('$frac{\sigma_o}{\sigma_d}$', color=frac_col)
    a.tick_params(axis='y', labelcolor=frac_col)
    a.plot(thresholds, std_ratios, marker='v', color=frac_col, linestyle='--')
    axs[2].plot(thresholds, n_missed, marker='o', linestyle='--', color=missed_col)


    axs[3].set_xlabel("segment length")
    segment_len = [2, 3, 4, 5, 7, 10, 20]
    n_missed = [22, 20, 16, 12, 12, 12, 2]
    std_ratios = get_std_ratios('C64-S*-L1-H32-t5-rbf')

    # distortion = [0.2, 0.3, 0.1, 0.3, 0.1]
    a = axs[3].twinx()
    a.set_ylabel('$frac{\sigma_o}{\sigma_d}$', color=frac_col)
    a.tick_params(axis='y', labelcolor=frac_col)
    a.plot(segment_len, std_ratios, marker='v', color=frac_col, linestyle='--')
    axs[3].plot(segment_len, n_missed, marker='o', linestyle='--', color=missed_col)

    fig.savefig('parameters1.pdf')
    plt.show()

def get_std_ratios_list(dirs: list):
    import csv
    files = []
    for dir in dirs:
        directory_pattern = f'data/output/{dir}/cleaning_stats.csv'
        files.append(directory_pattern)
    files.sort()
    ratios = []
    for file in files:
        ratio = 0
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ratio += float(row['distortion_std']) / float(row['original_std'])
        ratio = ratio / 3
        ratios.append(ratio)
    return ratios


def get_std_ratios(directory_pattern=None):
    import glob
    import pathlib
    import csv
    if not directory_pattern:
        directory_pattern = 'data/output/C64-S5-L1-H32-t*-rbf/*.csv'
    else:
        directory_pattern = f'data/output/{directory_pattern}/*.csv'
    files = glob.glob(directory_pattern)
    files.sort()
    ratios = []
    for file in files:
        ratio = 0
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ratio += float(row['distortion_std']) / float(row['original_std'])
        ratio = ratio / 3
        ratios.append(ratio)
    return ratios


if __name__ == "__main__":
    big_plot()
