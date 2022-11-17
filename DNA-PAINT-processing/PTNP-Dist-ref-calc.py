# Import modules used
import os, glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from scipy import optimize
from scipy.signal import find_peaks
import seaborn as sns


# Functions used
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev) ** 2)


def flatten(lis):
    from collections import Iterable

    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


#
def gauss_plot(data, n, x_range):
    from scipy.stats import norm
    import numpy as np

    mean, std = norm.fit(data)
    x = np.linspace(x_range[0], mean * n + 3 * std * n, 100)

    gauss_x = []
    gauss_y = []

    for i in range(n):
        factor = i + 1
        gauss_mean = mean * factor
        gauss_std = std * factor
        gauss_x.append(x)
        gauss_y.append(norm.pdf(x, gauss_mean, gauss_std))

    return gauss_x, gauss_y


# Specify directories
folder_path = str(input("Path to folder with h5 file with exported distance data: "))
directory = folder_path
os.chdir(folder_path)

print('Parameters for localization visualisation:')
mag = float(input("Magnification used to collect data: "))
pix_nm = 13000.0 / mag

# Input information needed for processing
for locs_file in glob.glob("*dist.h5"):
    print(locs_file)
    # Import loc file and transform it into a np array
    locs_df = pd.read_hdf(locs_file, key='str_locs')


    # Plotting site distance distribution
    dist_l = locs_df['dist'].values.tolist()
    mean_data = np.mean(dist_l)
    std_data = np.std(dist_l)
    dist_l_filt = [x for x in dist_l if mean_data - 2 * std_data < x < mean_data + 2 * std_data]


    kde1 = sns.kdeplot(dist_l_filt, color='powderblue', alpha=0.0)
    line1 = kde1.lines[0]
    x1, y1 = line1.get_data()
    peaks1, _ = find_peaks(y1)
    x_c = x1[peaks1[0]]
    y_c = y1[peaks1[0]]
    plt.close()

    plt.hist(dist_l_filt, color='powderblue', bins=40, edgecolor='grey',
             density=True) #50
    # print the answer
    popt, _ = optimize.curve_fit(gaussian, x1, y1, p0=[y_c, np.mean(dist_l_filt), np.std(dist_l_filt)])
    amplitude, d_mean, d_std = popt
    plt.plot(x1, gaussian(x1, *popt), color='r', label='fit')
    plt.xlabel('Neighbouring site distance [nm]')
    plt.ylabel('Normalized frequency')
    plt.legend()
    plt.show()
    plt.close()

    # Plotting site distance distribution
    srt_loc_l = locs_df['str_loc'].values.tolist()
    pos_std_l = [np.mean([np.std([x[0]]),np.std([x[0]])])*pix_nm for x in srt_loc_l]
    mean_data = np.mean(pos_std_l)
    std_data = np.std(pos_std_l)
    pos_std_l_filt = [x for x in pos_std_l if mean_data - 3 * std_data < x < mean_data + 3 * std_data]

    kde1 = sns.kdeplot(pos_std_l_filt, color='powderblue', alpha=0.0)
    line1 = kde1.lines[0]
    x1, y1 = line1.get_data()
    peaks1, _ = find_peaks(y1)
    x_c = x1[peaks1[0]]
    y_c = y1[peaks1[0]]
    plt.close()
    plt.hist(pos_std_l_filt, color='powderblue', bins=25, edgecolor='grey',
             density=True)#35


    # print the answer
    popt, _ = optimize.curve_fit(gaussian, x1, y1, p0=[y_c, np.mean(pos_std_l_filt), np.std(pos_std_l_filt)])
    amplitude, pos_std_mean, pos_std_std = popt
    plt.plot(x1, gaussian(x1, *popt), color='r', label='fit')
    plt.xlabel('Site positional spread [nm]')
    plt.ylabel('Normalized frequency')
    plt.legend()
    plt.show()
    plt.close()

    with open(folder_path + '/Exported_site_reference_values.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['N', len(dist_l)])
        spamwriter.writerow(['Data type', 'Mean', 'STD'])
        spamwriter.writerow(['Site distance', d_mean, d_std])
        spamwriter.writerow(['Site spread', pos_std_mean, pos_std_std])
