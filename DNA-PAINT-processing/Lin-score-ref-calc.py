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
def gauss(x,mu,sigma,A):
    import numpy as np
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)




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

def infl_p(data):
    from scipy import signal
    smooth = signal.savgol_filter(data[1], 55, 3)#55
    dxdy = signal.savgol_filter(np.gradient(smooth), 55, 3)
    dx2dy2 = signal.savgol_filter(np.gradient(dxdy), 55, 3)
    infl_p_ind = []
    for ind_l in np.where(dx2dy2==dx2dy2.min()):
        ind = ind_l[0]
        if ind != 0:
            infl_p_ind.append(ind)

    infl_p = [[data[0][x], data[1][x]] for x in infl_p_ind]

    return infl_p

# Specify directories
folder_path = str(input("Path to folder with h5 file with exported distance data: "))
directory = folder_path
os.chdir(folder_path)

print('Parameters for localization visualisation:')

# Input information needed for processing
for locs_file in glob.glob("*lin-data.h5"):
    print(locs_file)
    # Import loc file and transform it into a np array
    locs_df = pd.read_hdf(locs_file, key='str_locs')

    # Plotting site distance distribution
    df_filt = locs_df.loc[(4 == locs_df['Str_spot_n'])]
    ROI_lin_score = df_filt['Str_lin_score'].values.tolist()
    # sort the data:
    data = ROI_lin_score
    spot_prot_ratio_mean = np.mean(data)
    spot_prot_ratio_std = np.std(data)
    data_sorted = np.sort(data)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    input_data = [data_sorted, p]
    #
    infl_point_l = infl_p(data=input_data)

    # plot the sorted data:
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    for infl_point in infl_point_l:
        ax1.axvspan(0, infl_point[0], facecolor='r', alpha=0.5)
    ax1.hist(ROI_lin_score, bins=28, color='powderblue', edgecolor='grey')
    ax1.set_xlabel('Linearity score')
    ax1.set_ylabel('Frequency')

    ax2 = fig.add_subplot(122)
    for infl_point in infl_point_l:
        ax2.axvspan(0, infl_point[0], facecolor='r', alpha=0.5)
    ax2.plot(data_sorted, p, color='cadetblue', linewidth=3)
    for infl_point in infl_point_l:
        ax2.plot(infl_point[0], infl_point[1], 'ro')
    ax2.set_xlabel('Linearity score')
    ax2.set_ylabel('$p$')

    plt.show()
    plt.close()

    with open(folder_path + '/Exported_linearity_reference_values.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['N', len(ROI_lin_score)])
        spamwriter.writerow(['Data type', 'Thr'])
        spamwriter.writerow(['Linearity score', infl_point[0]])
