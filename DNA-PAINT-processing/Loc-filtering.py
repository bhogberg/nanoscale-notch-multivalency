#Funtcion for fitlering localizations based on precision
def filter_prec_loc (locs_file, AuNP_locs_file, filt_param_exp_file):
    # import modules used
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import colors
    import statistics
    import random
    import math

    #Open localization file:
    locs_xy = pd.read_hdf(locs_file, key='locs')

    # Open AuNP localization file:
    if AuNP_locs_file != 'NaN':
        AuNP_locs = pd.read_hdf(AuNP_locs_file, key='locs')
    
    #Determine prec threshold
    #Export prec values
    lpx = locs_xy['lpx'].values.tolist()
    lpy = locs_xy['lpy'].values.tolist()

    # Transfrom prec values from pix to nm
    mag = float(input("Magnification used to collect data: "))
    pix_nm = 13000.0/mag
    lpx_crop = [i * pix_nm for i in lpx if i*pix_nm <=15]
    lpy_crop = [i * pix_nm for i in lpy if i*pix_nm <=15]

    #subsample
    lpx_sub = random.sample(lpx_crop, 200000)
    lpy_sub = random.sample(lpy_crop, 200000)

    #plot
    plt.hist2d(lpx_sub, lpy_sub,
               bins=300,
               norm=colors.LogNorm(),
               cmap="viridis")


    #Plot couple of prec values
    xcoords = [10.0, 5.0, 2.0, 1.0]
    xcoords = [10.0, 5.0, 2.0, 1.0]
    plot_max = xcoords[0]*1.5
    # colors for the lines
    color_l = sns.color_palette("hls", len(xcoords))

    # label
    labels = ['10nm precision', '5nm precision', '2nm precision', '1nm precision']

    for xc, c, l in zip(xcoords, color_l, labels):
        plt.hlines(y=xc,xmin=0,xmax=plot_max, colors=c)
        plt.axvline(x=xc, label=l.format(xc), c=c)
    plt.xlim(0, plot_max)
    plt.ylim(0, plot_max)
    plt.xlabel('X coordinate fit precision [nm]')
    plt.ylabel('Y coordinate fit precision [nm]')
    plt.legend()
    plt.show()

    prec_thr_input = float(input("Precision threshold: [nm] (float)"))
    f = filt_param_exp_file
    f.write("\n")
    f.write("Threshold for localization precision (in nm):" +str(prec_thr_input) +"\n")
    prec_thr_filt = prec_thr_input / pix_nm

    # Plot precisions for determining threshold
    plt.hist2d(lpx_sub, lpy_sub,
               bins=300,
               norm=colors.LogNorm(),
               cmap="viridis")

    # Plot couple of prec values
    xcoords = [prec_thr_input]

    # colors for the lines
    color_l = ['red']

    # label
    labels = ['Precision threshold']

    for xc, c, l in zip(xcoords, color_l, labels):
        plt.hlines(y=xc,xmin=0,xmax=plot_max, colors=c)
        plt.axvline(x=xc, label=l.format(xc), c=c)
    plt.xlim(0, plot_max)
    plt.ylim(0, plot_max)
    plt.xlabel('X coordinate fit precision [nm]')
    plt.ylabel('Y coordinate fit precision [nm]')
    plt.legend()
    plt.show()

    #Determin ellipticity threshold
    # Export prec values
    ell = locs_xy['ellipticity'].values.tolist()
    ell_max_thr = 0.4
    ell = [i for i in ell if i <=ell_max_thr]
    ell_sub = random.sample(ell, 200000)

    # Plot precisions for determining threshold
    y, x, _ = plt.hist(ell_sub, bins=200)


    # Plot couple of prec values
    xcoords = [0.2, 0.1, 0.05]

    # colors for the lines
    colors = sns.color_palette("hls", len(xcoords))

    # label
    labels = ['0.2 ellipticity', '0.1 ellipticity', '0.05 ellipticity']

    for xc, c, l in zip(xcoords, colors, labels):
        plt.axvline(x=xc, label=l.format(xc), c=c)
    plt.xlim(0, ell_max_thr)
    plt.xlabel('Ellipticity of event')
    plt.ylabel('Number of events')
    plt.legend()
    plt.show()

    ell_thr_input = float(input("Ellipticity threshold: (float)"))
    f.write("Threshold for event ellipticity :" + str(ell_thr_input) + "\n")

    # Plot precisions for determining threshold
    y, x, _ = plt.hist(ell_sub, bins=200)

    # Plot couple of prec values
    xcoords = [ell_thr_input]

    # colors for the lines
    colors = ['red']

    # label
    labels = ['Ellipticity threshold']

    for xc, c, l in zip(xcoords, colors, labels):
        plt.axvline(x=xc, label=l.format(xc), c=c)
    plt.xlim(0, ell_max_thr)
    plt.xlabel('Ellipticity of event')
    plt.ylabel('Number of events')
    plt.legend()
    plt.show()
    
    #Filter localizations using the set threshold values
    print('Filtering localizations')
    loc_xy_filt = locs_xy.loc[(prec_thr_filt > locs_xy['lpx']) & (prec_thr_filt > locs_xy['lpy']) & (ell_thr_input > locs_xy['ellipticity'])]
    if AuNP_locs_file != 'NaN':
        AuNP_locs_filt = AuNP_locs.loc[(prec_thr_filt > AuNP_locs['lpx']) & (prec_thr_filt > AuNP_locs['lpy']) & (ell_thr_input > AuNP_locs['ellipticity'])]
    else:
        AuNP_locs_filt = 'NaN'
    return loc_xy_filt, AuNP_locs_filt, prec_thr_filt, ell_thr_input

def AuNP_filt (loc_xy_filt, AuNP_locs_filt):
    print('Removing AuNP localizations')
    # import modules used
    import numpy as np
    import pandas as pd

    # Load AuNP localization coordinates:
    x = AuNP_locs_filt['x'].values.tolist()
    y = AuNP_locs_filt['y'].values.tolist()

    #Removing AuNP localizations from main file
    loc_xy_filt_AuNP = loc_xy_filt.drop(loc_xy_filt.loc[(loc_xy_filt['x'].isin(x)) & (loc_xy_filt['y'].isin(y))].index.tolist())

    return loc_xy_filt_AuNP

# Funtcion for fitlering multi-localizations
def filter_multi_loc(loc_xy_filt, filt_param_exp_file):
    # import modules used
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import statistics
    import matplotlib.pyplot as plt

    # Load photon count data
    pc_list = loc_xy_filt['photons'].values.tolist()

    # Plot distribution for threshold determination
    y, x, _ = plt.hist(pc_list, bins=3000)
    # Calculate mean
    pc_mean = np.mean(pc_list)
    pc_std = np.std(pc_list)

    # Plot couple of std values for theshold determination
    # Plot couple of prec values
    std_l = [1.0, 2.0, 3.0]
    xcoords = []
    for i in range(len(std_l)):
        xcoords.append(pc_mean + (std_l[i] * pc_std))

        # colors for the lines
    colors = sns.color_palette("hls", len(std_l))

    # label
    labels = ['1 STD', '2 STD', '3 STD']

    for xc, c, l in zip(xcoords, colors, labels):
        plt.axvline(x=xc, label=l.format(xc), c=c)
    plt.xlim(0, pc_mean * 3)
    plt.xlabel('Photon count per event')
    plt.ylabel('Number of events')
    plt.legend()
    plt.show()

    pc_thr_input = float(input("Photon count threshold: [STD](float)"))
    f = filt_param_exp_file
    pc_thr_filt = pc_mean + (pc_thr_input*pc_std)
    f.write("Threshold for event photoncount (multiple of STD):" + str(pc_thr_input) + "\n")
    f.write("Threshold for event photoncount (count):" + str(pc_thr_filt) + "\n")

    # Plot distribution for threshold determination
    y, x, _ = plt.hist(pc_list, bins=3000)

    # x coordinates for the lines
    xcoords = [pc_thr_filt]
    # colors for the lines
    colors = ['r']
    # label
    labels = ['Photon count threshold']

    for xc, c, l in zip(xcoords, colors, labels):
        plt.axvline(x=xc, label=l.format(xc), c=c)
    plt.xlim(0, pc_mean * 3)
    plt.xlabel('Photon count per event')
    plt.ylabel('Number of events')
    plt.legend()
    plt.show()

    # Filter localizations using the set threshold values
    loc_xy_multifilt = loc_xy_filt.loc[(pc_thr_filt > loc_xy_filt['photons'])]
    return loc_xy_multifilt, pc_thr_filt

def df_to_nprecarray (loc_xy_multifilt):
    # import modules used
    import numpy as np
    import pandas as pd

    mydata = []
    for row in loc_xy_multifilt.values:
        mydata.append(row)

    mydata = np.array(mydata)

    list(tuple(mydata.transpose()))

    mydtype = np.dtype([
            ("frame", "u4"),
            ("x", "f4"),
            ("y", "f4"),
            ("photons", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
            ("bg", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
            ("ellipticity", "f4"),
            ("net_gradient", "f4"),
        ])

    myRecord = np.core.records.array(list(tuple(mydata.transpose())), dtype=mydtype)

    return myRecord

def export_hdf (file_name, data_rec_array):
    # import modules used
    import h5py

    with h5py.File(str(file_name[:-5]) + '_filter.hdf5',
                   "w") as locs_file:
        locs_file.create_dataset("locs", data=data_rec_array)

def yaml_export (file_name, lp_thr, ellipticity_max, pc_thr):
    import yaml
    dict_file = [{'Generated by': 'Filtering script', 'bg': None, 'photons': [0.0, float(pc_thr)], 'lpx': [0.0, float(lp_thr)], 'lpy': [0.0, float(lp_thr)], 'ellipticity': [0.0, float(ellipticity_max)], 'frame': None, 'net_gradient': None, 'sx': None, 'sy': None, 'x': None, 'y': None}]

    with open(file_name[:-5] + '.yaml', "r") as stream:
        d = list(yaml.safe_load_all(stream))
        d.append(dict_file[0])

    with open(file_name[:-5] + '_filter.yaml', "w") as stream:
        yaml.dump_all(d, stream, default_flow_style=False)

# #Defining path to data folder
import os, glob
folder_path = str(input("Path to folder with RCC undrifted hdf5 file: "))
directory = folder_path
os.chdir(folder_path)

#
AuNP_remove = str(input("AuNP used in data as drift markers?: (y/n)"))

f = open(folder_path + '/Loc-filtering-parameters.txt', "w")

#Open file to be filtered
for locs_file in glob.glob("*RCC_undrift.hdf5"):
    print('Imported localization file : ' + str(locs_file))


    if AuNP_remove == 'y':
        AuNP_file = str(locs_file[:-5]) + '-AuNP.hdf5'
        print('Imported localization file : ' + str(AuNP_file))
    else:
        AuNP_file ='NaN'

#Filter file
    loc_xy_filt, AuNP_locs_filt, loc_prec_thr, ellip_thr = filter_prec_loc(locs_file=locs_file, AuNP_locs_file=AuNP_file, filt_param_exp_file=f)

    if AuNP_remove == 'y':
        loc_xy_filt2 = AuNP_filt(loc_xy_filt=loc_xy_filt, AuNP_locs_filt=AuNP_locs_filt)

    else:
        loc_xy_filt2 = loc_xy_filt

    loc_xy_multifilt, photonc_thr = filter_multi_loc(loc_xy_filt=loc_xy_filt2, filt_param_exp_file=f)
    locs_xy_filt_df_nrec_array = df_to_nprecarray (loc_xy_multifilt=loc_xy_multifilt)
    export_hdf(file_name = locs_file, data_rec_array = locs_xy_filt_df_nrec_array)
    yaml_export(file_name=str(locs_file), lp_thr=loc_prec_thr, ellipticity_max=ellip_thr, pc_thr=photonc_thr)


