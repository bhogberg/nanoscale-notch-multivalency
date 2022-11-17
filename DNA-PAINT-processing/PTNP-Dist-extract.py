def loc_to_img_crop(file_name, loc_array, window_size, SPR, norm_max, int_scale, gauss_size, index):
    #import packages
    import numpy as np

    #Make empty image
    img = np.zeros((window_size, window_size), dtype='uint8')

    #Modify coordinates for centering
    x_coord_l = [item[0] for item in loc_array]
    y_coord_l = [item[1] for item in loc_array]
    x_cent = np.mean(x_coord_l)
    y_cent = np.mean(y_coord_l)
    orig_x = x_cent - (window_size/(2*SPR))
    orig_y = y_cent - (window_size/(2*SPR))
    loc_array_n = [[x[0] - orig_x, x[1] - orig_y] for x in loc_array]

    # Fill up image with localizations
    for loc in loc_array_n:
        # Make empty image for detection
        x = int((loc[0])*SPR)
        y = int((loc[1])*SPR)
        if 0 < x < window_size and 0 < y < window_size:
            img[x][y] = img[x][y] + 1.0

    # Normalize intensity
    img_max = np.amax(img)
    img_min = np.amin(np.where(img > 0, img, np.inf).min(axis=1))
    img = norm_max * ((img - img_min) / img_max)
    img[img < 0.0] = 0.0

    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img, (int(gauss_size), int(gauss_size)), 0)
    scale_factor = int_scale / np.amax(img_blur)
    img_blur = img_blur * scale_factor

    cv2.imwrite(file_name[:-3] + '-Str-' + str(index) + '-render-GS-.jpg', img_blur)
    cv2.destroyAllWindows()

    image = cv2.imread(file_name[:-3] + '-Str-' + str(index) + '-render-GS-.jpg', 0)
    im_color = cv2.applyColorMap(image, cv2.COLORMAP_HOT)

    cv2.imwrite(file_name[:-3] + '-Str-' + str(index) + '-render-CLR-.jpg',im_color)
    cv2.destroyAllWindows()

    im_rgb = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)

    return im_rgb, loc_array_n

def Dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def Dist(p1, p2):
    import numpy as np
    return np.sqrt(Dist2(p1, p2))

# Clustering structure localizations into two spots using DBSCAN algorithm
def loc_dbscan_clust(str_coor_a, SPR_det, clust_n_thr, eps, min_samples,
                     print_plots, pix_size_nm, window_size, render_param, str_cnt, file_name):
    #import used packages
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    import heapq
    import math

    # Create lists to export values (localization coordinates, times, photon count, localization center for detected spots)
    spot_a_locs = []
    spot_a_center = [0, 0]
    spot_b_locs = []
    spot_b_center = [0, 0]
    noise_locs = []

    # Modify coordinates for centering
    x_coord_l = [item[0] for item in str_coor_a]
    y_coord_l = [item[1] for item in str_coor_a]
    x_cent = np.mean(x_coord_l)
    y_cent = np.mean(y_coord_l)
    orig_x = x_cent - (window_size / (2 * SPR_det))
    orig_y = y_cent - (window_size / (2 * SPR_det))
    centered_str_coor_a = [[x[0] - orig_x, x[1] - orig_y] for x in str_coor_a]

    # Compute DBSCAN
    spots_a_n = StandardScaler().fit_transform(centered_str_coor_a)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(spots_a_n)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Finding cluster for spots by excluding noise (label = '-1') and finding the two clusters with the largest number of members
    unique_labels = list(set(labels))

    # Filtering out non structure samples
    spot_count = 0
    for unique_label in unique_labels:
        if unique_label != -1:
            spot_count = spot_count + 1

    # return "0" for values if detected number of clusters is below the user given threshold
    if spot_count < clust_n_thr:
        return 0, 0, 0, 0
    #Choose the two cluster with the highest number of points in them
    else:

        unique_labels_sample = []
        for i in range(len(unique_labels)):
            if unique_labels[i] != -1:
                unique_labels_sample.append(unique_labels[i])
        label_count = []
        for i in range(len(unique_labels_sample)):
            label_count.append(list(labels).count(unique_labels_sample[i]))

        label_joined_l = zip(label_count, unique_labels_sample)


        spots_loc_n = heapq.nlargest(2, label_count)
        # exception

        spot_label = list(zip(*heapq.nlargest(2, label_joined_l)))[1]

        # Sorting clustered localizations

        for i in range(len(centered_str_coor_a)):
            label = labels[i]
            if label == spot_label[0]:
                spot_a_locs.append([x for x in centered_str_coor_a[i]])
                spot_a_center[0] = spot_a_center[0] + centered_str_coor_a[i][0]
                spot_a_center[1] = spot_a_center[1] + centered_str_coor_a[i][1]
            else:
                if label == spot_label[1]:
                    spot_b_locs.append([x for x in centered_str_coor_a[i]])
                    spot_b_center[0] = spot_b_center[0] + centered_str_coor_a[i][0]
                    spot_b_center[1] = spot_b_center[1] + centered_str_coor_a[i][1]
                else:
                    noise_locs.append([x for x in centered_str_coor_a[i]])

        spot_a_center[0] = spot_a_center[0] / spots_loc_n[0]
        spot_a_center[1] = spot_a_center[1] / spots_loc_n[0]
        spot_b_center[0] = spot_b_center[0] / spots_loc_n[1]
        spot_b_center[1] = spot_b_center[1] / spots_loc_n[1]

        # Calculate dist.
        dist_px = math.sqrt(
            (math.pow((spot_a_center[0] - spot_b_center[0]), 2) + math.pow((spot_a_center[1] - spot_b_center[1]), 2)))
        dist_nm = dist_px * pix_size_nm

        if print_plots == 'y' and dist_nm < 80:
        # Render localizations
            thr_low, thr_high, gauss_size = render_param
            img_clr, loc_array_n = loc_to_img_crop(file_name=file_name, loc_array=centered_str_coor_a, window_size=window_size,
                                SPR=SPR_det, norm_max=norm_max, int_scale=int_scale, gauss_size=gauss_size, index=str_cnt)

            fig = plt.figure(dpi=300, tight_layout=True)
            fig.set_size_inches(16, 4, forward=True)

            gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1])

            ax0 = plt.subplot(gs[0])
            ax0.imshow(img_clr)
            ax0.axis("off")

            ax1 = plt.subplot(gs[1])
            ax1.scatter([x[1] for x in spot_a_locs + spot_b_locs + noise_locs], [x[0] for x in spot_a_locs + spot_b_locs + noise_locs], color='grey', s=5,
                        marker='x',linewidths=1)
            ax1.set_xlim(0, window_size / SPR_det)
            ax1.set_ylim(0, window_size / SPR_det)
            # this is for putting originvon top left corner
            ax1.invert_yaxis()
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax1.get_yticklabels(), visible=False)
            ax1.tick_params(axis='both', which='both', length=0)


            # Plot only clustered plot locs
            ax2 = plt.subplot(gs[2])
            ax2.scatter([x[1] for x in noise_locs], [x[0] for x in noise_locs], color='lightgrey', s=5, marker='x', label='noise',linewidths=1)
            ax2.scatter([x[1] for x in spot_a_locs], [x[0] for x in spot_a_locs], color='r', s=5, marker='x', label='Spot A',linewidths=1)
            ax2.scatter([x[1] for x in spot_b_locs], [x[0] for x in spot_b_locs], color='b', s=5, marker='x', label='Spot B',linewidths=1)
            ax2.scatter(spot_a_center[1], spot_a_center[0], s=20, c='y', marker='o', label='Spot A center')
            ax2.scatter(spot_b_center[1], spot_b_center[0], s=20, c='g', marker='o', label='Spot B center')
            ax2.set_xlim(0, window_size / SPR_det)
            ax2.set_ylim(0, window_size / SPR_det)
            # this is for putting origin on top left corner
            ax2.invert_yaxis()
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax2.get_yticklabels(), visible=False)
            ax2.tick_params(axis='both', which='both', length=0)
            ax2.legend()

            # Plot only clustered plot locs
            ax3 = plt.subplot(gs[3])
            ax3.scatter([x[1] for x in spot_a_locs], [x[0] for x in spot_a_locs], color='r', s=5, marker='x',linewidths=1)
            ax3.scatter([x[1] for x in spot_b_locs], [x[0] for x in spot_b_locs], color='b', s=5, marker='x',linewidths=1)
            ax3.scatter(spot_a_center[1], spot_a_center[0], s=20, c='y', marker='o')
            ax3.scatter(spot_b_center[1], spot_b_center[0], s=20, c='g', marker='o')
            ax3.annotate("", xy=(spot_a_center[1], spot_a_center[0]), xytext=(spot_b_center[1], spot_b_center[0]),
                     arrowprops=dict(arrowstyle="<|-|>", lw=3, color='grey',shrinkA=0, shrinkB=0), zorder=10)
            ax3.set_xlim(0, window_size / SPR_det)
            ax3.set_ylim(0, window_size / SPR_det)
            # this is for putting origin on top left corner
            ax3.invert_yaxis()
            plt.setp(ax3.get_xticklabels(), visible=False)
            plt.setp(ax3.get_yticklabels(), visible=False)
            ax3.tick_params(axis='both', which='both', length=0)

            plt.savefig(file_name[:-3] + '-Str-' + str(str_cnt) + '-plot-DBSCAN-only-spot-loc.jpg', dpi=900)
            plt.close()

        return spot_a_locs, spot_b_locs, noise_locs, dist_nm

#import modules used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Specify directories
import os, glob
folder_path = str(input("Path to folder with h5 file with extracted PAINT localization coordinates: "))
directory = folder_path
os.chdir(folder_path)

#Import loc file and transform it into a np array

#Input information needed for processing
for locs_file in glob.glob("*_pick_undrift-data.h5"):

    print ('Imported localization file : "' + str(locs_file) + '"')
#Import loc file and transform it into a np array
    locs_df = pd.read_hdf(locs_file, key='str_locs')

#Load parameters
    print('Parameters for localization visualisation:')
    mag = float(input("Magnification used to collect data: "))
    pix_nm = 13000.0 / mag
    SPR = int(input("Sub-pixelation for final images: (int)"))
    pix_nm_SPR = pix_nm / SPR
    window_size = int(input("Window size for rendering grouped localizations (pix): (int)"))
    norm_max = float(input("Maximum pixel intensity to normalize to: (float)"))
    int_scale = float(input("Maximum pixel intensity to scale to: (float)"))
    gauss_size = float(input("Kernel size for gaussian blurring: (float)"))
    render_param = [norm_max, int_scale, gauss_size]
    clust_n_thr, eps, min_samples = [float(x) for x in input("Parameters for DBSCAN clustering algorithm (expected number of spots, eps, min_samples): ").split()]
    print_plots = str(input("Print cropped out origami ROIs and localizations plots? (y/n)"))

    file1 = open(locs_file[:-3] + '-dist-ref-processing-parameters.txt', "w")
    L = ["Parameters used for protein quantification:\n",
         "Sub-pixelation rate :\n", str(SPR) + "\n", "Visualization window size:\n", str(window_size) + "\n",
         "Maximum pixel intensity to normalize to:\n", str(norm_max) + "\n",
         "Maximum pixel intensity to scale to:\n", str(int_scale) + "\n",
         "Gaussian kernel size for blurring:\n", str(gauss_size) + "\n",
         "Parameters used for DBSCAN clustering algorithm (expected number of spots, eps, min_samples): " + str(clust_n_thr) + ", " + str(eps)+ ", " + str(min_samples) +"\n"]
    file1.writelines(L)
    file1.close()  # to change file access modes

    print('Finding number of stored structures')

    #
    indeces =  []
    for index in set(locs_df['Str_index'].values.tolist()):
        indeces.append(index)

    str_index = []  # Structure index
    str_pos = []  # Structure position
    str_loc = []  # Localizations
    Dist = []  # Spot dist in nm

    print('Exporting images of structures')
    from tqdm import tqdm

    with tqdm(total=len(indeces)) as pbar:
        for i in range(len(indeces)):
            pbar.update(1)

            ind = indeces[i]
            df_ind = locs_df.loc[(ind == locs_df['Str_index'])]
            ROI_origami_det = df_ind['Origami_det'].values.tolist()[0]
            ROI_origami_pos = df_ind['Str_ROI_coor'].values.tolist()[0]

            if ROI_origami_det == 1:
                x_coor = df_ind['loc_coor_x'].values.tolist()[0]
                y_coor = df_ind['loc_coor_y'].values.tolist()[0]
                str_locs = []
                spots_coor_a = list(zip(x_coor, y_coor))
                spot_a_locs, spot_b_locs, noise_locs, dist_nm = loc_dbscan_clust(str_coor_a=spots_coor_a,
                                                                                       SPR_det=SPR,
                                                                                       clust_n_thr=clust_n_thr, eps=eps,
                                                                                       min_samples=min_samples,
                                                                                       print_plots=print_plots,
                                                                                       pix_size_nm=pix_nm,
                                                                                       window_size=window_size,
                                                                                       render_param=render_param,
                                                                                       str_cnt=ind,
                                                                                       file_name=locs_file)

                if spot_a_locs != 0 and spot_b_locs != 0 and noise_locs != 0 and dist_nm != 0:

                    # Exporting values:
                    # Structure index
                    str_index.append(ind)

                    # Structure position
                    str_pos.append(ROI_origami_pos)

                    # Localizations:
                    str_loc.append([spot_a_locs, spot_b_locs, noise_locs])

                    # Spot dist in nm
                    Dist.append(dist_nm)

        # Export data
        df = pd.DataFrame(list(zip(str_index, str_pos, str_loc, Dist)),
                          columns=['Str_index', 'str_pos', 'str_loc', 'dist'])

        df.to_hdf(locs_file[:-5] + '-str-dist.h5', key='str_locs', mode='w')

