# Function for importing data calculated from reference structures
def import_ref_data(folder_path):

    #Import packages used
    import csv

    #Open csv file containing reference frequency and intensity interval values
    with open(folder_path + '/Exported_site_reference_values.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        exported_data = []
        for row in spamreader:
            exported_data.append(row)

        #Export mean event frequency (sec^-1)
        site_d_nm_mean = float(exported_data[2][1])
        site_d_nm_std = float(exported_data[2][2])
        site_spread_nm_mean = float(exported_data[3][1])
        site_spread_nm_std = float(exported_data[3][2])

    return site_d_nm_mean, site_d_nm_std, site_spread_nm_mean, site_spread_nm_std

#Function to generate a cropped image from the extracted structure localization coordinates
def loc_to_img_crop(directory, loc_array, window_size, SPR, norm_max, int_scale, gauss_size, index):
    #import packages
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import os, errno
    import cv2

    new_dir = directory + '\Images_for_prot_det'
    try:
        os.makedirs(new_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

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
    img_min = np.amin(np.where(img>0, img, np.inf).min(axis=1))
    img = norm_max * ((img - img_min) / img_max)
    img[img < 0.0] = 0.0


    #Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img, (int(gauss_size), int(gauss_size)), 0)
    scale_factor = int_scale / np.amax(img_blur)
    img_blur = img_blur * scale_factor

    cv2.imwrite(new_dir + '\Render-crop-GS-re-dyn-' + str(index) + '.png', img_blur)
    cv2.destroyAllWindows()

    image = cv2.imread(new_dir + '\Render-crop-GS-re-dyn-' + str(index) + '.png', 0)
    im_color = cv2.applyColorMap(image, cv2.COLORMAP_HOT)

    cv2.imwrite(new_dir + '\Render-crop-CLR-re-dyn-'+str(index)+'.png',im_color)
    cv2.destroyAllWindows()

    img_color_file = new_dir + '\Render-crop-CLR-re-dyn-'+str(index)+'.png'

    return image, im_color, img_color_file, new_dir, loc_array_n

#Function to calculate the square of dist of two points
def Dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

#Function to calcualte the dist of two points
def Dist(p1, p2):
    import numpy as np
    return np.sqrt(Dist2(p1, p2))

#Function to merge points clustered in a certain distance to a mean point
def fuse2(points, d):
    ret_exp = []
    d2 = d * d
    len_l = [len(points)]
    check = 0
    while check == 0:
        ret = []
        n = len(points)
        taken = [False] * n
        for i in range(n):
            if not taken[i]:
                count = 1
                point = [points[i][0], points[i][1]]
                taken[i] = True
                for j in range(i+1, n):
                    if Dist2(points[i], points[j]) < d2:
                        point[0] += points[j][0]
                        point[1] += points[j][1]
                        count+=1
                        taken[j] = True
                point[0] /= count
                point[1] /= count
                ret.append((point[0], point[1]))
        ret_exp = ret
        points = ret
        len_l.append(len(points))
        if len_l[-1] == len_l[-2]:
            check = 1

    return ret_exp

def get_cmap(n, name):
    import matplotlib.pyplot as plt
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

#Function to calculate centroid coordinate from a set of x,y coordinates
def centroid_calc(data):
    centroid_l = []
    for i in range(len(data)):
        x = sum(h[0] for h in data[i] if h[0] is not None) / len(data[i])
        y = sum(h[1] for h in data[i] if h[1] is not None) / len(data[i])
        cent_1 = [x, y]
        centroid_l.append(cent_1)
    return centroid_l

#Function to annotate points (Find external points -> name one as first -> annotate rest by using point by point dist
def point_annot(coor):
    # import packages used
    import numpy as np

    #Check number of points in structure
    n_coor = len(coor)

    #Generate distance matrix
    dist_mat = np.zeros((n_coor, n_coor))
    for i in range(len(dist_mat)):
        for j in range(len(dist_mat[i])):
            dist_mat[i][j] = dist_mat[i][j] + Dist(coor[i], coor[j])

    #Annotate points based on site distances and point number
    #For structure with one detected spot
    if n_coor == 1:
        indeces = list(range(0, 1))
        i1 = indeces[0]
        ord_coor = [coor[i1]]
        dist_l = [Dist(coor[i1], coor[i1])]
        dist_tot = [Dist(coor[i1], coor[i1])]

    #For structures with more than one spot
    else:
        #Make empty array to store point indeces in order
        indeces = np.zeros(n_coor)
        #find index of pos 1 from extracting the index of the maximum distance in the dist matrix
        i = np.unravel_index(np.argmax(dist_mat, axis=None), dist_mat.shape)
        indeces[0] += i[0]
        dist_p1_sorted = np.sort(dist_mat[i[0]])
        dist_p1_l = dist_mat[i[0]].tolist()
        for i in range(n_coor-1):
            d = dist_p1_sorted[i+1]
            index =dist_p1_l.index(d)
            indeces[i+1] += index

        #Creating a list of ordered coordinates and point to point distances for exporting
        ord_coor = []
        dist_l = []

        for i in range(len(indeces)):
            index = int(indeces[i])
            ord_coor.append(coor[index])

        for i in range(len(indeces)-1):
            index1 = int(indeces[i])
            index2 = int(indeces[i+1])
            dist_l.append((Dist(coor[index1], coor[index2])))


        index1 = int(indeces[0])
        index2 = int(indeces[-1])
        dist_tot = Dist(coor[index1], coor[index2])

    return ord_coor, dist_l, dist_tot

#Function to check if distance passed
def dist_thr(dist_l, d_mean, d_std):
    #
    check = 0
    for dist in dist_l:
        if dist < d_mean - d_std or d_mean + d_std < dist:
            check = check + 1

    return check

#Function to calculate vector and normalized vector between two points
def vect_calc(p1, p2):
    #import packages used
    import numpy as np

    #Calculate vector coordinates
    x1, y1 = p1
    x2, y2 = p2
    v_x = x2-x1
    v_y = y2-y1

    #Calcualte vector length
    l = np.sqrt(v_x**2 + v_y**2)

    #Calculate normalized vector coordinates
    v = [v_x, v_y]
    v_n = [v_x/l, v_y/l]
    return(v, v_n)

#Function to calculate linearity of set of points (STD of normalized vectors between adjacent points)
def lin_check(ann_spot_coor):
    # import packages used
    import numpy as np

    #
    v_l = []
    v_n_l = []

    # Calculate vectors from ref point to every point
    for i in range(len(ann_spot_coor) - 1):
        p1 = ann_spot_coor[i]
        p2 = ann_spot_coor[i+1]
        v, v_n = vect_calc(p1=p1, p2=p2)
        v_l.append(v)
        v_n_l.append(v_n)

    # Cal
    v_n_x, v_n_y = map(list, zip(*v_n_l))
    score = (np.std(v_n_x) + np.std(v_n_y))/2

    return score, v_l, v_n_l

def rot_angle_calc(p1,p2):
    #import packages used
    import numpy as np

    #Define reference point (p3) to define axis to rotate to
    x1,y1 = p1
    x2,y2 = p2
    p3 = [x1, y2]

    #Calculate distances between point for calculating angle between the initial axis and final
    d1 = Dist(p1, p2)
    d2 = Dist(p1, p3)
    angle = np.degrees(np.arccos(d2/d1))

    #Determine rotation angle based on relative position of original points
    if x2 > x1:
        if y2 > y1:
            rot_angle = angle * -1

        else:
            rot_angle = (180 - angle)*-1

    else:
        if y2 > y1:
            rot_angle = angle

        else:
            rot_angle = 180-angle

    return rot_angle

def rot_trans_loc(loc_list, SPR, trans_coor, origin, angle):
    import numpy as np

    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    mod_loc_list = []

    # Rotate
    angle = np.deg2rad(angle)

    x_trans, y_trans = trans_coor

    for loc in loc_list:
        loc_x, loc_y = loc

        #Translate

        loc_x_tr = loc_x*SPR + x_trans
        loc_y_tr = loc_y*SPR + y_trans

        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d([loc_x_tr, loc_y_tr])
        p_n = np.squeeze((R @ (p.T - o.T) + o.T).T)

        mod_loc_list.append(p_n)

    return mod_loc_list

def prot_det (test, red, SPR, processing_dir, img_gs, img_color_file, site_n_thr, int_thr, d_thr_site_nm, d_thr_str_nm, index, site_d, pix_nm_SPR):

    # Open png files in dir
    from skimage.feature import peak_local_max
    import cv2
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches


    #Open color image
    img_clr = mpimg.imread(img_color_file)

    #Calculate size of image
    dim = len(img_clr)
    c_im= [dim/2,dim/2]

    #Make binary image of gs image
    ret, thresh3 = cv2.threshold(img_gs, int_thr, 255, cv2.THRESH_TOZERO)

    #Find local maxima in image
    xy = peak_local_max(thresh3)

    #Merge local maxima using a dist thr
    dist_thr_pix = d_thr_site_nm/pix_nm_SPR
    new_peak_list = fuse2(xy, dist_thr_pix)

    #Remove maxima far from center of ROI
    max_d = d_thr_str_nm / pix_nm_SPR
    filt_coor = [x for x in new_peak_list if Dist(c_im, x)< max_d]

    #Cluster peaks into structures using a second distance metric
    if len(filt_coor) == site_n_thr:

        ord_coor, dist_l, dist_tot = point_annot(coor=filt_coor)
        lin_score, v_l, v_n_l = lin_check(ann_spot_coor=ord_coor)

        #Normalize coordinates:
        for i in range(len(filt_coor)):
            filt_coor[i]=list(filt_coor[i])
            filt_coor[i][0] = filt_coor[i][0]/SPR
            filt_coor[i][1] = filt_coor[i][1]/SPR

        # Normalize coordinates:
        for i in range(len(ord_coor)):
            ord_coor[i] = list(ord_coor[i])
            ord_coor[i][0] = ord_coor[i][0] / SPR
            ord_coor[i][1] = ord_coor[i][1] / SPR

        if test == 'y' and len(ord_coor)<8:
            from matplotlib import gridspec

            if red == 'n':
                fig = plt.figure(dpi=300, tight_layout=True)
                fig.set_size_inches(16,8, forward=True)

                gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1,1])

            else:
                fig = plt.figure(dpi=300, tight_layout=True)
                fig.set_size_inches(20, 4, forward=True)

                gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1])

            ax0 = plt.subplot(gs[0])
            ax0.imshow(img_clr)
            ax0.axis("off")

            if red == 'n':
                ax1 = plt.subplot(gs[1])
                ax1.imshow(img_clr)
                ax1.axis("off")
                for point in xy:
                    x = point[1]
                    y = point[0]
                    ax1.plot(x, y, 'bx')

                ax2 = plt.subplot(gs[2])
                ax2.imshow(img_clr)
                ax2.axis("off")

                for point in new_peak_list:
                    x = point[1]
                    y = point[0]
                    ax2.plot(x, y, 'gx')

                ax3 = plt.subplot(gs[3])
                ax3.imshow(img_clr)
                ax3.axis("off")

                for i in range(len(new_peak_list)):
                    point = new_peak_list[i]
                    x = point[1]
                    y = point[0]

                    ax3.plot(x, y, 'x', color='g')

                for i in range(len(filt_coor)):
                    point = filt_coor[i]
                    x = point[1]*SPR
                    y = point[0]*SPR

                    ax3.plot(x, y, 'x', color='cyan')
                circle1 = mpatches.Circle(c_im, max_d, linestyle='-.', color='cyan', fill=False)

                ax3.add_patch(circle1)

                ax4 = plt.subplot(gs[4])
                ax4.imshow(img_clr)
                ax4.axis("off")

                for i in range(len(filt_coor)):
                    point = filt_coor[i]
                    x = point[1]*SPR
                    y = point[0]*SPR

                    ax4.plot(x, y, 'x', color='cyan')

                ax5 = plt.subplot(gs[5])
                ax5.imshow(img_clr)
                ax5.axis("off")

                for i in range(len(ord_coor)):
                    point = ord_coor[i]
                    x = point[1] * SPR
                    y = point[0] * SPR
                    ax5.plot(x, y, 'x', color='cyan')
                    ax5.annotate(i + 1, (x, y), color='magenta',
                                 xytext=(2, 2), textcoords='offset points', )

                ax6 = plt.subplot(gs[6])
                ax6.imshow(img_clr,zorder=0)
                ax6.axis("off")

                lb_list = ['1st to 2nd', '2nd to 3rd', '3rd to 4th', '4th to 5th', '5th to 6th', '7th to 8th']
                colors = plt.cm.get_cmap('tab10', len(lb_list))
                colors_l = colors.colors

                for i in range(len(ord_coor)):
                    point = ord_coor[i]
                    x = point[1] * SPR
                    y = point[0] * SPR
                    ax6.plot(x, y, 'x', color='cyan',zorder=5)

                for i in range(len(v_l)):
                    v_x, v_y = v_l[i]
                    px, py = ord_coor[i]
                    x = [px* SPR, px* SPR + v_x]
                    y = [py* SPR, py* SPR + v_y]
                    ax6.annotate("", xy=(y[1], x[1]), xytext=(y[0], x[0]),
                                 arrowprops=dict(arrowstyle="->",lw=3, color=colors_l[i],shrinkA=0, shrinkB=0),zorder=10)

                ax7 = plt.subplot(gs[7])
                for i in range(len(v_n_l)):
                    vec_norm = v_n_l[i]
                    v_x, v_y = vec_norm
                    x = [0, v_x]
                    y = [0, v_y]
                    ax7.arrow(y[0], x[0], y[1], x[1], label=lb_list[i], ec=colors(i),fc=colors(i), width=0.01, head_width=0.03, alpha=0.8)

                ax7.legend()
                ax7.set_xlim(-1*(dim/(2*site_d)), (dim/(2*site_d)))
                ax7.set_ylim(-1*(dim/(2*site_d)), (dim/(2*site_d)))
                ax7.invert_yaxis()

            else:
                ax1 = plt.subplot(gs[1])
                ax1.imshow(img_clr)
                ax1.axis("off")

                for i in range(len(filt_coor)):
                    point = filt_coor[i]
                    x = point[1] * SPR
                    y = point[0] * SPR

                    ax1.plot(x, y, 'x', color='cyan')

                ax2 = plt.subplot(gs[2])
                ax2.imshow(img_clr)
                ax2.axis("off")

                for i in range(len(ord_coor)):
                    point = ord_coor[i]
                    x = point[1] * SPR
                    y = point[0] * SPR
                    ax2.plot(x, y, 'x', color='cyan')
                    ax2.annotate(i + 1, (x, y), color='magenta',
                                 xytext=(2, 2), textcoords='offset points', )

                ax3 = plt.subplot(gs[3])
                ax3.imshow(img_clr,zorder=0)
                ax3.axis("off")

                lb_list = ['1st to 2nd', '2nd to 3rd', '3rd to 4th', '4th to 5th', '5th to 6th', '7th to 8th']
                colors = plt.cm.get_cmap('tab10', len(lb_list))
                colors_l = colors.colors

                for i in range(len(ord_coor)):
                    point = ord_coor[i]
                    x = point[1] * SPR
                    y = point[0] * SPR
                    ax3.plot(x, y, 'x', color='cyan',zorder=5)

                for i in range(len(v_l)):
                    v_x, v_y = v_l[i]
                    px, py = ord_coor[i]
                    x = [px * SPR, px * SPR + v_x]
                    y = [py * SPR, py * SPR + v_y]
                    ax3.annotate("", xy=(y[1], x[1]), xytext=(y[0], x[0]),
                                 arrowprops=dict(arrowstyle="->",lw=3, color=colors_l[i],shrinkA=0, shrinkB=0),zorder=10)

                ax4 = plt.subplot(gs[4])
                for i in range(len(v_n_l)):
                    vec_norm = v_n_l[i]
                    v_x, v_y = vec_norm
                    x = [0, v_x]
                    y = [0, v_y]
                    ax4.arrow(y[0], x[0], y[1], x[1], label=lb_list[i], ec=colors(i), fc=colors(i), width=0.01,
                              head_width=0.03, alpha=0.8)

                ax4.legend()
                ax4.set_xlim(-1 * (dim / (2 * site_d)), (dim / (2 * site_d)))
                ax4.set_ylim(-1 * (dim / (2 * site_d)), (dim / (2 * site_d)))
                ax4.invert_yaxis()


            plt.savefig(processing_dir + '\Protein_detection_plot-'+str(index)+'.eps')
            plt.close()

    else:
        if test == 'y':

            from matplotlib import gridspec

            fig = plt.figure(figsize=(28, 4))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], height_ratios=[1])

            ax0 = plt.subplot(gs[0])
            ax0.imshow(img_clr)
            ax0.axis("off")

            ax1 = plt.subplot(gs[1])
            ax1.imshow(img_clr)
            ax1.axis("off")
            for point in xy:
                x = point[1]
                y = point[0]
                ax1.plot(x, y, 'bx')

            ax2 = plt.subplot(gs[2])
            ax2.imshow(img_clr)
            ax2.axis("off")

            for point in new_peak_list:
                x = point[1]
                y = point[0]
                ax2.plot(x, y, 'gx')
            plt.savefig(processing_dir + '\Protein_detection_plot-'+str(index)+'.eps')
            plt.close()

        ord_coor = new_peak_list
        for i in range(len(ord_coor)):
            ord_coor[i] = list(ord_coor[i])
            ord_coor[i][0] = ord_coor[i][0] / SPR
            ord_coor[i][1] = ord_coor[i][1] / SPR

        lin_score = 0.0
        dist_tot = 0.0

    return ord_coor, lin_score, dist_tot

#import modules used
import pandas as pd

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
    locs_df = pd.read_hdf(locs_file)

#Load parameters
    print('Parameters for localization visualisation:')
    mag = float(input("Magnification used to collect data: "))
    pix_nm = 13000.0 / mag
    SPR = int(input("Sub-pixelation for final images: (int)"))
    pix_nm_SPR = pix_nm / SPR
    window_size = int(input("Window size for rendering grouped localizations (pix): (int)"))
    norm_max = float(input("Maximum pixel intensity to normalize to: (float)"))
    int_scale = float(input("Maximum pixel intensity to scale to: (float)"))
    gauss_size = float(input("Kernel size for gaussian blurring (pix): (float)"))
    int_thr = int(input("Intensity threshold for spot detection (0-255): (int)"))

    print('Parameters for protein detection:')
    # Specify directories
    ref_folder_path = str(input("Path to folder with exported reference site distance values: "))

    # Import reference values
    site_d_nm_mean, site_d_nm_std, site_spread_nm_mean, site_spread_nm_std = import_ref_data(folder_path=ref_folder_path)

    site_n_limit =  int(input("Designed number of sites: (int)"))
    site_d_pix = site_d_nm_mean / pix_nm_SPR
    site_d_max_pix = (site_d_nm_mean + (site_d_nm_std*3)) / pix_nm_SPR
    tot_d_thr_pix = site_d_max_pix * site_n_limit
    site_sp_pix = site_spread_nm_mean / pix_nm_SPR
    d_thr_site_x = float(input("Distance threshold for maxima clustering (x * site distance): (float)"))
    d_thr_site_nm = site_d_nm_mean * d_thr_site_x
    d_thr_str_x = float(input("Distance threshold for peak clustering (x * site distance): (float)"))
    d_thr_str_nm = site_d_nm_mean * d_thr_str_x


    file1 = open(locs_file[:-3] + '-processing-parameters.txt', "w")
    L = ["Parameters used for protein quantification:\n",
         "Sub-pixelation rate :\n", str(SPR) + "\n", "Visualization window size:\n", str(window_size) + "\n",
         "Maximum pixel intensity to normalize to:\n", str(norm_max) + "\n",
         "Maximum pixel intensity to scale to:\n", str(int_scale) + "\n",
         "Gaussian kernel size for blurring:\n", str(gauss_size) + "\n",
         "Intensity threshold for spot detection:\n", str(int_thr) + "\n",
         "Designed number of sites:\n", str(site_n_limit) + "\n",
         "Designed distance of sites:\n", str(site_d_nm_mean) + "\n",
         "Measured spread of sites in nm:\n", str(site_spread_nm_mean) + "\n",
         "Distance threshold for maxima clustering (x * site distance):\n", str(d_thr_site_x) + "\n",
         "Distance threshold for peak clustering (x * site distance):\n", str(d_thr_str_x) + "\n"]

    file1.writelines(L)
    file1.close()  # to change file access modes

    t = str(input("Test run?: (y/n)"))
    if t == 'y':
        red = str(input("Reduced plot format? (Missing peak detection steps): (y/n)"))
    else:
        red = 'n'
    print('Finding number of stored structures')

    #
    indeces =  []
    for index in set(locs_df['Str_index'].values):
        indeces.append(index)

    str_index_l = []
    str_loc_x_l = []
    str_loc_y_l = []
    str_lin_score = []
    img_clr_l = []
    spot_n_l = []

    print('Exporting images of structures')
    from tqdm import tqdm

    with tqdm(total=len(indeces)) as pbar:
        for i in range(len(indeces)):
            pbar.update(1)

            ind = indeces[i]
            df_ind = locs_df.loc[(ind == locs_df['Str_index'])]
            ROI_origami_det = df_ind['Origami_det'].values.tolist()[0]

            if ROI_origami_det == 0:
                str_index_l.append(ind)
                str_lin_score.append('NaN')
                str_loc_x_l.append([])
                str_loc_y_l.append([])
                img_clr_l.append([])
                spot_n_l.append(0)

            else:
                loc_l = []
                x_coor = df_ind['loc_coor_x'].values.tolist()[0]
                y_coor = df_ind['loc_coor_y'].values.tolist()[0]
                str_locs = []

                for j in range(len(x_coor)):
                    str_locs.append([x_coor[j], y_coor[j]])

                img_gs, im_color, img_color_file, processing_dir, loc_array_n = loc_to_img_crop(directory=directory, loc_array=str_locs, window_size=window_size, SPR=SPR, norm_max=norm_max, int_scale=int_scale, gauss_size=gauss_size, index=ind)
                new_peak_list, lin_score, dist_tot = prot_det(SPR=SPR, test = t, red=red,  processing_dir=processing_dir, img_gs=img_gs, img_color_file=img_color_file, site_n_thr =  site_n_limit, int_thr=int_thr, d_thr_site_nm=d_thr_site_nm, d_thr_str_nm=d_thr_str_nm, index=ind, site_d =site_d_nm_mean, pix_nm_SPR = pix_nm_SPR)

                str_index_l.append(ind)
                str_lin_score.append(lin_score)
                str_loc_x_l.append(x_coor)
                str_loc_y_l.append(y_coor)
                img_clr_l.append(im_color)
                spot_n_l.append(len(new_peak_list))


    df1 = pd.DataFrame(list(zip(str_index_l,str_loc_x_l,str_loc_y_l,img_clr_l, str_lin_score, spot_n_l)),columns=['Str_index','Str_loc_coor_x', 'Str_loc_coor_y', 'Str_clr_img', 'Str_lin_score', 'Str_spot_n'])

    df1.to_hdf(locs_file[:-3] + '-lin-data.h5', key='str_locs', mode='w')
