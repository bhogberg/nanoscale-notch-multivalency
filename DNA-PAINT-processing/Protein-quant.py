def crop_image(img , crop_w, crop_h):

    # Determine original image dimensions
    rows, cols, channels = img.shape
    x_cent = cols / 2
    y_cent = rows / 2
    w_x1 = int(x_cent - (crop_w / 2))
    w_x2 = int(x_cent + (crop_w / 2))
    w_y1 = int(y_cent - (crop_h / 2))
    w_y2 = int(y_cent + (crop_h / 2))

    crop_coor = [w_y1/SPR, w_y2/SPR, w_x1/SPR, w_x2/SPR]
    crop_img = img[w_y1:w_y2, w_x1:w_x2]

    return crop_img, crop_coor

# Function for importing data calculated from reference structures
def import_ref_data(folder_path, file_type):

    #Import packages used
    import csv
    if file_type == 'site_ref':
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

    if file_type == 'lin_ref':
        # Open csv file containing reference frequency and intensity interval values
        with open(folder_path + '/Exported_linearity_reference_values.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            exported_data = []
            for row in spamreader:
                exported_data.append(row)

            # Export mean event frequency (sec^-1)
            lin_score_thr = float(exported_data[2][1])
            
        return lin_score_thr

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

def prot_det (test, SPR, processing_dir, img_gs, img_color_file, int_thr, d_thr_site_nm, d_thr_str_nm, index, site_d, site_n_limit,  pix_nm_SPR):

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
    spot_n = len(new_peak_list)

    #Remove maxima far from center of ROI
    max_d = d_thr_str_nm / pix_nm_SPR
    filt_coor = [x for x in new_peak_list if Dist(c_im, x)< max_d]

    #Cluster peaks into structures using a second distance metric
    if spot_n >= site_n_limit+1:
        ord_coor = 'NaN'
        lin_score = 'NaN'
        dist_tot = 'NaN'

    else:
        
        if site_n_limit+1 > len(filt_coor) > 1:

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

            if test == 'y':

                from matplotlib import gridspec

                fig = plt.figure(figsize=(50,4))
                gs = gridspec.GridSpec(1, 8, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1], height_ratios=[1])

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
                                 arrowprops=dict(arrowstyle="->", lw=3, color=colors_l[i], shrinkA=0, shrinkB=0), zorder=10)

                ax7 = plt.subplot(gs[7])

                for i in range(len(v_n_l)):
                    vec_norm = v_n_l[i]
                    v_x, v_y = vec_norm
                    x = [0, v_x]
                    y = [0, v_y]
                    ax7.arrow(y[0], x[0], y[1], x[1], label=lb_list[i], ec=colors(i), fc=colors(i), width=0.01,
                              head_width=0.03, alpha=0.8)

                ax7.legend()
                ax7.set_xlim(-1*(dim/(2*site_d)), (dim/(2*site_d)))
                ax7.set_ylim(-1*(dim/(2*site_d)), (dim/(2*site_d)))
                ax7.invert_yaxis()
                #plt.savefig(processing_dir + '\Protein_detection_plot-'+str(index)+'.jpg')
                plt.savefig(processing_dir + '\Protein_detection_plot-' + str(index) + '.eps')
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
                #plt.savefig(processing_dir + '\Protein_detection_plot-'+str(index)+'.jpg')
                plt.savefig(processing_dir + '\Protein_detection_plot-' + str(index) + '.eps')
                plt.close()

            ord_coor = new_peak_list
            for i in range(len(ord_coor)):
                ord_coor[i] = list(ord_coor[i])
                ord_coor[i][0] = ord_coor[i][0] / SPR
                ord_coor[i][1] = ord_coor[i][1] / SPR

            lin_score = 0.0
            dist_tot = 0.0

    return ord_coor, lin_score, dist_tot


def align_and_quant(img_clr, img_gs, ord_coor, loc_array, loc_frames, SPR, pix_gap, crop_w_w, crop_w_h, spot_d, sweep_w_perc, site_n, min_peak_prom,
                    test, str_index):
    # import packages used
    import numpy as np
    import copy
    from scipy.signal import find_peaks

    #
    str_index = str_index

    # Determine original image dimensions
    rows, cols, channels = img_clr.shape
    x_cent = cols / 2
    y_cent = rows / 2
    p_cent = [x_cent, y_cent]

    # If number of spots are less than max try both orientation of annotation
    ord_coor_l = [ord_coor, ord_coor[::-1]]
    results = [[], []]

    for i in range(len(ord_coor_l)):

        ord_coor = ord_coor_l[i]
        # Fetch coordinates for first spot
        y1, x1 = [item * SPR for item in ord_coor[0]]

        # Calculate centroid for structure
        str_cent_x = np.mean([item[1] * SPR for item in ord_coor])
        str_cent_y = np.mean([item[0] * SPR for item in ord_coor])

        # Calculate vector for translation
        x_v = x_cent - str_cent_x
        y_v = y_cent - str_cent_y
        trans_coor = [y_v, x_v]
        results[i].append([trans_coor])

        # Translate image
        M_t = np.float32([
            [1, 0, x_v],
            [0, 1, y_v]])

        shifted = cv2.warpAffine(img_clr, M_t, (rows, cols))
        shifted_gs = cv2.warpAffine(img_gs, M_t, (rows, cols))

        results[i].append(shifted)

        x1_n = x1 + x_v
        y1_n = y1 + y_v
        p1_n = [x1_n, y1_n]

        p1_c_d = Dist(p1_n, p_cent)

        # Rotate image
        if len(ord_coor) > 1:

            y2, x2 = [item * SPR for item in ord_coor[-1]]
            # calculate translated coordinates

            x2_n = x2 + x_v
            y2_n = y2 + y_v
            p2_n = [x2_n, y2_n]

            # calculate angle of rotation
            rot_angle = rot_angle_calc(p1=p1_n, p2=p2_n)
            rot_list = [rot_angle, [x_cent, y_cent]]
            results[i][0].append(rot_list)

            # rotate image
            M_r = cv2.getRotationMatrix2D((x_cent, y_cent), rot_angle, 1)
            rotated = cv2.warpAffine(shifted, M_r, (cols, rows))
            rotated_gs = cv2.warpAffine(shifted_gs, M_r, (cols, rows))

            results[i].append(rotated)

        else:
            rot_list = [0, [x_cent, y_cent]]
            results[i][0].append(rot_list)
            rotated = shifted
            rotated_gs = shifted_gs

            results[i].append(rotated)

        # Crop image
        crop_w = crop_w_w
        crop_h = crop_w_h

        w_x1 = int((cols / 2) - (crop_w / 2))
        w_x2 = int((cols / 2) + (crop_w / 2))
        w_y1 = int(y_cent - p1_c_d - pix_gap)
        w_y2 = int(w_y1 + crop_h)
        results[i][0].append([w_x1, w_x2, w_y1, w_y2])

        crop = rotated[w_y1:w_y2, w_x1:w_x2]
        crop2 = copy.deepcopy(crop)
        crop_gs = rotated_gs[w_y1:w_y2, w_x1:w_x2]

        results[i].append(crop)

        # Define final spot spot areas
        p1_final = [(cols / 2) - w_x1, y_cent - p1_c_d - w_y1]
        spot_r = spot_d / 2

        spot_1_p1 = [p1_final[0] - spot_r, p1_final[1] - spot_r]
        spot_1_p2 = [p1_final[0] + spot_r, p1_final[1] + spot_r]

        spot_pos = [[spot_1_p1, spot_1_p2]]
        for j in range(site_n - 1):
            spot_pos.append(
                [[spot_pos[-1][0][0], spot_pos[-1][0][1] + spot_d], [spot_pos[-1][1][0], spot_pos[-1][1][1] + spot_d]])


        colors = [(255, 255, 0),(0, 255, 0),(0, 255, 255),(255, 0, 255)]
        for j in range(len(spot_pos)):
            cv2.rectangle(crop2, (int(spot_pos[j][0][0])+1, int(spot_pos[j][0][1])+1),
                          (int(spot_pos[j][1][0])-1, int(spot_pos[j][1][1])-1), colors[j], 1)

        results[i].append(crop2)
        results[i].append(spot_pos)

        # Check site occupancy
        spot_occ_l = []
        for j in range(site_n):
            spot_occ_l.append(0)

            # Find maxima y coordinates
            y_lines = []
            sweep_w = spot_r * sweep_w_perc
            for h in range(int(p1_final[0] - sweep_w) + 1, int(p1_final[0] + sweep_w) - 1):
                y_line = crop_gs[:, h]
                y_lines.append(y_line)

            y_lines_sum = list(map(sum, map(lambda l: map(float, l), zip(*y_lines))))

            y_peaks, _ = find_peaks(y_lines_sum, prominence=min_peak_prom)

            spot_y_peak_l = []
            for h in range(site_n):
                spot_y_peak_l.append([])

            for h in range(len(y_peaks)):
                for k in range(len(spot_pos)):
                    y_peak = y_peaks[h]
                    if spot_pos[k][0][1] < y_peak < spot_pos[k][1][1]:
                        spot_y_peak_l[k].append(y_peak)
                        continue

            spot_y_peak_l_joined = []

            for h in range(site_n):
                spot_y_peak_l_joined.append([])

            for h in range(len(spot_y_peak_l)):
                y_pos = spot_y_peak_l[h]
                if len(y_pos) > 0:
                    spot_y_peak_l_joined[h].append(int(np.mean(y_pos)))
                else:
                    continue
        results[i].append([y_lines_sum, spot_y_peak_l_joined])

        #Find maxima x coordinates
        x_lines = []
        for y_peak in spot_y_peak_l:
            if len(y_peak) != 0:
                y= y_peak[0]
                x_lines.append(crop_gs[y, :])
            else:
                x_lines.append([])

        spot_x_peak_l = []
        for j in range(site_n):
            spot_x_peak_l.append([])

        for j in range(len(x_lines)):
            x_line = x_lines[j]
            if len(x_line) != 0:
                x_peaks, _ = find_peaks(x_line, prominence=min_peak_prom)
                for h in range(len(x_peaks)):
                    x_peak = x_peaks[h]
                    if spot_pos[j][0][0] < x_peak < spot_pos[j][1][0]:
                        spot_x_peak_l[j].append(x_peak)
                        continue

        spot_x_peak_l_joined = []
        for j in range(site_n):
            spot_x_peak_l_joined.append([])

        for j in range(len(spot_x_peak_l)):
            x_pos = spot_x_peak_l[j]
            if len(x_pos) > 0:
                spot_x_peak_l_joined[j].append(int(np.mean(x_pos)))
            else:
                continue

        results[i].append([x_lines, spot_x_peak_l_joined])


        spot_peak_l = []
        final_spot_pos = []

        for j in range(len(spot_x_peak_l_joined)):
            x_pos = spot_x_peak_l_joined[j]
            y_pos = spot_y_peak_l_joined[j]
            if len(x_pos) > 0 and len(y_pos) > 0:
                spot_peak_l.append([x_pos[0], y_pos[0]])
                spot_occ_l[j] = 1
                point1 = [x_pos[0]- spot_r, y_pos[0]-spot_r]
                point2 = [x_pos[0]+ spot_r, y_pos[0]+spot_r]
                final_spot_pos.append([point1, point2])

            else:
                spot_peak_l.append([])
                final_spot_pos.append(spot_pos[j])


        results[i].append(spot_peak_l)
        results[i].append(final_spot_pos)
        results[i].append(spot_occ_l)

    spot_n_l = []
    for i in range(len(results)):
        spot_n_l.append(sum(results[i][-1]))

    index = spot_n_l.index(max(spot_n_l))
    final_results = results[index]
    final_ord_coor = ord_coor_l[index]

    trans_coor, rot_list, crop_w_list = final_results[0]
    rot_angle, rot_cent = rot_list
    x_min, x_max, y_min, y_max = crop_w_list
    shifted_final = final_results[1]
    rotated_final = final_results[2]
    crop_final = final_results[3]
    crop_w_spots_final = final_results[4]
    spot_pos = final_results[5]
    y_line, y_peaks = final_results[6]
    x_lines, x_peaks = final_results[7]
    final_peaks = final_results[8]
    final_spot_pos = final_results[9]
    spot_occ_l = final_results[10]
    mod_ord_coor = rot_trans_loc(loc_list=final_ord_coor, trans_coor=trans_coor, SPR=SPR, origin=rot_cent, angle=rot_angle)
    mod_ord_coor = [[x[0]-y_min, x[1]-x_min] for x in mod_ord_coor]
    mod_loc_list = rot_trans_loc(loc_list=loc_array, trans_coor=trans_coor, SPR=SPR, origin=rot_cent, angle=rot_angle)

    #Sort localizations into sites
    loc_coord_sorted = []
    loc_frame_sorted = []
    for j in range(len(final_spot_pos) + 1):
        loc_coord_sorted.append([])
        loc_frame_sorted.append([])

    for j in range(len(mod_loc_list)):
        pos_index = 0
        loc = mod_loc_list[j]
        x, y = loc
        frame = loc_frames[j]
        for h in range(len(final_spot_pos)):


            if final_spot_pos[h][0][0] + x_min < y < final_spot_pos[h][1][0] + x_min and final_spot_pos[h][0][1] + y_min < x < final_spot_pos[h][1][1] + y_min and spot_occ_l[h] == 1:
                loc_coord_sorted[h + 1].append([x / SPR, y / SPR])
                loc_frame_sorted[h + 1].append(frame)
                pos_index = pos_index + h + 1
            else:
                loc_coord_sorted[0].append([x / SPR, y / SPR])
                loc_frame_sorted[0].append(frame)

    prot_pos_coor_l = []
    for point in final_peaks:
        if len(point) != 0:
            x = point[0]
            y = point[1]
            prot_pos_coor_l.append([float(x) / float(SPR), float(y) / float(SPR)])
        else:
            prot_pos_coor_l.append([])

    crop_coor = [y_min / SPR, y_max / SPR, x_min / SPR, x_max / SPR]

    if test == 'y':

        from matplotlib import gridspec
        from matplotlib.patches import Rectangle

        fig = plt.figure(dpi=300, tight_layout=True)
        fig.set_size_inches(20, 9, forward=True)

        gs = fig.add_gridspec(2, 28, width_ratios=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], height_ratios=[1,1])


        ax0 = plt.subplot(gs[0, 0:6])
        ax0.imshow(cv2.cvtColor(img_clr, cv2.COLOR_BGR2RGB))
        ax0.axis("off")

        ax2 = plt.subplot(gs[0, 7:13])
        ax2.imshow(cv2.cvtColor(shifted_final, cv2.COLOR_BGR2RGB))
        ax2.axis("off")

        ax4 = plt.subplot(gs[0, 14:20])
        ax4.imshow(cv2.cvtColor(rotated_final, cv2.COLOR_BGR2RGB))
        ax4.axis("off")

        ax5 = plt.subplot(gs[0, 23:25])
        ax5.imshow(cv2.cvtColor(crop_final, cv2.COLOR_BGR2RGB))
        for point in mod_ord_coor:
            x = point[1]
            y = point[0]
            ax5.plot(x, y, 'x', color='cyan')
        ax5.axis("off")

        ax6 = plt.subplot(gs[1,8:10])
        ax6.imshow(cv2.cvtColor(crop_w_spots_final, cv2.COLOR_BGR2RGB))
        plt.axvline(int(p1_final[0] - spot_r) + 1, color='lightgrey', linestyle='dashed')
        plt.axvline(int(p1_final[0] + spot_r) - 1, color='lightgrey', linestyle='dashed')
        plt.arrow(int(p1_final[0] - spot_r) + 1, 2, int(p1_final[0] + spot_r) - 1 - (int(p1_final[0] - spot_r) + 1), 0,
                  color='white', width=0.5, length_includes_head=True)
        ax6.axis("off")

        #Check intensity in spot positions
        site_names = ['Pos 1', 'Pos 2', 'Pos 3', 'Pos 4']
        site_colors = ['cyan', 'limegreen', 'gold', 'magenta']
        y_peaks_l = [x[0] for x in y_peaks if len(x) != 0]
        ax7 = plt.subplot(gs[1, 10:12])
        for i in range(len(spot_pos)):
            pos = spot_pos[i]
            ax7.axhspan(ymax=pos[0][1], ymin=pos[1][1], xmin=0, xmax=max(y_line), color=site_colors[i],label=site_names[i], alpha=0.2)
        ax7.plot(y_line, range(len(y_line)), color = 'lightgrey')
        ax7.plot([y_line[j] for j in y_peaks_l], y_peaks_l, 'x', color = 'r')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax7.get_xticklabels(), visible=False)
        plt.setp(ax7.get_yticklabels(), visible=False)
        ax7.tick_params(axis='both', which='both', length=0)
        ax7.invert_yaxis()

        ax8 = plt.subplot(gs[1, 15:17])
        ax8.imshow(cv2.cvtColor(crop_w_spots_final, cv2.COLOR_BGR2RGB))
        for y_peak in y_peaks:
            if len(y_peak) != 0:
                y= y_peak[0]
                plt.axhline(int(y), color='lightgrey', linestyle='dashed')
        ax8.axis("off")

        gs9 = gs[1,17:19].subgridspec(4, 1)
        first_counter = 0
        for i in range(site_n):
            x_line = x_lines[i]
            if len(x_line) != 0:
                if first_counter == 0:
                    ax9_m = fig.add_subplot(gs9[i])


                    ax9_m.plot(range(len(x_line)), x_line, color='lightgrey', label=site_names[i])
                    if len(x_peaks[i]) != 0:
                        ax9_m.plot(x_peaks[i], x_line[x_peaks[i]], 'x', color='r')
                    ax9_m.legend(bbox_to_anchor=(1.05, 1), loc='center left')
                    plt.setp(ax9_m.get_xticklabels(), visible=False)
                    plt.setp(ax9_m.get_yticklabels(), visible=False)
                    ax9_m.tick_params(axis='both', which='both', length=0)
                    first_counter = 1

                else:
                    ax9 = fig.add_subplot(gs9[i], sharex=ax9_m)

                    if len(x_line) != 0:
                        ax9.plot(range(len(x_line)), x_line, color='lightgrey', label=site_names[i])
                        if len(x_peaks[i]) != 0:
                            ax9.plot(x_peaks[i], x_line[x_peaks[i]], 'x', color='r')
                        ax9.legend(bbox_to_anchor=(1.05, 1), loc='center left')
                    plt.setp(ax9.get_xticklabels(), visible=False)
                    plt.setp(ax9.get_yticklabels(), visible=False)
                    ax9.tick_params(axis='both', which='both', length=0)


        ax10 = plt.subplot(gs[1, 23:25])
        ax10.imshow(cv2.cvtColor(crop_final, cv2.COLOR_BGR2RGB))
        prot_pos_coor_l = []
        for point in final_peaks:
            if len(point) != 0:
                x = point[0]
                y = point[1]
                ax10.plot(x, y, 'x', color='green')
                prot_pos_coor_l.append([float(x)/float(SPR), float(y)/float(SPR)])
            else:
                prot_pos_coor_l.append([])
        ax10.axis("off")

        ax11 = plt.subplot(gs[1, 26:28])
        site_names = ['Noise', 'Pos 1', 'Pos 2', 'Pos 3', 'Pos 4']
        site_colors = ['lightgrey','cyan', 'limegreen', 'gold', 'magenta']


        plot_spot_pos = []
        for j in range(len(spot_occ_l)):
            spot = spot_occ_l[j]
            if spot == 1:
                plot_spot_pos.append(final_spot_pos[j])


        for j in range(len(loc_coord_sorted)):
            color = site_colors[j]
            name = site_names[j]
            loc_pos = loc_coord_sorted[j]
            x_l = [item[0]*SPR for item in loc_pos]
            y_l = [item[1]*SPR for item in loc_pos]
            ax11.scatter(y_l, x_l, color=color, s=5, label=name, )

        ax11.set_xlim([x_min, x_max])
        ax11.set_ylim([y_min, y_max])
        for j in range(len(plot_spot_pos)):
            w = plot_spot_pos[j][1][0] - plot_spot_pos[j][0][0]
            h = plot_spot_pos[j][1][1] - plot_spot_pos[j][0][1]
            x_rec = plot_spot_pos[j][0][0] + x_min
            y_rec = plot_spot_pos[j][0][1] + y_min
            ax11.add_patch(Rectangle((x_rec, y_rec),
                                    w, h,
                                    fc='none',
                                    color='red',
                                    linewidth=2,
                                    linestyle="dotted"))
        ax11.xaxis.set_ticks([])
        ax11.yaxis.set_ticks([])
        ax11.invert_yaxis()
        ax11.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        #plt.savefig(processing_dir + '/Protein_counting_plot-' + str(str_index) + '.jpg')
        plt.savefig(processing_dir + '/Protein_counting_plot-' + str(str_index) + '.eps')

        plt.close()

    return crop_final, crop_coor, spot_occ_l, loc_coord_sorted, loc_frame_sorted, prot_pos_coor_l


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
for locs_file in glob.glob("*pick_undrift-data.h5"):

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
    ref_folder_path = str(input("Path to folder with exported reference site distance and size values: "))

    # Import reference values
    site_d_nm_mean, site_d_nm_std, site_spread_nm_mean, site_spread_nm_std = import_ref_data(folder_path=ref_folder_path , file_type='site_ref')

    ref_folder_path = str(input("Path to folder with exported reference linearity values: "))

    # Import reference values
    lin_score_thr = import_ref_data(folder_path=ref_folder_path, file_type='lin_ref')

    site_n_limit = int(input("Designed number of sites: (int)"))
    site_d_pix = site_d_nm_mean / pix_nm_SPR
    site_d_max_pix = (site_d_nm_mean + abs(site_d_nm_std)) / pix_nm_SPR
    tot_d_thr_pix = site_d_max_pix * site_n_limit
    d_thr_site_x = float(input("Distance threshold for maxima clustering (x * site distance): (float)"))
    d_thr_site_nm = site_d_nm_mean * d_thr_site_x
    d_thr_str_x = float(input("Distance threshold for peak clustering (x * site distance): (float)"))
    d_thr_str_nm = site_d_nm_mean * d_thr_str_x
    w_fact = float(input("Window size for cropped images (multiple of total distance and point spread): (float)"))
    crop_w_w, crop_w_h = (site_d_max_pix * 4 * w_fact)/3, site_d_max_pix * 4 * w_fact
    pix_gap = site_d_max_pix * 4 * ((w_fact-1.0)/2) + (site_d_pix/2)
    peak_prom_perc = float(input("Minimum prominence of peaks for peak detection (fraction of max ROI int) : (float)"))
    min_peak_prom = int_scale * peak_prom_perc
    sweep_w_perc = float(input("Percentage of site width to use for peak search : (float)"))

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
         "Linearity threshold of sites (score):\n", str(lin_score_thr) + "\n",
         "Minimum prominence value used for peak detection (counts/% of max):\n", str(min_peak_prom) + " / " + str(peak_prom_perc) + "\n",
         "Peak search sweep width (% of site width):\n",str(sweep_w_perc) + "\n",
         "Distance threshold for maxima clustering (x * site distance):\n", str(d_thr_site_x) + "\n",
         "Distance threshold for peak clustering (x * site distance):\n", str(d_thr_str_x) + "\n",
         "Croping size factor (multiple of total distance):\n",str(w_fact) + "\n",
         "Croping dimensions (width/height, y offset) in pixels:\n",
         str(crop_w_w) + ", " + str(crop_w_h) + ", " + str(pix_gap) + "\n"]

    file1.writelines(L)
    file1.close()  # to change file access modes

    t = str(input("Test run?: (y/n)"))
    print('Finding number of stored structures')

    #
    indeces =  []
    for index in set(locs_df['Str_index'].values):
        indeces.append(index)

    str_index_l = []
    str_pos_l = []
    str_loc_x_l = []
    str_loc_y_l = []
    str_loc_sorted_l = []
    str_loc_frame_sorted_l = []
    str_prot_n_l = []
    str_prot_pos_l = []
    str_prot_coor_l = []
    img_clr_l = []
    crop_l = []
    str_crop_coor_l = []
    str_lin_score_l = []
    cy5_roi_index_l = []

    #
    empty_occ_l = []

    for i in range(site_n_limit):
        empty_occ_l.append(0)


    print('Exporting images of structures')
    from tqdm import tqdm
    with tqdm(total=len(indeces)) as pbar:
        for i in range(len(indeces)):
            pbar.update(1)

            ind = indeces[i]

            # if ind != 78:
            #     continue

            df_ind = locs_df.loc[(ind == locs_df['Str_index'])]
            ROI_origami_det = df_ind['Origami_det'].values.tolist()[0]
            Str_ROI_coor = df_ind['Str_ROI_coor'].values.tolist()[0]
            Str_cent_coor = [np.mean([Str_ROI_coor[0],Str_ROI_coor[1]]),np.mean([Str_ROI_coor[2],Str_ROI_coor[3]])]
            Cy5_ROI_index = df_ind['cy5_roi_index'].values.tolist()[0]

            if ROI_origami_det == 0:
                x_coor = df_ind['loc_coor_x'].values.tolist()[0]
                y_coor = df_ind['loc_coor_y'].values.tolist()[0]

                str_locs = []
                for j in range(len(x_coor)):
                    str_locs.append([x_coor[j], y_coor[j]])

                img_gs, im_color, img_color_file, processing_dir, loc_array_n = loc_to_img_crop(directory=directory, loc_array=str_locs, window_size=window_size, SPR=SPR, norm_max=norm_max, int_scale=int_scale, gauss_size=gauss_size, index=ind)
                crop_img, crop_coor = crop_image(img=im_color, crop_w=crop_w_w, crop_h=crop_w_h)

                str_index_l.append(ind)
                str_pos_l.append(Str_cent_coor)
                str_loc_x_l.append('NaN')
                str_loc_y_l.append('NaN')
                str_prot_n_l.append(0)
                str_prot_pos_l.append(empty_occ_l)
                str_prot_coor_l.append('NaN')
                img_clr_l.append(im_color)
                crop_l.append(crop_img)
                str_crop_coor_l.append(crop_coor)
                str_loc_sorted_l.append('NaN')
                str_loc_frame_sorted_l.append('NaN')
                str_lin_score_l.append('NaN')
                cy5_roi_index_l.append(Cy5_ROI_index)

            else:
                loc_l = []
                x_coor = df_ind['loc_coor_x'].values.tolist()[0]
                y_coor = df_ind['loc_coor_y'].values.tolist()[0]
                coor_frames = df_ind['loc_frame'].values.tolist()[0]

                str_locs = []
                for j in range(len(x_coor)):
                    str_locs.append([x_coor[j], y_coor[j]])

                img_gs, im_color, img_color_file, processing_dir, loc_array_n = loc_to_img_crop(directory=directory, loc_array=str_locs, window_size=window_size, SPR=SPR, norm_max=norm_max, int_scale=int_scale, gauss_size=gauss_size, index=ind)

                new_peak_list, lin_score, dist_tot = prot_det(SPR=SPR, test = t, processing_dir=processing_dir, img_gs=img_gs, img_color_file=img_color_file, int_thr=int_thr, d_thr_site_nm=d_thr_site_nm, d_thr_str_nm=d_thr_str_nm, index=ind, site_d =site_d_nm_mean, site_n_limit=site_n_limit, pix_nm_SPR = pix_nm_SPR)

                if new_peak_list == 'NaN':
                    continue

                if 0 < len(new_peak_list) <= site_n_limit and lin_score < lin_score_thr and dist_tot < d_thr_str_nm:

                    crop, crop_coor, spot_occ_l, loc_coord_sorted, loc_frame_sorted, spot_coor_l = align_and_quant(img_clr=im_color, img_gs=img_gs,
                                                                         ord_coor=new_peak_list, loc_array=loc_array_n, loc_frames=coor_frames,
                                                                         SPR=SPR, pix_gap=pix_gap, crop_w_w=crop_w_w,
                                                                         crop_w_h=crop_w_h,
                                                                         spot_d=site_d_pix, sweep_w_perc=sweep_w_perc, site_n=site_n_limit, min_peak_prom=min_peak_prom, test=t,
                                                                         str_index=ind)

                    str_index_l.append(ind)
                    str_pos_l.append(Str_cent_coor)
                    str_loc_x_l.append(x_coor)
                    str_loc_y_l.append(y_coor)
                    str_prot_n_l.append(sum(spot_occ_l))
                    str_prot_pos_l.append(spot_occ_l)
                    str_prot_coor_l.append(spot_coor_l)
                    img_clr_l.append(im_color)
                    crop_l.append(crop)
                    str_crop_coor_l.append(crop_coor)
                    str_loc_sorted_l.append(loc_coord_sorted)
                    str_loc_frame_sorted_l.append(loc_frame_sorted)
                    str_lin_score_l.append(lin_score)
                    cy5_roi_index_l.append(Cy5_ROI_index)


    df1 = pd.DataFrame(list(zip(str_index_l,str_pos_l, str_loc_x_l, str_loc_y_l, str_loc_sorted_l, str_loc_frame_sorted_l, str_prot_n_l, str_prot_pos_l, str_prot_coor_l, img_clr_l,
                    crop_l, str_crop_coor_l, str_lin_score_l, cy5_roi_index_l)), columns=['Str_index','Str_pos', 'Str_loc_coor_x', 'Str_loc_coor_y', 'Str_loc_coor_sorted', 'Str_loc_frame_sorted',
                                       'Str_protein_n', 'Str_protein_pos','Str_protein_coor', 'Str_clr_img', 'Str_crop', 'Str_crop_coor', 'Str_lin_score', 'cy5_roi_index'])

    df1.to_hdf(locs_file[:-3] + '-protein-q.h5', key='str_locs', mode='w')

