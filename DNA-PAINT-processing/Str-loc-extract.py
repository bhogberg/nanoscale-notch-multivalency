def overlap_check ( x1,y1,r1,x2,y2,r2):
    import math
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    overlap = 1.0 - (d / (r1 + r2))
    dx = x1 - x2
    dy = y1 -  y2
    return overlap, [dx, dy]

def reject_outliers(data,coupled_list):
    import numpy as np
    zip(data, coupled_list)
    m = 0.9 * np.median(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return coupled_list[s < m]

def ROI_overlap_merge(dim_x,dim_y,ROI_cnt_coor, file_name, data_type):
    #import packages used
    import cv2
    import numpy as np

    #Make list to store merged
    img = np.full((dim_x, dim_y), 0, dtype=np.uint8)

    #loop through contour coordinates to draw contours as mask
    for cnt in ROI_cnt_coor:
        (x1, y1), r1 = cv2.minEnclosingCircle(cnt)
        cv2.circle(img, (int(x1), int(y1)), int(r1), (255, 255, 255), -1)

    #
    cv2.imwrite(file_name[:-4] + '-' + data_type + '-ROI-contour-mask.jpg', img)

    #Detect final contours in mask
    contour_points_merge = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    #Remove contours extending the edge of the image
    contour_points_merge_filt = []
    for cnt in contour_points_merge:
        (y, x), r = cv2.minEnclosingCircle(cnt)
        if x > r and y > r:
            contour_points_merge_filt.append(cnt)

    return contour_points_merge_filt

def Cy5_loc_det(file_name):
    #import packages used
    import cv2
    import numpy as np
    from skimage.feature import peak_local_max
    from skimage.morphology import watershed
    from scipy import ndimage
    import imutils
    from matplotlib.image import imread

    #Read image
    im = cv2.imread(file_name, -1)

    im_dim = len(im)

    # Normalize intensity of image
    im = 255.0 * (im / np.amax(im)) * 3.0
    im[im < 0] = 0
    im[im > 255] = 255
    cv2.imwrite(file_name[:-4] + '-Render-Norm.jpg', im)


    # Invert image
    im = cv2.imread(file_name[:-4] + '-Render-Norm.jpg', 0)
    imagem = cv2.bitwise_not(im)
    cv2.imwrite(file_name[:-4] + '-Render-GF-inv.jpg', imagem)


    #Threshold image with loose adaptive thresholding to detect low int spots
    thr = cv2.adaptiveThreshold(imagem, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)#11,7
    thr_inv = cv2.bitwise_not(thr)
    cv2.imwrite(file_name[:-4] + '-Render-GF-thr.jpg', thr_inv)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thr_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(file_name[:-4] + '-Render-GF-thr-fg.jpg', opening)

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(opening)
    localMax = peak_local_max(D, indices=False, min_distance=3,
                              labels=opening)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=opening)

    # loop over the unique labels returned by the Watershed
    # algorithm
    cy5_contour_points = []

    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(opening.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        cy5_contour_points.append(c)

    #Load image for extracting ROI properties
    image_array = imread(file_name)

    # Create lists to store contour properties
    cnt_area_l = []
    cnt_mean_int_l = []
    cnt_max_int_l = []
    cnt_r_l = []
    cnt_index = []

    #Plot and export Cy5-ROI data

    img = cv2.imread(file_name[:-4] + '-Render-Norm.jpg')

    r_l = []
    cy5_roi_index = 0
    for cnt in cy5_contour_points:
        (x, y), r = cv2.minEnclosingCircle(cnt)

        # Filter out structures on the edge of the image
        if x > r and y > r:
            cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 1)
            r_l.append(r)
            x1 = int(x - r)
            x2 = int(x + r)
            y1 = int(y - r)
            y2 = int(y + r)
            cnt_area = cv2.contourArea(cnt)
            #
            crop = image_array[x1:x2, y1:y2]
            mean_int = np.mean(crop)
            max_int = np.max(crop)

            cnt_area_l.append(cnt_area)
            cnt_mean_int_l.append(mean_int)
            cnt_max_int_l.append(max_int)
            cnt_r_l.append(r)
            cnt_index.append(cy5_roi_index)
            cy5_roi_index = cy5_roi_index + 1

    cv2.imwrite(file_name[:-4] + '-det-contour.jpg', img)

    Cy5_ROI_df = pd.DataFrame(list(
        zip(cnt_index, cy5_contour_points, cnt_area_l, cnt_mean_int_l, cnt_max_int_l)),
        columns=['Cy5_ROI_index', 'Cy5_ROI_contour', 'Cy5_ROI_area', 'Cy5_ROI_int_mean', 'Cy5_ROI_int_max'])

    return Cy5_ROI_df

def Cy5_PAINT_pick(Cy5_ROI_df, PAINT_contour_points, file_name, overlap_thr, img_gs_name):
    # import packages used
    import cv2
    import numpy as np
    import copy

    # Read image
    im = cv2.imread(file_name, -1)
    im_dim = len(im)
    dim_x = im_dim * SPR_vis
    dim_y = dim_x

    #
    img_clr = cv2.imread(img_gs_name)

    #Find PAINT contours that are within the Cy5 contours


    #
    Cy5_contour_points = Cy5_ROI_df['Cy5_ROI_contour'].values.tolist()
    Cy5_contour_index = Cy5_ROI_df['Cy5_ROI_index'].values.tolist()

    offset_l = []
    #Loop through contours detected in the PAINT data
    for cnt_PAINT in PAINT_contour_points:

        #Get centroid coordinates
        (x1, y1), r1 = cv2.minEnclosingCircle(cnt_PAINT)

        #Filter out structures on the edge of the image
        if x1 > r1 and y1 > r1:

            #Loop through the contours detected in the Cy5 channel
            for i in range(len(Cy5_contour_points)):
                cnt_Cy5 = Cy5_contour_points[i]
                # Get centroid coordinates
                (y2, x2), r2 = cv2.minEnclosingCircle(cnt_Cy5)

                #Check if PAINT and Cy5 contour overlap
                overlap, v = overlap_check(x1, y1, r1, x2*SPR_vis, y2*SPR_vis, r2*SPR_vis)
                if overlap > overlap_thr:
                    offset_l.append(v)
    #
    offset_v = [np.mean([x[0] for x in offset_l]), np.mean([x[1] for x in offset_l])]

    PAINT_contour_points_filt = []

    # Loop through contours detected in the PAINT data
    for cnt_PAINT in PAINT_contour_points:

        # Get centroid coordinates
        (x1, y1), r1 = cv2.minEnclosingCircle(cnt_PAINT)

        # Filter out structures on the edge of the image
        if x1 > r1 and y1 > r1:

            # Loop through the contours detected in the Cy5 channel
            for i in range(len(Cy5_contour_points)):
                cnt_Cy5 = Cy5_contour_points[i]
                # Get centroid coordinates
                (y2, x2), r2 = cv2.minEnclosingCircle(cnt_Cy5)

                # Check if PAINT and Cy5 contour overlap
                overlap, v = overlap_check(x1, y1, r1, (x2 * SPR_vis)+offset_v[0], (y2 * SPR_vis)+offset_v[1], r2 * SPR_vis)
                if overlap > overlap_thr:
                    offset_l.append(v)
                    PAINT_contour_points_filt.append(cnt_PAINT)


    #Visualize kept PAINT contours
    # Filtering out aggregates from detected contours
    r_l_PAINT = []
    for cnt in PAINT_contour_points_filt:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        r_l_PAINT.append(r)

    # Visualize contour radii distribution
    # Calculate ROI r mean and std
    r_mean_PAINT = np.mean(r_l_PAINT)
    r_std_PAINT = np.std(r_l_PAINT)
    win_size_PAINT = r_mean_PAINT + (2.5 * r_std_PAINT)

    #Merge overlapping PAINT ROIs
    PAINT_contour_points_merge = ROI_overlap_merge(dim_x=dim_x,dim_y=dim_y,ROI_cnt_coor=PAINT_contour_points_filt, file_name=file_name, data_type='PAINT-w-str-')

    #

    Cy5_contour_origami_n = [0 for x in Cy5_contour_points]
    PAINT_Cy5_ROI_index = []
    #
    # Loop through contours detected in the PAINT data
    PAINT_contour_points_merge_new = []
    for cnt_PAINT in PAINT_contour_points_merge:
        # Get centroid coordinates
        (x1, y1), r1 = cv2.minEnclosingCircle(cnt_PAINT)

        # Loop through the contours detected in the Cy5 channel
        for i in range(len(Cy5_contour_points)):
            cnt_Cy5 = Cy5_contour_points[i]
            # Get centroid coordinates
            (y2, x2), r2 = cv2.minEnclosingCircle(cnt_Cy5)

            # Check if PAINT and Cy5 contour overlap
            overlap, v = overlap_check(x1, y1, r1, (x2 * SPR_vis)+offset_v[0], (y2 * SPR_vis)+offset_v[1], r2 * SPR_vis)
            if overlap > overlap_thr:
                Cy5_contour_origami_n[i] = Cy5_contour_origami_n[i] + 1
                PAINT_Cy5_ROI_index.append(Cy5_contour_index[i])
                PAINT_contour_points_merge_new.append(cnt_PAINT)

    Cy5_ROI_df['Origami_N'] = Cy5_contour_origami_n
    #Visualize


    for i in range(len( Cy5_contour_points)):

        cnt_Cy5 = Cy5_contour_points[i]
        index = Cy5_contour_index[i]
        (y,x), r = cv2.minEnclosingCircle(cnt_Cy5)
        cv2.circle(img_clr, (int((x*SPR_vis)+offset_v[0]), int((y*SPR_vis)+offset_v[1])), int(r*SPR_vis), (255, 0, 0), 3)
        cv2.putText(img_clr, str(index), (int(x*SPR_vis + r*SPR_vis+offset_v[0]), int(y*SPR_vis + r*SPR_vis+offset_v[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(255, 0, 0))



    #Draw Cy5 ROI with origami ROI
    Cy5_ROI_w_origami_df = Cy5_ROI_df.loc[(0 < Cy5_ROI_df['Origami_N'])]
    Cy5_ROI_w_origami_cnt = Cy5_ROI_w_origami_df['Cy5_ROI_contour'].values.tolist()
    Cy5_ROI_w_origami_index = Cy5_ROI_w_origami_df['Cy5_ROI_index'].values.tolist()
    Cy5_ROI_w_origami_n = Cy5_ROI_w_origami_df['Origami_N'].values.tolist()

    for i in range(len(Cy5_ROI_w_origami_cnt)):
        n = Cy5_ROI_w_origami_n[i]
        cnt_Cy5 = Cy5_ROI_w_origami_cnt[i]
        Cy5_ROI_ind = Cy5_ROI_w_origami_index[i]
        (y,x), r = cv2.minEnclosingCircle(cnt_Cy5)

        cv2.circle(img_clr, (int((x*SPR_vis)+offset_v[0]), int((y*SPR_vis)+offset_v[1])), int(r*SPR_vis), (0, 255, 0), 3)
        cv2.putText(img_clr, str(Cy5_ROI_ind), (int(x*SPR_vis + r*SPR_vis+offset_v[0]), int(y*SPR_vis + r*SPR_vis+offset_v[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0, 255, 0))
        cv2.putText(img_clr, 'N: ' +str(n), (int(x*SPR_vis + r*SPR_vis+offset_v[0]), int(y*SPR_vis + r*SPR_vis+offset_v[1]+25)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0, 255, 0))

    #Draw Cy5 ROI without origami ROI
    Cy5_ROI_wo_origami_df = Cy5_ROI_df.loc[(0 == Cy5_ROI_df['Origami_N'])]
    Cy5_ROI_wo_origami_cnt = Cy5_ROI_wo_origami_df['Cy5_ROI_contour'].values.tolist()
    Cy5_ROI_wo_origami_index = Cy5_ROI_wo_origami_df['Cy5_ROI_index'].values.tolist()
    Cy5_ROI_wo_origami_n = Cy5_ROI_wo_origami_df['Origami_N'].values.tolist()

    for i in range(len(Cy5_ROI_wo_origami_cnt)):
        n = Cy5_ROI_wo_origami_n[i]
        Cy5_ROI_cnt = Cy5_ROI_wo_origami_cnt[i]
        Cy5_ROI_ind = Cy5_ROI_wo_origami_index[i]
        (y, x), r = cv2.minEnclosingCircle(Cy5_ROI_cnt)
        cv2.circle(img_clr, (int((x*SPR_vis)+offset_v[0]), int((y*SPR_vis)+offset_v[1])), int(r*SPR_vis), (0, 0, 255), 3)
        cv2.putText(img_clr, str(Cy5_ROI_ind), (int(x*SPR_vis + r*SPR_vis+offset_v[0]), int(y*SPR_vis+ r*SPR_vis+offset_v[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255))
        cv2.putText(img_clr, 'N: ' +str(n), (int(x*SPR_vis + r*SPR_vis+offset_v[0]), int(y*SPR_vis + r*SPR_vis+offset_v[1]+25)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0, 0, 255))


    # Draw origami ROI
    PAINT_origami_det = []
    PAINT_ROI_coor = []
    PAINT_str_index= []
    PAINT_str_ind = 0

    for cnt in PAINT_contour_points_merge_new:
        PAINT_str_index.append(PAINT_str_ind)


        (x, y), r = cv2.minEnclosingCircle(cnt)


        x = x/SPR_vis
        y = y/SPR_vis
        ws = win_size_PAINT/SPR_vis
        x1 = x - ws
        x2 = x + ws
        y1 = y - ws
        y2 = y + ws
        cv2.rectangle(img_clr, (int(x1*SPR_vis), int(y1*SPR_vis)), (int(x2*SPR_vis), int(y2*SPR_vis)), (0, 255, 255), 2)
        cv2.putText(img_clr, str(PAINT_str_ind), (int(x2*SPR_vis+5), int(y2*SPR_vis+5)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 255))
        PAINT_origami_det.append(1)
        PAINT_ROI_coor.append([x1, x2, y1, y2])
        PAINT_str_ind = PAINT_str_ind + 1

    for i in range(len(Cy5_ROI_wo_origami_cnt)):
        PAINT_str_index.append(PAINT_str_ind)

        index = Cy5_ROI_wo_origami_index[i]
        cnt = Cy5_ROI_wo_origami_cnt[i]
        (y, x), r = cv2.minEnclosingCircle(cnt)
        x = x + offset_v[0]/SPR_vis
        y = y + offset_v[1]/SPR_vis
        ws = win_size_PAINT/SPR_vis
        x1 = x - ws
        x2 = x + ws
        y1 = y - ws
        y2 = y + ws
        cv2.rectangle(img_clr, (int(x1*SPR_vis), int(y1*SPR_vis)), (int(x2*SPR_vis), int(y2*SPR_vis)), (255, 0, 255), 2)
        cv2.putText(img_clr, str(PAINT_str_ind), (int(x2*SPR_vis+5), int(y2*SPR_vis+5)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 0, 255))
        PAINT_origami_det.append(0)
        PAINT_Cy5_ROI_index.append(index)
        PAINT_ROI_coor.append([x1, x2, y1, y2])
        PAINT_str_ind = PAINT_str_ind + 1

    cv2.imwrite(file_name[:-4] + '-Render-GS-PAINT-det.png', img_clr)

    return PAINT_ROI_coor, PAINT_Cy5_ROI_index, PAINT_origami_det, Cy5_ROI_df

def gauss(x, mu, sigma, A):
    import numpy as np
    return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)


# Function for generating colors for plotting and drawing on images
def color_picker(colormap, channels):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    cmap = plt.cm.get_cmap(name=colormap, lut=channels)
    colors_rgba = []
    colors_bgr = []
    for i in range(channels):
        colors_rgba.append(cmap(i))

    for i in colors_rgba:
        color_bgr = []
        for j in reversed(mpl.colors.to_rgb(i)):
            color_bgr.append(int(j * 255))
        colors_bgr.append(tuple(color_bgr))

    return colors_rgba, colors_bgr


# Functions for exporting cropped images
def loc_to_img_for_det(file_name, locs_df, SPR_det, thr_low, thr_high, gauss_size):
    # Import packages
    import numpy as np
    from scipy.ndimage import gaussian_filter

    locs_x = locs_df['x'].values.tolist()
    locs_y = locs_df['y'].values.tolist()

    locs_x = [int(i * SPR_det) for i in locs_x]
    locs_y = [int(i * SPR_det) for i in locs_y]

    # Make empty image for detection
    dim_x, dim_y = max(locs_x), max(locs_y)
    img = np.zeros((dim_x, dim_y))

    # Fill up image with localizations
    for x, y in zip(locs_x, locs_y):
        img[x - 1][y - 1] = img[x - 1][y - 1] + 1.0

    # Normalize intensity
    img = 255.0 * ((img - thr_low) / (thr_high - thr_low))
    img[img < 0.0] = 0.0

    # Apply Gaussian blur
    img_blur = gaussian_filter(img, sigma=gauss_size)
    cv2.imwrite(file_name[:-5] + '-Render-GS.png', img_blur)
    cv2.destroyAllWindows()

    gs_img_name = file_name[:-5] + '-Render-GS.png'

    return gs_img_name, img_blur

# Image filtering and structure detection
def str_filter_det(img_gs, dil_it_n, file_name):
    # import packages
    import cv2

    # Make list to store contours areas
    contour_area_l = []

    # Read grayscale image
    image = cv2.imread(img_gs, 0)

    # Binary thresholding after gaussian filtering
    ret3, th3 = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = th3
    cv2.imwrite(file_name[:-5] + '-Render-GF-Otsu.jpg', th3)

    # Dilation to increase
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilation = cv2.dilate(th3, kernel, iterations=dil_it_n)
    cv2.imwrite(file_name[:-5] + '-Render-GF-Otsu-DIL.jpg', dilation)

    # Detection of contours in dilated image
    contour_points = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    image = cv2.imread(img_gs)

    for cnt in contour_points:
        contour_area_l.append(cv2.contourArea(cnt))
        # Define ROI coordinates
        (y1, x1), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(image, (int(y1), int(x1)), int(radius), (0, 0, 255), 1)
    cv2.imwrite(file_name[:-5] + '-all-det-contour.jpg', image)

    return contour_points, bin_img


# Function for structure detection by size thresholding
def size_thr(str_contours, bin_img, pop_spread, SPR_det, img_gs, file_name):
    # import packages used
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy import stats
    from scipy.signal import argrelmax
    import cv2
    import pandas as pd

    # For checking size distribution in sample
    Area = []
    for cnt in str_contours:
        Area.append(cv2.contourArea(cnt))
    y, x = plt.hist(Area, bins=100, density=True)[:2]  # range=(0, 1500)
    plt.close()

    x = x.tolist()
    density = stats.gaussian_kde(Area)
    y_n = density(x)

    # Finding populations in peak detection in the probability density plot
    Sizes = []
    peaks = argrelmax(y_n)
    for peaklist in peaks:
        for peak_point in peaklist:
            Sizes.append(x[peak_point])

    # Checking visually population to determine size
    img_rgb = cv2.imread(img_gs, 1)
    #img_rgb = cv2.cvtColor(img_gs, cv2.COLOR_GRAY2RGB)
    Colors_plot, Colors = color_picker(colormap='gist_rainbow', channels=len(Sizes))

    for i in range(len(Sizes)):
        if i == 0:
            Min_thr = int(Sizes[i] - (Sizes[i] * pop_spread))
        else:
            Min_thr = int(Sizes[i - 1] + (Sizes[i - 1] * pop_spread))
        Max_thr = int(Sizes[i] + (Sizes[i] * pop_spread))

        for cnt in contour_points:
            if cv2.contourArea(cnt) > Min_thr and cv2.contourArea(cnt) < Max_thr:
                x_cnt, y_cnt, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img_rgb, (x_cnt, y_cnt), (x_cnt + w, y_cnt + h), Colors[i], 2)

    fig = plt.figure(dpi=120, tight_layout=True)
    fig.set_size_inches(10, 4, forward=True)

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], height_ratios=[1])

    ax0 = plt.subplot(gs[0])
    ax0.hist(Area, bins=100, density=True)
    ax0.plot(x, y_n, 'g--')
    for i in range(len(Sizes)):
        if i == 0:
            Min_thr = int(Sizes[i] - (Sizes[i] * pop_spread))
        else:
            Min_thr = int(Sizes[i - 1] + (Sizes[i - 1] * pop_spread))
        Max_thr = int(Sizes[i] + (Sizes[i] * pop_spread))
        ax0.axvspan(Min_thr, Max_thr, color=Colors_plot[i], alpha=0.5, label=str(Min_thr) + ' - ' + str(Max_thr))

    ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Contour Area [pixel]')
    ax0.set_xlabel('Contour area [pixel]')
    ax0.set_ylabel('Normalized frequency')

    ax1 = plt.subplot(gs[1])
    ax1.imshow(img_rgb)
    ax1.axis("off")
    plt.tight_layout()
    plt.show()
    cv2.destroyAllWindows()

    # for determining threshold values based on given population size mean and spread
    Min_thr, Max_thr = [float(x) for x in input("Contour size threshold to use: (float) (min max): ").split()]

    # Checking visually population to determine size
    img_rgb = cv2.imread(img_gs, 1)

    for cnt in contour_points:
        if cv2.contourArea(cnt) > Min_thr and cv2.contourArea(cnt) < Max_thr:
            x_cnt, y_cnt, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_rgb, (x_cnt, y_cnt), (x_cnt + w, y_cnt + h), (0, 255, 0), 2)

    fig = plt.figure(dpi=120, tight_layout=True)
    fig.set_size_inches(10, 4, forward=True)

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], height_ratios=[1])

    ax0 = plt.subplot(gs[0])
    ax0.hist(Area, bins=100, density=True)
    ax0.plot(x, y_n, 'g--')
    ax0.axvspan(Min_thr, Max_thr, color=(0, 1.0, 0), label=str(Min_thr) + ' - ' + str(Max_thr), alpha=0.5)
    ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Selected contour area range [pixel]')

    #for figure
    # ax0.plot(x, y_n, 'g--', label= 'kde')
    # ax0.axvspan(Min_thr, Max_thr, color=(0, 1.0, 0), label='Selected contour\n    area range   ', alpha=0.5)
    # ax0.legend()


    ax0.set_xlabel('Contour area [pixel]')
    ax0.set_ylabel('Normalized frequency')

    ax1 = plt.subplot(gs[1])
    ax1.imshow(img_rgb)
    ax1.axis("off")
    plt.tight_layout()
    plt.show()
    cv2.destroyAllWindows()

    # Filtering contours
    cnt_df = pd.DataFrame(list(
        zip(str_contours, Area)),
        columns=['Cnt_points', 'Cnt_area'])

    cnt_df_filt = cnt_df.loc[(Min_thr < cnt_df['Cnt_area']) & (Max_thr > cnt_df['Cnt_area'])]
    filt_cnt_points = cnt_df_filt['Cnt_points'].values.tolist()

    # Filtering out aggregates from detected contours
    r_l_PAINT = []
    for cnt in filt_cnt_points:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        r_l_PAINT.append(r)

    # Visualize contour radii distribution
    # Calculate ROI r mean and std
    r_mean_PAINT = np.mean(r_l_PAINT)
    r_std_PAINT = np.std(r_l_PAINT)
    win_size_PAINT = r_mean_PAINT + (2.5 * r_std_PAINT)

    PAINT_origami_det = []
    PAINT_ROI_coor = []
    PAINT_str_index = []
    PAINT_str_ind = 0

    image = cv2.imread(img_gs, 1)

    for cnt in filt_cnt_points:
        PAINT_str_index.append(PAINT_str_ind)


        (x, y), r = cv2.minEnclosingCircle(cnt)

        x = x / SPR_det
        y = y / SPR_det
        ws = win_size_PAINT / SPR_det
        x1 = x - ws
        x2 = x + ws
        y1 = y - ws
        y2 = y + ws

        cv2.rectangle(image, (int(x1*SPR_det), int(y1*SPR_det)), (int(x2*SPR_det), int(y2*SPR_det)), (0, 255, 255), 3)
        cv2.putText(image, str(PAINT_str_ind), (int(x2*SPR_det+5), int(y2*SPR_det+5)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 255))
        PAINT_origami_det.append(1)
        PAINT_ROI_coor.append([x1, x2, y1, y2])
        PAINT_str_ind = PAINT_str_ind + 1
    cv2.imwrite(file_name[:-5] + '-origami-ROI.jpg', image)

    return filt_cnt_points,PAINT_ROI_coor, PAINT_str_index, PAINT_origami_det , Min_thr, Max_thr

# Function for collecting localizations falling into a certain ROI (optional export of images of ROI)
def str_ROI_loc(PAINT_ROI, locs_df):

    #Define ROI coordinates
    y_min, y_max, x_min, x_max = PAINT_ROI

    #Find localizations within the ROI
    loc_xy_crop = locs_df.loc[(x_min < locs_df['x']) & (x_max > locs_df['x']) & (y_min < locs_df['y']) & (y_max > locs_df['y'])]

    #
    str_coor = []
    prec_list = []
    x = loc_xy_crop['x'].values.tolist()
    y = loc_xy_crop['y'].values.tolist()
    x_prec = loc_xy_crop['lpx'].values.tolist()
    y_prec = loc_xy_crop['lpy'].values.tolist()

    for i in range(len(x)):
        x[i] = x[i] - x_min
        y[i] = y[i] - y_min
        str_coor.append([x[i], y[i]])
        prec_list.append([x_prec[i], y_prec[i]])

    pc_list = loc_xy_crop['photons'].values.tolist()
    ellipticity_list = loc_xy_crop['ellipticity'].values.tolist()
    frame_list = loc_xy_crop['frame'].values.tolist()
    crop_coor = [x_min, x_max, y_min, y_max]

    return str_coor, prec_list, pc_list, crop_coor, ellipticity_list, frame_list

#Specify directories
import os, glob
Cy5_pick = str(input("Cy5-channel based picking?: (y/n)"))

folder_path1 = str(input("Path to folder with RCC undrifted hdf5 file: "))
directory1 = folder_path1
os.chdir(folder_path1)

if Cy5_pick == 'y':
    folder_path2 = str(input("Path to folder with Cy5 snap image: "))
    directory2 = folder_path2

#import modules used
import pandas as pd
import numpy as np
import cv2

#Input information needed for processing
print('Parameters for processing:')
SPR_vis = int(input("Sub-pixelation rate for ROI visualization: (int)"))
thr_low, thr_high = [float(x) for x in input("Thresholding values for image intensity scaling (min max): ").split()]
gauss_size = float(input("Kernel size for gaussian blurring: (float)"))
dil_it_n = int(input("Number of dilation steps applied to image: (int)"))
if Cy5_pick == 'y':
    overlap_thr = float(input("Fraction overlap between Cy5 channel and PAINT channel for ROI picking: (float)"))

f = open(folder_path1 + '/Origami-ROI-detection-parameters.txt', "w")
f.write("\n")
f.write("Parameters used for localization visualization: \n")
f.write("Sub-pixelation rate : " +  str(SPR_vis) + "\n")
f.write( "Upper threshold for normalization: " + str(thr_low) + "\n")
f.write("Lower threshold for normalization: " + str(thr_high) + "\n")
f.write("Gaussian kernel size for blurring: " + str(gauss_size) + "\n")
f.write("Number of dilation steps applied to image: " + str(dil_it_n) + "\n")
if Cy5_pick == 'y':
    f.write("Fraction of area overlap above which PAINT ROIs are merged: " + str(overlap_thr) + "\n")


#List for storing values to be exported
Str_index = []
loc_coor_x = []
loc_coor_y = []
loc_frame = []
loc_prec_x = []
loc_prec_y = []
photon_count = []
ellipticity = []
Cy5_ROI_index_l = []
Str_ROI_coor = []


#Import localization file
for locs_file in glob.glob("*pick_undrift.hdf5"):
    print ('Imported localization file : "' + str(locs_file) + '"')

#Import loc file and transform it into a np array
    locs_df = pd.read_hdf(locs_file, key='locs')

    #Import Cy5 image and determine str positions

    if Cy5_pick == 'y':
        os.chdir(folder_path2)
        for cy5_file in glob.glob("*.tif"):
            print ('Imported Cy5 image file : "' + str(cy5_file) + '"')

            #
            print('Detecting Cy5 ROIs in image')
            Cy5_ROI_df = Cy5_loc_det(file_name=cy5_file)

            gs_img_name, gs_img = loc_to_img_for_det(file_name=locs_file, locs_df=locs_df, SPR_det=SPR_vis,
                                                     thr_low=thr_low,
                                                     thr_high=thr_high, gauss_size=gauss_size)

            # Detecting contours in reconstructed image
            print('Detecting contours in super-resolved image')
            contour_points, bin_img = str_filter_det(img_gs=gs_img_name, dil_it_n=dil_it_n, file_name=locs_file)

            # Filter noise out from contours
            print('Determining origami ROI coordinates by picking size interval for origami contours')
            PAINT_cnt, PAINT_ROI_coor, PAINT_str_index, PAINT_origami_det, Min_thr, Max_thr = size_thr(str_contours=contour_points,
                                                                                            bin_img=bin_img,
                                                                                            pop_spread=0.6,
                                                                                            SPR_det=SPR_vis,
                                                                                            img_gs=gs_img_name,
                                                                                            file_name=locs_file)
            f = open(folder_path1 + '/Origami-ROI-detection-parameters.txt', 'a')
            f.write("Size threshold used for origami ROI picking (in pixels): " + str(Min_thr) + "-" + str(Max_thr) + "\n")
            f.close()

            #
            print('Finding origami ROIs overlapping with Cy5 ROIs')
            PAINT_ROI_coor, PAINT_Cy5_ROI_index, PAINT_origami_det, Cy5_ROI_df = Cy5_PAINT_pick(Cy5_ROI_df=Cy5_ROI_df, PAINT_contour_points= PAINT_cnt , file_name=cy5_file, overlap_thr=overlap_thr, img_gs_name=gs_img_name)

            Cy5_ROI_df.to_hdf(cy5_file[:-4] + '-ROI-data.h5', key='str_locs', mode='w')

    else:
        # Reconstruct image from localizations
        print('Rendering localizations and exporting super-resolved images')
        gs_img_name, gs_img = loc_to_img_for_det(file_name=locs_file, locs_df=locs_df, SPR_det=SPR_vis, thr_low=thr_low,
                                                 thr_high=thr_high, gauss_size=gauss_size)

        # Detecting contours in reconstructed image
        print('Detecting contours in super-resolved image')
        contour_points, bin_img = str_filter_det(img_gs=gs_img_name, dil_it_n=dil_it_n, file_name=locs_file)

        # Filter noise out from contours
        print('Determining origami ROI coordinates by picking size interval for origami contours')
        PAINT_cnt, PAINT_ROI_coor, PAINT_str_index, PAINT_origami_det , Min_thr, Max_thr = size_thr(str_contours=contour_points, bin_img=bin_img, pop_spread=0.6, SPR_det=SPR_vis, img_gs=gs_img_name, file_name=locs_file)
        PAINT_Cy5_ROI_index = ['NaN' for x in PAINT_ROI_coor]
        f = open(folder_path1 + '/Origami-ROI-detection-parameters.txt', 'a')
        f.write("Size threshold used for origami ROI picking (in pixels): " + str(Min_thr) + "-" + str(Max_thr) + "\n")
        f.close()

    print('Exporting structure localizations')
    from tqdm import tqdm

    with tqdm(total=len(PAINT_ROI_coor)) as pbar:
        row_count = 0
        for i in range(len(PAINT_ROI_coor)):
            origami_det = PAINT_origami_det[i]
            cnt = PAINT_ROI_coor[i]
            cy5_roi_ind = PAINT_Cy5_ROI_index[i]
            pbar.update(1)
            index = PAINT_str_index[i]

            if origami_det == 1:
                str_coor, prec_list, pc_list, crop_coor, ellipticity_list, frame_list = str_ROI_loc(PAINT_ROI=cnt, locs_df=locs_df)

                Str_index.append(index)
                Cy5_ROI_index_l.append(cy5_roi_ind)
                loc_coor_x.append([x[0] for x in str_coor])
                loc_coor_y.append([x[1] for x in str_coor])
                loc_frame.append(frame_list)
                loc_prec_x.append([x[0] for x in prec_list])
                loc_prec_y.append([x[1] for x in prec_list])
                photon_count.append(pc_list)
                ellipticity.append(ellipticity_list)
                Str_ROI_coor.append(cnt)

            else:

                Str_index.append(index)
                Cy5_ROI_index_l.append(cy5_roi_ind)
                loc_coor_x.append(['NaN'])
                loc_coor_y.append(['NaN'])
                loc_frame.append(['NaN'])
                loc_prec_x.append(['NaN'])
                loc_prec_y.append(['NaN'])
                photon_count.append(['NaN'])
                ellipticity.append(['NaN'])
                Str_ROI_coor.append(cnt)

    df = pd.DataFrame(list(zip(Str_index, Str_ROI_coor, loc_coor_x, loc_coor_y, loc_frame, loc_prec_x, loc_prec_y, photon_count, ellipticity, Cy5_ROI_index_l, PAINT_origami_det)),
        columns=['Str_index', 'Str_ROI_coor', 'loc_coor_x', 'loc_coor_y', 'loc_frame', 'loc_prec_x', 'loc_prec_y', 'int', 'ellipticity', 'cy5_roi_index', 'Origami_det'])
    df.to_hdf(locs_file[:-5] + '-data.h5', key='str_locs', mode='w')
