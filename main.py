import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """Load the images and put the result in list
    the first image of the list is the reference image"""
    chambre_imgs = []
    chambre_imgs.append(cv2.imread(".\Images\Chambre\Reference.JPG"))
    for i in range(67, 74):
        chambre_imgs.append(cv2.imread(f".\Images\Chambre\IMG_65{i}.JPG"))

    cuisine_imgs = []
    cuisine_imgs.append(cv2.imread(".\Images\Cuisine\Reference.JPG"))
    for i in range(62, 66):
        img = cv2.imread(f".\Images\Cuisine\IMG_65{i}.JPG")
        cuisine_imgs.append(cv2.imread(f".\Images\Cuisine\IMG_65{i}.JPG"))

    salon_imgs = []
    salon_imgs.append(cv2.imread(".\Images\Salon\Reference.JPG"))
    for i in range(51, 61):
         salon_imgs.append(cv2.imread(f".\Images\Salon\IMG_65{i}.JPG"))

    return chambre_imgs , cuisine_imgs, salon_imgs


def resize_image(imgs):
    resized_imgs = []
    for img in imgs:
        resized_img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        resized_imgs.append(resized_img)

    return resized_imgs


def convert_bgr_to_gray(imgs):
    gray_imgs = []
    for img in imgs:
        gray_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    return gray_imgs


def show_imgs(imgs):
    for i, img in enumerate(imgs):
        cv2.imshow(f"image: {i}", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hist(img):
    plt.hist(img.ravel(), 256, [0,256])
    plt.xlim([0, 256])
    plt.show()


def show_hist_with_matplotlib_gray(hist, title, pos, color):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(2, 2, pos)
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


def draw_bin_boxes(binary_img, img_to_draw):
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the bounding box
    for c in contours:
        rect = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        print(area)
        if area < 1200: # or area > 15000:
            print(f'area to short or to big: {area}')
            continue
        x, y, w, h = rect
        cv2.rectangle(img_to_draw, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return img_to_draw


def remove_shadow_from_img(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result_img = cv2.merge(result_planes)
    result_norm_img = cv2.merge(result_norm_planes)

    return result_img


def detection_algorithm_simple(ref_img, img):
    """Simple detection algorithm"""

    # resize the image
    resized_img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    resized_ref_img = cv2.resize(ref_img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

    img_without_shadow = remove_shadow_from_img(resized_img)
    ref_img_without_shadow = remove_shadow_from_img(resized_ref_img)

    # Convert bgr to gray
    img_gray = cv2.cvtColor(img_without_shadow, cv2.COLOR_BGR2GRAY)
    ref_img_gray = cv2.cvtColor(ref_img_without_shadow, cv2.COLOR_BGR2GRAY)

    # blur img
    blur_img = cv2.GaussianBlur(img_gray, (25, 25), 0)
    blur_ref_img = cv2.GaussianBlur(ref_img_gray, (25, 25), 0)

    diff_img = cv2.absdiff(blur_img, blur_ref_img)

    # Apply a binary threshold
    _, thresh_img = cv2.threshold(diff_img, 0, 255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    dilated_img = cv2.dilate(thresh_img, np.ones((15, 15), np.uint8))

    cv2.imshow("difference image", dilated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return_img = draw_bin_boxes(dilated_img, resized_img)

    return return_img

def detection_algorithm(ref_img, img):
    # resize the image
    resized_img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    resized_ref_img = cv2.resize(ref_img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

    # Convert bgr to gray
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    ref_img_gray = cv2.cvtColor(resized_ref_img, cv2.COLOR_BGR2GRAY)

    blur_img = cv2.medianBlur(img_gray, 25)
    th2 = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)
    dilated_img = cv2.dilate(th2, np.ones((25, 25), np.uint8))

    blur_ref_img = cv2.medianBlur(ref_img_gray, 25)
    th3 = cv2.adaptiveThreshold(blur_ref_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)
    dilated_ref = cv2.dilate(th3, np.ones((25, 25), np.uint8))

    diff_dilate_img = cv2.absdiff(dilated_ref, dilated_img)

    contours, hierarchy = cv2.findContours(diff_dilate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img_gray, contours, -1, (0, 255, 0), 3)

    # Draw the bounding box
    for c in contours:
        rect = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        print(area)
        if area < 1200:
            print(f'area to short: {area}')
            continue
        x, y, w, h = rect
        cv2.rectangle(resized_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return resized_img


def detection_algorithm_otsu(ref_img, img):
    # resize the image
    resized_img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    resized_ref_img = cv2.resize(ref_img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

    # remove the shadow in the image
    rgb_planes = cv2.split(resized_img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result_img = cv2.merge(result_planes)
    result_norm_img = cv2.merge(result_norm_planes)

    # Remove the shadow in the reference image
    rgb_planes = cv2.split(resized_ref_img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result_ref_img = cv2.merge(result_planes)
    result_norm_ref_img = cv2.merge(result_norm_planes)

    # Convert bgr to gray
    img_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    ref_img_gray = cv2.cvtColor(result_ref_img, cv2.COLOR_BGR2GRAY)

    blur_img = cv2.medianBlur(img_gray, 15)

    blur_ref_img = cv2.medianBlur(ref_img_gray, 15)

    diff_blur_img = cv2.absdiff(blur_img, blur_ref_img)

    _, th = cv2.threshold(diff_blur_img,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dilated_img = cv2.dilate(th, np.ones((17, 17), np.uint8))

    contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img_gray, contours, -1, (0, 255, 0), 3)

    # Draw the bounding box
    for c in contours:
        rect = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        print(area)
        if area < 1200 or area > 25000:
            print(f'area to short: {area}')
            continue
        x, y, w, h = rect
        cv2.rectangle(resized_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return resized_img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load the data
    chambre_imgs, cuisine_imgs, salon_imgs  = load_data()

    # Resize the image
    # chambre_imgs = resize_image(chambre_imgs)
    # cuisine_imgs = resize_image(cuisine_imgs)
    # salon_imgs = resize_image(salon_imgs)
    #
    # # Convert to gray
    # chambre_imgs = convert_bgr_to_gray(chambre_imgs)
    # cuisine_imgs = convert_bgr_to_gray(cuisine_imgs)
    # salon_imgs = convert_bgr_to_gray(salon_imgs)



    ref_chambre_img = chambre_imgs[0]

    imgs = []
    # Select an image
    for i in range(1, 7, 2):
        img = chambre_imgs[i]

        img = detection_algorithm_simple(ref_chambre_img, img)

        imgs.append(img)

    ref_cuisine_img = cuisine_imgs[0]
    for i in range(1, 4):
        img = cuisine_imgs[i]

        img = detection_algorithm_simple(ref_cuisine_img, img)

        imgs.append(img)

    ref_salon_img = salon_imgs[0]
    for i in range(1, 11):
        img = salon_imgs[i]

        img = detection_algorithm_simple(ref_salon_img, img)

        imgs.append(img)

    show_imgs(imgs)

'''
    img = chambre_imgs[5]
    img = cv2.medianBlur(img, 25)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    dilated_img = cv2.dilate(th2, np.ones((19, 19), np.uint8))

    ref_img = cv2.medianBlur(ref_img, 25)
    th3 = cv2.adaptiveThreshold(ref_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    dilated_ref = cv2.dilate(th3, np.ones((19, 19), np.uint8))

    diff_img = cv2.subtract(ref_img, img)
    diff_dilate_img = cv2.absdiff(dilated_ref, dilated_img)

    show_imgs([img, th2, dilated_img, dilated_ref, ref_img, th3, diff_img, diff_dilate_img])'''

'''
    # substract the original img
    new_img = cv2.absdiff(chambre_imgs[0], img)

    # Make the mask for the image
    M = np.zeros(new_img.shape[:2], dtype="uint8")
    M[new_img==10] = 255

    # Equalize the histogram
    new_img_eq = cv2.equalizeHist(new_img)

    h_without_mask = cv2.calcHist([new_img], [0], None, [256], [0, 256])
    h_with_mask = cv2.calcHist([new_img], [0], M, [256], [0, 256])
    h_with_mask_eq = cv2.calcHist([new_img_eq], [0], M, [256], [0, 256])

    show_hist_with_matplotlib_gray(h_without_mask, "Without mask", 1, 'm')
    show_hist_with_matplotlib_gray(h_with_mask, "With mask", 2, 'b')
    show_hist_with_matplotlib_gray(h_with_mask_eq, "With mask equalize", 3, 'm')
    plt.show()

    # Apply bilateral filter in order to reduce noise while keeping the edges sharp
    new_img = cv2.bilateralFilter(new_img, 20, 30, 30)

    # Simple thresholding of the image
    ret1, thresh1 = cv2.threshold(new_img, 50, 255, cv2.THRESH_BINARY)

    # Adaptive thresholding of the image
    adaptive_thresh = cv2.adaptiveThreshold(new_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # ==================================================
    # filter the ref image to make the edge sharp
    filter_ref_img = cv2.GaussianBlur(ref_img, (25, 25), 0)

    # Apply a filter to the image
    _, thresh_ref_img = cv2.threshold(filter_ref_img, 75, 255, cv2.THRESH_BINARY)

    # find the contour of the image
    contours, hierarchy = cv2.findContours(thresh_ref_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw the contour in the image
    cv2.drawContours(ref_img, contours, -1, (0, 255, 0), 3)

    show_imgs([ref_img, img, filter_ref_img, thresh_ref_img])'''