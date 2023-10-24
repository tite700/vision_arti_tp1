import cv2
import numpy as np
import sys
import os


def load_data(path):
    """Load the images and put the result in list
    the first image of the list is the reference image"""
    chambre_imgs = []
    chambre_imgs.append(cv2.imread(path + "\Images\Chambre\Reference.JPG"))
    for i in range(67, 74):
        chambre_imgs.append(cv2.imread(path + f"\Images\Chambre\IMG_65{i}.JPG"))

    cuisine_imgs = []
    cuisine_imgs.append(cv2.imread(path + "\Images\Cuisine\Reference.JPG"))
    for i in range(62, 66):
        img = cv2.imread(path + f"\Images\Cuisine\IMG_65{i}.JPG")
        cuisine_imgs.append(cv2.imread(path + f"\Images\Cuisine\IMG_65{i}.JPG"))

    salon_imgs = []
    salon_imgs.append(cv2.imread(path + "\Images\Salon\Reference.JPG"))
    for i in range(51, 61):
         salon_imgs.append(cv2.imread(path + f"\Images\Salon\IMG_65{i}.JPG"))

    return chambre_imgs , cuisine_imgs, salon_imgs


def show_imgs(imgs):
    result_dir = "Resultats"
    os.makedirs(result_dir, exist_ok=True)

    for i, img in enumerate(imgs):
        cv2.imshow(f"image: {i}", img)
        image_name = f"result_{i}.jpg"
        image_path = os.path.join(result_dir, image_name)
        cv2.imwrite(image_path, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
        dilated_img = cv2.dilate(plane, np.ones((9, 9), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 23)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result_img = cv2.merge(result_planes)
    result_norm_img = cv2.merge(result_norm_planes)

    return result_img


def mask_for_chambre(img):
    img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

    # Use a circle for the mask
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.circle(mask, (675, 470), 275, (255, 0, 0), -1)

    # Mask the image
    masked = cv2.bitwise_and(img, img, mask=mask)

    return masked


def mask_for_cuisine(img):
    img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (375, 375), (850, 1000), (255, 0, 0), -1)

    masked = cv2.bitwise_and(img, img, mask=mask)

    return masked


def mask_for_salon(img):
    img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (100, 450), (1100, 1000), (255, 0, 0), -1)

    masked = cv2.bitwise_and(img, img, mask=mask)

    return masked


def detection_algorithm_simple(ref_img, img, original_img):
    """Simple detection algorithm"""

    # resize the image
    resized_original_img = cv2.resize(original_img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

    img_without_shadow = remove_shadow_from_img(img)
    ref_img_without_shadow = remove_shadow_from_img(ref_img)

    # cv2.imshow("img without shadow", img_without_shadow)
    # cv2.waitKey(0)

    # Convert bgr to gray
    img_gray = cv2.cvtColor(img_without_shadow, cv2.COLOR_BGR2GRAY)
    ref_img_gray = cv2.cvtColor(ref_img_without_shadow, cv2.COLOR_BGR2GRAY)

    # blur img
    blur_img = cv2.GaussianBlur(img_gray, (25, 25), 0)
    blur_ref_img = cv2.GaussianBlur(ref_img_gray, (25, 25), 0)

    # cv2.imshow("img blur", blur_img)
    # cv2.waitKey(0)

    diff_img = cv2.absdiff(blur_img, blur_ref_img)

    # cv2.imshow("diff_img", diff_img)
    # cv2.waitKey(0)

    _, thresh_img = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # cv2.imshow("thresh_img", thresh_img)
    # cv2.waitKey(0)

    dilated_img = cv2.dilate(thresh_img, np.ones((15, 15), np.uint8))

    # cv2.imshow("dilated_img", dilated_img)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    return_img = draw_bin_boxes(dilated_img, resized_original_img)

    return return_img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get the path
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "."

    # Load the data
    chambre_imgs, cuisine_imgs, salon_imgs  = load_data(path)
    imgs = []

    ref_chambre_img = chambre_imgs[0]

    ref_chambre_img = mask_for_chambre(ref_chambre_img)

    # Select an image
    for i in range(1, 7):
        img = chambre_imgs[i]

        original_img = img.copy()
        img = mask_for_chambre(img)
        img = detection_algorithm_simple(ref_chambre_img, img, original_img)

        imgs.append(img)

    ref_cuisine_img = cuisine_imgs[0]
    ref_cuisine_img = mask_for_cuisine(ref_cuisine_img)

    for i in range(1, 4):
        img = cuisine_imgs[i]

        original_img = img.copy()
        img = mask_for_cuisine(img)
        img = detection_algorithm_simple(ref_cuisine_img, img, original_img)

        imgs.append(img)

    ref_salon_img = salon_imgs[0]
    ref_salon_img = mask_for_salon(ref_salon_img)
    for i in range(1, 11):
        img = salon_imgs[i]

        original_img = img.copy()
        img = mask_for_salon(img)
        img = detection_algorithm_simple(ref_salon_img, img, original_img)

        imgs.append(img)

    show_imgs(imgs)