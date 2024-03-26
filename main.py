# menambahkan library
import cv2
import numpy as np
import random

def random_cropping(img, crop_size):
    h, w = img.shape[:2]
    if h < crop_size[0] or w < crop_size[1]:
        raise ValueError("Ukuran potongan harus lebh kecil dari dimensi gambar")
    
    x = random.randint(0, w - crop_size[1])
    y = random.randint(0, h - crop_size[0])

    return img[y:y+crop_size[0], x:x+crop_size[1]]

def half_size_center_cropping(img):
    h, w = img.shape[:2]
    crop_h, crop_w = h // 2, w // 2
    start_x = w // 4
    start_y = h // 4

    return img[start_y:start_y+crop_h, start_x:start_x+crop_w]

def invert_colors(img):
    return 255 - img

def add_salt_and_pepper_noise(image, amount=0.004):
    row, col, ch = image.shape
    s_vs_p = 0.5
    out = np.copy(image)

    num_salt = np.ceil(amount * image.size * s_vs_p)
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))

    # Salt
    for _ in range(int(num_salt)):
        i = np.random.randint(0, row - 1)
        j = np.random.randint(0, col - 1)
        out[i, j] = 1

    # Pepper
    for _ in range(int(num_pepper)):
        i = np.random.randint(0, row - 1)
        j = np.random.randint(0, col - 1)
        out[i, j] = 0

    return out

def add_gaussian_noise(image, mean=0, sigma=0.1):
    row, col, ch = image.shape
    gaussian = np.random.normal(mean, sigma, (row, col, ch))
    gaussian = gaussian.reshape(row, col, ch)
    noisy = image + gaussian
    return noisy

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def flip_vertical(image):
    return cv2.flip(image, 0)

def flip_horizontal(image):
    return cv2.flip(image, 1)

# menambahkan gambar
image = cv2.imread('Plat Kendaraan.jpeg').astype(np.float32) / 255

# menambahkan salt and pepper noise
image_sp_noise = add_salt_and_pepper_noise(image)

# manambahkan gaussian noise
image_gaussian_noise = add_gaussian_noise(image)

# random cropping
crop_size = (300, 400)

image_random_cropped = random_cropping(image_sp_noise, crop_size)
image_center_cropped = half_size_center_cropping(image_gaussian_noise)

image_random_cropped_8u = cv2.convertScaleAbs(image_random_cropped)
image_center_cropped_8u = cv2.convertScaleAbs(image_center_cropped)

# invert warna
image_random_cropped_invert = 1 - image_random_cropped
image_center_cropped_invert = 1 - image_center_cropped

image_random_cropped_invert_8u = cv2.convertScaleAbs(image_random_cropped_invert * 255)
image_center_cropped_invert_8u = cv2.convertScaleAbs(image_center_cropped_invert * 255)

# rotate
rotate_image = rotate_image(image_random_cropped_invert_8u, 45)
flipped_vertically_image = flip_vertical(image_random_cropped_invert_8u)
flipped_horizontally_image = flip_vertical(image_center_cropped_invert_8u)

# konversi ke grayscale
gray_random_cropped = cv2.cvtColor(image_random_cropped_8u, cv2.COLOR_BGR2GRAY)
gray_center_cropped = cv2.cvtColor(image_center_cropped_8u, cv2.COLOR_BGR2GRAY)

gray_random_cropped_invert = cv2.cvtColor(image_random_cropped_invert_8u, cv2.COLOR_BGR2GRAY)
gray_center_cropped_invert = cv2.cvtColor(image_center_cropped_invert_8u, cv2.COLOR_BGR2GRAY)

# menerapkan metode canny
edges_random_cropped_invert = cv2.Canny(gray_random_cropped_invert, 50, 150)
edges_center_cropped_invert = cv2.Canny(gray_center_cropped_invert, 50, 150)

# menampilkan hasil
cv2.imshow('Original image with salt and pepper noise', image_sp_noise)
cv2.imshow('Original image with gaussian noise', image_gaussian_noise)
cv2.imshow('Random Cropping', edges_random_cropped_invert)
cv2.imshow('half-size center cropping',edges_center_cropped_invert)
cv2.imshow('Rotated', rotate_image)
cv2.imshow('Flipped Vertical', flipped_vertically_image)
cv2.imshow('Flipped Horizontal', flipped_horizontally_image)

cv2.waitKey(0)
cv2.destroyAllWindows()