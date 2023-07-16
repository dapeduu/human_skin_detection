import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Carregando imagens
original = cv.imread("./images/input/11.png")
assert original is not None, "file could not be read, check with os.path.exists()"


# Equalização de histograma

ycbcr = cv.cvtColor(original, cv.COLOR_BGR2YCrCb)

y_channel, cr_channel, cb_channel = cv.split(ycbcr)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
y_channel_equalized = cv.equalizeHist(y_channel) # Equalizando somente o canal de brilho
ycbcr_equalized = cv.merge((y_channel_equalized, cr_channel, cb_channel))
ycbcr_equalized_rgb = cv.cvtColor(original, cv.COLOR_YCrCb2RGB)

# Suavização
# https://www.youtube.com/watch?v=YVBxM64kpkU&t=206s

# fourier_transform = np.fft.fft2(y_channel)
# fourier_shift = np.fft.fftshift(fourier_transform)

# rows, cols = y_channel.shape
# crow, ccol = int(rows / 2), int(cols / 2)
# mask = np.zeros((rows, cols), np.uint8)
# mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

# fourier_shift_filtered = fourier_shift * mask
# inverse_shift = np.fft.ifftshift(fourier_shift_filtered)
# filtered_y_channel = np.fft.ifft2(inverse_shift)
# filtered_y_channel = np.abs(filtered_y_channel)
# filtered_ycbcr_image = cv.merge((filtered_y_channel, cb_channel, cr_channel))

def truncate_v2(image, kernel):
    m, n = kernel.shape
    m = int((m-1) / 2)

    for i in range(0, m):
        line, row = image.shape
        image = np.delete(image, line-1, 0)
        image = np.delete(image, row-1, 1)
        image = np.delete(image, 0, 0)
        image = np.delete(image, 0, 1)
    return image

def low_pass_function(image):

    # # Gaussian Filter (Smoothing)
    # kernel = np.array([[1, 2, 1],
    #                    [2, 4, 2],
    #                    [1, 2, 1]]) / 16

    # low pass filter
    kernel = np.ones((3, 3)) / 9

    convolved_image = image * kernel

    truncated_image = truncate_v2(convolved_image, kernel)

    low_pass_filtered_image = truncated_image

    return low_pass_filtered_image

# Detecção HSV
def hsv_detection(bgr_img):
    hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
    hsv_mask = cv.inRange(hsv_img, (0, 30, 37), (50,170,255))
    return hsv_mask

# Detecção YCbCr
def ycbcr_detection(bgr_img):
    ycbcr_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2YCrCb)
    ycbcr_mask = cv.inRange(ycbcr_img, (0, 133, 77), (235,173,127))
    return ycbcr_mask

# Juntar resultados
def get_final_mask(ycbcr_mask, hsv_mask):
    result = cv.bitwise_and(ycbcr_mask, hsv_mask)
    result = cv.medianBlur(result,11)

    result = cv.morphologyEx(result, cv.MORPH_CLOSE, np.ones((13,13), np.uint8))
    result = cv.morphologyEx(result, cv.MORPH_OPEN, np.ones((13,13), np.uint8))

    return result


# low_pass = apply_low_pass_filter(ycbcr_equalized_rgb, 1)

hsv_mask = hsv_detection(original)
ycbcr_mask = ycbcr_detection(original)

result = get_final_mask(ycbcr_mask, hsv_mask)
skin_image = cv.bitwise_and(original, original, mask=result)

# Resultados



plt.subplot(121), plt.imshow(hsv_mask, cmap='gray')
plt.title('HSV Resultado'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(ycbcr_mask, cmap='gray')
plt.title('YCbCr Resultado'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121), plt.imshow(result, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(skin_image, cmap='gray')
plt.title('Resultado'), plt.xticks([]), plt.yticks([])
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()