import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Carregando imagens
original = cv.imread("./images/input/1.png")
assert original is not None, "file could not be read, check with os.path.exists()"


# Equalização de histograma

ycbcr = cv.cvtColor(original, cv.COLOR_BGR2YCrCb)
y_channel, cr_channel, cb_channel = cv.split(ycbcr)
y_channel_equalized = cv.equalizeHist(y_channel) # Equalizando somente o canal de brilho

ycbcr_equalized = cv.merge((y_channel_equalized, cr_channel, cb_channel))

ycbcr_equalized_rgb = cv.cvtColor(ycbcr_equalized, cv.COLOR_YCrCb2BGR)

hsv_equalized = cv.cvtColor(ycbcr_equalized, cv.COLOR_BGR2HSV)

# Suavização
# https://www.youtube.com/watch?v=YVBxM64kpkU&t=206s

# fourier_transform = np.fft.fft2(equalized)
# fourier_shift = np.fft.fftshift(fourier_transform)

# rows, cols, _ = original.shape
# crow, ccol = int(rows / 2), int(cols / 2)
# mask = np.zeros((rows, cols), np.uint8)
# mask[crow - 50:crow + 50, ccol - 50:ccol + 50] = 1

# fourier_shift_filtered = fourier_shift * mask
# inverse_shift = np.fft.ifftshift(fourier_shift_filtered)
# filtered_y_channel = np.fft.ifft2(inverse_shift)
# filtered_y_channel = np.abs(filtered_y_channel)
# filtered_ycbcr_image = cv.merge((filtered_y_channel, cb_channel, cr_channel))

# Detecção HSV
def hsv_detection(bgr_img):
    hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
    hsv_mask = cv.inRange(hsv_img, (0, 15, 0), (50,170,255))
    hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    return hsv_mask

# Detecção YCbCr
def ycbcr_detection(bgr_img):
    ycbcr_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2YCrCb)
    ycbcr_mask = cv.inRange(ycbcr_img, (0, 135, 85), (255,180,135))
    ycbcr_mask = cv.morphologyEx(ycbcr_mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    return ycbcr_mask

# Juntar resultados
hsv_mask = hsv_detection(original)
ycbcr_mask = ycbcr_detection(original)

result = cv.bitwise_and(ycbcr_mask, hsv_mask)
result = cv.medianBlur(result,3)
result = cv.morphologyEx(result, cv.MORPH_OPEN, np.ones((4,4), np.uint8))
result = cv.bitwise_not(result)

hsv_mask_result = cv.bitwise_not(hsv_detection(original))
ycbcr_mask_result = cv.bitwise_not(ycbcr_detection(original))
# Resultados


plt.subplot(121), plt.imshow(hsv_mask_result, cmap='gray')
plt.title('HSV Resultado'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(ycbcr_mask_result, cmap='gray')
plt.title('YCbCr Resultado'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121), plt.imshow(original, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(result, cmap='gray')
plt.title('Resultado'), plt.xticks([]), plt.yticks([])
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()