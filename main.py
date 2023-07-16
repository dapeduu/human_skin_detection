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

# Suavização
# https://www.youtube.com/watch?v=YVBxM64kpkU&t=206s

def low_pass_filter(image):
    dft = cv.dft(np.float32(image),flags = cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow,ccol = rows//2 , cols//2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-100:crow+100, ccol-100:ccol+100] = 1

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])

    filtered_image = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    return filtered_image

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

low_pass = low_pass_filter(y_channel_equalized)
ycbcr_equalized = cv.merge((low_pass, cr_channel, cb_channel))

ycbcr_equalized_rgb = cv.cvtColor(ycbcr_equalized, cv.COLOR_YCrCb2RGB)

hsv_mask = hsv_detection(ycbcr_equalized_rgb)
ycbcr_mask = ycbcr_detection(ycbcr_equalized_rgb)

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