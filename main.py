import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Carregando imagens
original = cv.imread("./images/input/11.png")
assert original is not None, "file could not be read, check with os.path.exists()"


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


ycbcr = cv.cvtColor(original, cv.COLOR_BGR2YCrCb)

y_channel, cr_channel, cb_channel = cv.split(ycbcr)
y_channel_equalized = cv.equalizeHist(y_channel) # Equalizando somente o canal de brilho
ycbcr_equalized = cv.merge((y_channel_equalized, cr_channel, cb_channel))
ycbcr_equalized_rgb = cv.cvtColor(original, cv.COLOR_YCrCb2RGB)

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