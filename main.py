import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Carregando imagens
original = cv.imread("./images/input/11.png")
assert original is not None, "file could not be read, check with os.path.exists()"

# Suavização
# https://www.youtube.com/watch?v=YVBxM64kpkU&t=206s

def low_pass_filter(image):
    dft = cv.dft(np.float32(image),flags = cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow,ccol = rows//2 , cols//2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-25:crow+25, ccol-25:ccol+25] = 1

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])

    filtered_magnitude_spectrum = cv.normalize(magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
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


# Pre processamento
ycbcr = cv.cvtColor(original, cv.COLOR_BGR2YCrCb)
y_channel, cr_channel, cb_channel = cv.split(ycbcr)
y_channel_equalized = cv.equalizeHist(y_channel) # Equalizando somente o canal de brilho
low_pass = low_pass_filter(y_channel_equalized)
ycbcr_low_passed = cv.merge((low_pass, cr_channel, cb_channel))
ycbcr_equalized_rgb = cv.cvtColor(ycbcr_low_passed, cv.COLOR_YCrCb2RGB)

# Detecção HSV e YCbCr
hsv_mask = hsv_detection(ycbcr_equalized_rgb)
ycbcr_mask = ycbcr_detection(ycbcr_equalized_rgb)

# Junção dos resultados
result = get_final_mask(ycbcr_mask, hsv_mask)
skin_image = cv.bitwise_and(original, original, mask=result)

# Salvar resultados

cv.imwrite(f"./images/results/1_y_channel_equalized.png", y_channel_equalized)
cv.imwrite(f"./images/results/2_ycbcr_low_passed.png", ycbcr_low_passed)
cv.imwrite(f"./images/results/3_hsv_mask.png", hsv_mask)
cv.imwrite(f"./images/results/4_ycbcr_mask.png", ycbcr_mask)
cv.imwrite(f"./images/results/5_result.png", result)
cv.imwrite(f"./images/results/6_skin_image.png", skin_image)

cv.waitKey(0)
cv.destroyAllWindows()