import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

##Funções Auxuliares
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
    result = cv.medianBlur(result,3)

    result = cv.morphologyEx(result, cv.MORPH_CLOSE, np.ones((5,5), np.uint8))
    result = cv.morphologyEx(result, cv.MORPH_OPEN, np.ones((5,5), np.uint8))

    return result


# Carregando imagens
original = cv.imread("./images/input/11.png")
assert original is not None, "file could not be read, check with os.path.exists()"

##Start Image Processing
##Start YCBCR Image Processing
ycbcr = cv.cvtColor(original, cv.COLOR_BGR2YCrCb)
y_channel, cr_channel, cb_channel = cv.split(ycbcr) # Pre-passo - Pegar um canal - Grey
y_channel_equalized = cv.equalizeHist(y_channel) # Passo 1 - Equalizando somente o canal de brilho
ycbcr_equalized = cv.merge((y_channel_equalized, cr_channel, cb_channel)) # Juntar os canais apos alterar Y
ycbcr_equalized_rgb = cv.cvtColor(original, cv.COLOR_YCrCb2RGB) # Voltar pra RGB
gauss_blur = cv.GaussianBlur(ycbcr_equalized_rgb,(3,3),0) # Passo 2 - Smoothing, low-pass,  GaussianBlur
##End YCBCR Image Processing

##Start HSV Image Processing
hsv = cv.cvtColor(original, cv.COLOR_BGR2HSV)
h_channel, s_channel, v_channel = cv.split(hsv) # Pre-passo - Pegar um canal - Grey
s_channel_equalized = cv.equalizeHist(s_channel) # Passo 1 - Equalizando somente o canal de brilho
s_equalized = cv.merge((h_channel, s_channel_equalized, v_channel)) # Juntar os canais apos alterar Y
hsv_equalized_rgb = cv.cvtColor(original, cv.COLOR_HSV2RGB) # Voltar pra RGB
gauss_blur = cv.GaussianBlur(hsv_equalized_rgb,(3,3),0) # Passo 2 - Smoothing, low-pass,  GaussianBlur
##End HSV Image Processing
##End Image Processing

##Start Mask Building
##Start YCBCR Masking
ycbcr = cv.cvtColor(gauss_blur, cv.COLOR_RGB2YCrCb) # Imagem filtrada em ycbcr
ycbcr_mask = ycbcr_detection(ycbcr) # faz mascara ycbcr
##End YCBCR Masking

##Start HSV Masking
hsv = cv.cvtColor(gauss_blur, cv.COLOR_RGB2HSV) # Imagem filtrada em hsv
hsv_mask = hsv_detection(hsv) # faz mascara hsv
##End HSV Masking
##End Mask Building

##Start Result Display
result = get_final_mask(ycbcr_mask, hsv_mask)
skin_image = cv.bitwise_and(original, original, mask=result)

# Resultados

cv.imshow('result y channel', y_channel)
cv.imshow('result y equalized', y_channel)
cv.imshow('result gauss', gauss_blur)
cv.imshow('result ycbcr', ycbcr)
cv.imshow('result hsv', hsv)
cv.imshow('result mask', result)
cv.imshow('result skin', skin_image)
cv.waitKey(0)
cv.destroyAllWindows()
##End Result Display

### Apendice
# Suavização
# https://www.youtube.com/watch?v=YVBxM64kpkU&t=206s

def low_pass_filter(image):
    dft = cv.dft(np.float32(image),flags = cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow,ccol = rows//2 , cols//2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-20:crow+20, ccol-20:ccol+20] = 1

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])

    filtered_image = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    return filtered_image
