import cv2 as cv
import numpy as np

# Carregando imagens
original = cv.imread("./images/input/1.png")
assert original is not None, "file could not be read, check with os.path.exists()"

# Equalização de histograma

ycbcr = cv.cvtColor(original, cv.COLOR_BGR2YCrCb)
ycbcr[:,:,0] = cv.equalizeHist(ycbcr[:, :,0]) # Equalizando somente o canal de brilho

hsv = cv.cvtColor(original, cv.COLOR_BGR2HSV)
hsv[:,:,0] = cv.equalizeHist(hsv[:, :,0]) # Equalizando somente o canal de brilho

# Suavização
averagingBlurYCbCr = cv.blur(ycbcr,(5,5))
gaussianBlurYCbCr = cv.GaussianBlur(ycbcr,(5,5),0)
medianBlurYCbCr = cv.medianBlur(ycbcr,5)
bilateralFilterYCbCr = cv.bilateralFilter(ycbcr,9,75,75)

averagingBlurHSV = cv.blur(hsv,(5,5))
gaussianBlurHSV = cv.GaussianBlur(hsv,(5,5),0)
medianBlurHSV = cv.medianBlur(hsv,5)
bilateralFilterHSV = cv.bilateralFilter(hsv,9,75,75)

# Detecção HSV

# Detecção YCbCr

# Resultados
cv.imshow('Original', original)
cv.imshow('Equalized YCbCr', cv.cvtColor(ycbcr, cv.COLOR_YCrCb2BGR))
cv.imshow('Average YCbCr', cv.cvtColor(averagingBlurYCbCr, cv.COLOR_YCrCb2BGR))
cv.imshow('Gaussian YCbCr', cv.cvtColor(gaussianBlurYCbCr, cv.COLOR_YCrCb2BGR))
cv.imshow('Median YCbCr', cv.cvtColor(medianBlurYCbCr, cv.COLOR_YCrCb2BGR))
cv.imshow('Bilateral YCbCr', cv.cvtColor(bilateralFilterYCbCr, cv.COLOR_YCrCb2BGR))

cv.imshow('Equalized HSV', cv.cvtColor(ycbcr, cv.COLOR_BGR2HSV))
cv.imshow('Average HSV', cv.cvtColor(averagingBlurHSV, cv.COLOR_BGR2HSV))
cv.imshow('Gaussian HSV', cv.cvtColor(gaussianBlurHSV, cv.COLOR_BGR2HSV))
cv.imshow('Median HSV', cv.cvtColor(medianBlurHSV, cv.COLOR_BGR2HSV))
cv.imshow('Bilateral HSV', cv.cvtColor(bilateralFilterHSV, cv.COLOR_BGR2HSV))
cv.waitKey(0) # Caso mostremos uma figura
cv.destroyAllWindows()