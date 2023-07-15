import cv2 as cv
import numpy as np

# Carregando imagens
original = cv.imread("./images/input/1.jpg")
assert original is not None, "file could not be read, check with os.path.exists()"

# Equalização de histograma

ycbcr = cv.cvtColor(original, cv.COLOR_BGR2YCrCb)
ycbcr[:,:,0] = cv.equalizeHist(ycbcr[:, :,0]) # Equalizando somente o canal de brilho

# Suavização

# Detecção HSV

# Detecção YCbCr

# Resultados
cv.imshow('Original', original)
cv.imshow('Equalized', cv.cvtColor(ycbcr, cv.COLOR_YCrCb2BGR))

cv.waitKey(0) # Caso mostremos uma figura
cv.destroyAllWindows()