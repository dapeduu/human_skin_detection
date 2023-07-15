import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Carregando imagens
original = cv.imread("./images/input/1.png")
assert original is not None, "file could not be read, check with os.path.exists()"


# Equalização de histograma

ycbcr = cv.cvtColor(original, cv.COLOR_BGR2YCrCb)
ycbcr[:,:,0] = cv.equalizeHist(ycbcr[:, :,0]) # Equalizando somente o canal de brilho

# Suavização
# https://www.youtube.com/watch?v=YVBxM64kpkU&t=206s

fourier_transform = np.fft.fft2(ycbcr[:,:,0])
fourier_shift = np.fft.fftshift(fourier_transform)
magnitude_spectrum = 20*np.log(np.abs(fourier_shift))

# Detecção HSV

# Detecção YCbCr

# Resultados
# plt.imshow(np.log1p(np.abs(fourier_shift)),
#            cmap='gray')
# plt.axis('off')
# plt.show()

# plt.imshow(np.log1p(np.abs(filtro)),
#            cmap='gray')
# plt.axis('off')
# plt.show()

# plt.imshow(np.log1p(np.abs(transformada_inversa)),
#            cmap='gray')
# plt.axis('off')
# plt.show()

plt.subplot(121),plt.imshow(cv.cvtColor(ycbcr, cv.COLOR_YCrCb2RGB))
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()