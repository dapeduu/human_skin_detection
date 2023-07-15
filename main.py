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

## Filtro passa-baixas ideal
M,N,_ = original.shape
H = np.zeros((M,N), dtype=np.float32)
D0 = 100
for u in range(M):
    for v in range(N):
        D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        if D <= D0:
            H[u,v] = 1
        else:
            H[u,v] = 0

filtro = fourier_shift * H

fourier_shift_inverso = np.fft.ifftshift(filtro)
transformada_inversa = np.abs(np.fft.ifft2(fourier_shift_inverso))
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

plt.imshow(np.log1p(np.abs(transformada_inversa)),
           cmap='gray')
plt.axis('off')
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()