import numpy as np
import scipy.stats as st

# Gaussian Kernel generation 
def gkern(sigma):
    kernlen = 2*int(3 * sigma + 0.5) + 1
    x = np.linspace(-sigma, sigma, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class fourier():
    # Fourier 2D fft using 1D fft
    def dft(self, img):
        return np.fft.fft(np.fft.fft(img, axis=0), axis=1)

    def idft(self, img):
        conjugate = np.conj(img)
        reconstruct = self.dft(conjugate)/(img.shape[0]*img.shape[1])
                
        return reconstruct

