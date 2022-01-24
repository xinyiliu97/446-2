
import numpy as np
import scipy.fft

class Basis:

    def __init__(self, N, interval):
        self.N = N
        self.interval = interval


class Fourier(Basis):

    def __init__(self, N, interval=(0, 2*np.pi)):
        super().__init__(N, interval)

    def grid(self, scale=1):
        N_grid = int(np.ceil(self.N*scale))
        return np.linspace(self.interval[0], self.interval[1], num=N_grid, endpoint=False)

    def transform_to_grid(self, data, axis, dtype, scale=1):
        if dtype == np.complex128:
            return self._transform_to_grid_complex(data, axis, scale)
        elif dtype == np.float64:
            return self._transform_to_grid_real(data, axis, scale)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def transform_to_coeff(self, data, axis, dtype):
        if dtype == np.complex128:
            return self._transform_to_coeff_complex(data, axis)
        elif dtype == np.float64:
            return self._transform_to_coeff_real(data, axis)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def _transform_to_grid_complex(self, data, axis, scale):
        scale = scale
        N = self.N
        N_data = data.shape[0]
        N_expand = scale * N
#         print(N_expand)
        n = np.arange(0, N_expand)
        x_basis = self.grid(scale = scale)
#         print(x_basis.shape[0])
        coeff_expand = np.zeros(int(scale*N), dtype=np.complex128)
        coeff_expand[0:N//2] = data[0:N//2]
        coeff_expand[-N//2:] = data[-N//2:]
        M = np.exp(1j*n[None,:]*x_basis[:,None])
        grid_data = M @ coeff_expand
#         print(grid_data.shape[0])
        return(grid_data)
    def _transform_to_coeff_complex(self, data, axis):
        N = data.shape[0]
        print(N)
        n = np.arange(0, N)
        scale = N/self.N
        x_basis = self.grid(N/self.N)
        M = np.exp(-1j*n[None,:]*x_basis[:,None])/N
        coeff_data = M @ data
        coeff_data_decrememt = np.zeros(self.N, dtype=np.complex128)
        coeff_data_decrememt[0:self.N//2] = coeff_data[0:self.N//2]
        coeff_data_decrememt[-self.N//2:] = coeff_data[-self.N//2:]
        return(coeff_data_decrememt)
    def _transform_to_grid_real(self, data, axis, scale):
        N = self.N
        N_expand = scale * N
        n = np.arange(0, N_expand//2)
        x_basis = self.grid(scale = scale)
        #complex_data = np.zeros(x_basis.N//2, dtype=np.complex128)
        coeff_exp_real = np.zeros(int(N_expand)//2)
        coeff_exp_imag = np.zeros(int(N_expand)//2)
        #complex_data = np.zeros(x_basis.N//2, dtype=np.complex128)
        coeff_exp_real[0:N//4] = data[0:N//2:2]
        coeff_exp_real[-N//4:] = data[-N//2::2]

        coeff_exp_imag[0:N//4] = data[1:N//2:2]
        coeff_exp_imag[-N//4:] = data[-N//2+1::2]
        M = np.cos(x_basis[:,None]*n[None, :])
        I = np.sin(x_basis[:,None]*n[None, :])

        grid_data = M @ coeff_exp_real - I @ coeff_exp_imag
        
        return(grid_data)


    def _transform_to_coeff_real(self, data, axis):
        N = data.shape[0]
        scale = N/self.N
        
        n = np.arange(0, N//2)
        x_basis = self.grid(N/self.N)
        M = 2*np.cos(n[:,None]*x_basis[None, :])/N
        I = -2*np.sin(n[:,None]*x_basis[None, :])/N
        
        
        coeff_real = M @ data
        coeff_imag = I @ data
        coeff_data = np.zeros(N)
        coeff_data[::2] = coeff_real[:N//2]
        coeff_data[1::2] = coeff_imag[:N//2]
        coeff_data_dec = np.zeros(self.N)
        coeff_data_dec[0:self.N//2] = coeff_data[0:self.N//2]
        coeff_data_dec[-self.N//2:] = coeff_data[-self.N//2:]
        coeff_data_dec[1] = 0
        zeropt = coeff_data_dec[0]
        coeff_data_dec[0] = zeropt/2
        return(coeff_data_dec)
        


class Domain:

    def __init__(self, bases):
        if isinstance(bases, Basis):
            # passed single basis
            self.bases = (bases, )
        else:
            self.bases = tuple(bases)
        self.dim = len(self.bases)

    @property
    def coeff_shape(self):
        return [basis.N for basis in self.bases]

    def remedy_scales(self, scales):
        if scales is None:
            scales = 1
        if not hasattr(scales, "__len__"):
            scales = [scales] * self.dim
        return scales


class Field:

    def __init__(self, domain, dtype=np.float64):
        self.domain = domain
        self.dtype = dtype
        self.data = np.zeros(domain.coeff_shape, dtype=dtype)
        self.coeff = np.array([True]*self.data.ndim)

    def towards_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        axis = np.where(self.coeff == False)[0][0]
        self.data = self.domain.bases[axis].transform_to_coeff(self.data, axis, self.dtype)
        self.coeff[axis] = True

    def require_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        else:
            self.towards_coeff_space()
            self.require_coeff_space()

    def towards_grid_space(self, scales=None):
        if not self.coeff.any():
            # already in full grid space
            return
        axis = np.where(self.coeff == True)[0][-1]
        scales = self.domain.remedy_scales(scales)
        self.data = self.domain.bases[axis].transform_to_grid(self.data, axis, self.dtype, scale=scales[axis])
        self.coeff[axis] = False

    def require_grid_space(self, scales=None):
        if not self.coeff.any(): 
            # already in full grid space
            return
        else:
            self.towards_grid_space(scales)
            self.require_grid_space(scales)



