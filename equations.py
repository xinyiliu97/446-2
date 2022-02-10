
import numpy as np
import spectral
from scipy import sparse

class KdVEquation:
    
    def __init__(self, domain, u):
        # store data we need for later, make M and L matrices
        self.dtype = u.dtype
        dtype = self.dtype
        self.u = u
        self.domain = domain
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.RHS = spectral.Field(domain, dtype=dtype) # 6u*dudx
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.RHS], dtype=dtype)

        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        if self.dtype == np.complex128:
            diag = -1j*x_basis.wavenumbers(self.dtype)**3
            print("type is complex")
            p.L = sparse.diags(diag)
        elif self.dtype == np.float64:
            upper_diag = np.zeros(x_basis.N-1)
            upper_diag[::2] = x_basis.wavenumbers(self.dtype)[::2]**3
            lower_diag = -upper_diag
            D = sparse.diags([upper_diag, lower_diag], offsets=(1,-1))
            L = D.tocsr()
            # L[0,0] = 1
            # L[1,1] = 1
            p.L = L
    def evolve(self, timestepper, dt, num_steps): # take timesteps
        ts = timestepper(self.problem)
        x_basis = self.domain.bases[0]
        u = self.u
        dudx = self.dudx
        RHS = self.RHS

        for i in range(num_steps):
            # need to calculate -u*ux and put it into RHS
            if self.dtype == np.complex128: 
                u.require_coeff_space()
                dudx.require_coeff_space()
                dudx.data = 1j*x_basis.wavenumbers(self.dtype)*u.data
                u.require_grid_space(scales=128//x_basis.N)
                dudx.require_grid_space(scales=128//x_basis.N)
                RHS.require_grid_space(scales=128//x_basis.N)
                RHS.data = 6*u.data * dudx.data
            elif self.dtype == np.float64:
                #building RHS matrix
                upper_diag = np.zeros(x_basis.N-1)
                upper_diag[::2] = -x_basis.wavenumbers(self.dtype)[::2]
                lower_diag = -upper_diag
                D = sparse.diags([upper_diag, lower_diag], offsets=(1,-1))
                #apply D on u data
                u.require_coeff_space()
                dudx.require_coeff_space()
                dudx.data = D @ u.data
                dudx.require_grid_space(scales=128//x_basis.N)
                u.require_grid_space(scales=128//x_basis.N)
                dudx.require_grid_space(scales=128//x_basis.N)
                RHS.require_grid_space(scales=128//x_basis.N)
                RHS.data = 6 * u.data * dudx.data
            # take timestep
            ts.step(dt)

class SHEquation:

    def __init__(self, domain, u):
        self.dtype = u.dtype
        dtype = self.dtype
        self.u = u
        self.domain = domain
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.RHS = spectral.Field(domain, dtype=dtype) 
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.RHS], dtype=dtype)
        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        if self.dtype == np.complex128:
            diag = (1-x_basis.wavenumbers(dtype)**2)**2
            D = sparse.diags(diag)
            p.L = D.tocsr()
            
        elif self.dtype == np.float64:
            diag = (1-x_basis.wavenumbers(dtype)**2)**2
            D = sparse.diags(diag)
            p.L = D.tocsr()

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        x_basis = self.domain.bases[0]
        u = self.u
        dudx = self.dudx
        RHS = self.RHS
        
        for i in range(num_steps):
            u.require_grid_space(scales=128//x_basis.N)
            RHS.require_grid_space(scales=128//x_basis.N)
            RHS.data = -0.3*u.data - (u.data**3 - 1.8*u.data**2)

            # take timestep
            ts.step(dt)

