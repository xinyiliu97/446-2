
import spectral
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

class SoundWaves:
    
    def __init__(self, domain, u, pre, p0):
        self.u = u
        self.pre = pre
        self.domain = domain
        self.dtype = dtype = u.dtype
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.p_RHS = spectral.Field(domain, dtype=dtype)
        self.p0 = p0
        
        self.problem = spectral.InitialValueProblem(domain, [u, pre], [self.u_RHS, self.p_RHS], num_BCs=2, dtype=dtype)
        p = self.problem.pencils[0]
        self.x_basis = domain.bases[0]
        N = self.x_basis.N
        self.interval = self.x_basis.interval
        lowerbd = self.interval[0]
        upperbd = self.interval[1]
        lenbd = upperbd - lowerbd
        # build the D and C matrices
        
        diag = (2/lenbd) * (np.arange(N-1)+1)
        D = sparse.diags(diag, offsets=1)
        self.D = D
        
        
        diag0 = np.ones(N)/2
        diag0[0] = 1
        diag2 = -np.ones(N-2)/2
        C = sparse.diags((diag0, diag2), offsets=(0,2))
        self.C = C
        # M matrix
        M = sparse.csr_matrix((2*N+2, 2*N+2))

        M[:N, :N] = C
        M[N:2*N, N:2*N] = C

        # L matrix
        Z = np.zeros((N,N))
        L = sparse.bmat([[Z, D],
                         [D, Z]])
    
        BC_rows = np.zeros((2, 2*N))
        i = np.arange(N)

        BC_rows[0, :N] = (-1)**i
        BC_rows[1, :N] = (+1)**i

        corner = np.zeros((2, 2))

        cols = np.zeros((2*N, 2))
        cols[  N-1, 0] = 1
        cols[2*N-1, 1] = 1

        L = sparse.bmat([[L, cols],
                         [BC_rows, corner]])

        p.M = M
        p.L = L
        p.L.eliminate_zeros()
        p.M.eliminate_zeros()
        
        self.t = 0
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        pre = self.pre
        p_RHS = self.p_RHS
        p0 = self.p0
        x_basis = self.x_basis
        D = self.D
        C = self.C
        gridnum = 256
        for i in range(num_steps):
            u.require_coeff_space()
            pre.require_coeff_space()
            p_RHS.require_coeff_space()
            p0.require_coeff_space()
            p_RHS.data = spla.spsolve(C, D@u.data)
            p_RHS.require_grid_space(scales=gridnum//x_basis.N)
            p0.require_grid_space(scales=gridnum//x_basis.N)
            p_RHS.data = (1-p0.data) * p_RHS.data
            p_RHS.require_coeff_space()
            p_RHS.data = C@p_RHS.data
        
            ts.step(dt, 0.0)
            self.t += dt

class CGLEquation:

    def __init__(self, domain, u):
        self.u = u
        self.b = 0.5
        self.c = -1.76
        self.domain = domain
        self.dtype = dtype = u.dtype
        self.ux = spectral.Field(domain, dtype=dtype)
        ux = self.ux
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.ux_RHS = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u, ux], [self.u_RHS, self.ux_RHS], num_BCs=2, dtype=dtype)
        
        p = self.problem.pencils[0]
        
        self.x_basis = domain.bases[0]
        self.N = self.x_basis.N
        N = self.N
        self.interval = self.x_basis.interval
        lowerbd = self.interval[0]
        upperbd = self.interval[1]
        lenbd = upperbd - lowerbd

        
        # build the D and C matrices
        
        diag = (2/lenbd) * (np.arange(N-1)+1)
        D = sparse.diags(diag, offsets=1)
        self.D = D
        
        
        diag0 = np.ones(N)/2
        diag0[0] = 1
        diag2 = -np.ones(N-2)/2
        C = sparse.diags((diag0, diag2), offsets=(0,2))
        self.C = C
        
        # M matrix
        M = sparse.csr_matrix((2*N+2, 2*N+2))
        M[N:2*N, :N] = C

        # L matrix
        Z = np.zeros((N,N))

        L = sparse.bmat([[D, -C],
                         [-C, -(1+1j*self.b)*D]], dtype=dtype)

        BC_rows = np.zeros((2, 2*N))
        i = np.arange(N)
        BC_rows[0, :N] = (-1)**i
        BC_rows[1, :N] = (+1)**i

        corner = np.zeros((2, 2))

        cols = np.zeros((2*N, 2))
        cols[  N-1, 0] = 1
        cols[2*N-1, 1] = 1

        L = sparse.bmat([[L, cols],
                         [BC_rows, corner]])

        p.M = M
        p.L = L
        p.L.eliminate_zeros()
        p.M.eliminate_zeros()
        
        self.t = 0
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        ux = self.ux
        ux_RHS = self.ux_RHS
        c = self.c
        C = self.C
        N = self.N
        gridnum = 256
        for i in range(num_steps):
            u.require_coeff_space()
            ux.require_coeff_space()
            ux_RHS.require_coeff_space()
            u.require_grid_space(scales=gridnum//N)
            ux.require_grid_space(scales=gridnum//N)
            ux_RHS.require_grid_space(scales=gridnum//N)
            ux_RHS.data = -(1+1j*c) * np.abs(u.data)**2  * u.data
            ux_RHS.require_coeff_space()
            ux_RHS.data = C @ ux_RHS.data
            ts.step(dt, [0,0])
            self.t += dt

class BurgersEquation:
    
    def __init__(self, domain, u, nu):
        dtype = u.dtype
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)
        
        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        p.L = -nu*D@D
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        u_RHS = self.u_RHS
        for i in range(num_steps):
            dudx.require_coeff_space()
            u.require_coeff_space()
            dudx.data = u.differentiate(0)
            u.require_grid_space()
            dudx.require_grid_space()
            u_RHS.require_grid_space()
            u_RHS.data = -u.data*dudx.data
            ts.step(dt)


class KdVEquation:
    
    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 3/2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)
        
        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        p.L = D@D@D
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        u_RHS = self.u_RHS
        for i in range(num_steps):
            dudx.require_coeff_space()
            u.require_coeff_space()
            dudx.data = u.differentiate(0)
            u.require_grid_space(scales=self.dealias)
            dudx.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 6*u.data*dudx.data
            ts.step(dt)


class SHEquation:

    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)

        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        op = I + D@D
        p.L = op @ op + 0.3*I

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        u_RHS = self.u_RHS
        for i in range(num_steps):
            u.require_coeff_space()
            u.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 1.8*u.data**2 - u.data**3
            ts.step(dt)



