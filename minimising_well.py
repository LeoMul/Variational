import numpy as np
import scipy
import matplotlib.pyplot as plt
def psi_n(x,n):
    return (x**n) * (x-1.0) * (x+1.0)

def secondderiv(x,n):
    return (4*n + 2) * x**n + n*(n-1) * (x-1.0) * (x+1.0) * (x**(n-2))

def integrate(func,h):
    return scipy.integrate.simpson(func,dx = h)

def full_psi(x,cvector):
    psi = np.zeros(len(x))
    for j in range(0,len(cvector)):
        psi = psi + psi_n(x,j)
    return psi 

def kineticpsi(x,cvector):
    psi = np.zeros(len(x))
    for j in range(0,len(cvector)):
        psi = psi + secondderiv(x,j)
    return psi * -1.0

def energy(cvector):
    N = len(cvector)
    x,h = np.linspace(-1,1,2000,retstep=True)
    ham = construct_hamiltonian_matrix(x,h,N)
    overlap = construct_overlap_matrix(x,h,N)
    norm = 0.0
    energy = 0.0
    for i in range(0,len(cvector)):
        for j in range(0,len(cvector)):
            norm = norm + cvector[i] * cvector[j]*overlap[i,j]
            energy = energy + cvector[i] * cvector[j] * ham[i,j]

    energy = energy/norm
    print(cvector,energy)
    return energy 

def construct_hamiltonian_matrix(x,h,N):
    ham = np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            psi =  -1.0*secondderiv(x,i) * psi_n(x,j)
            ham[i,j] = integrate(psi,h)
    return ham 
def construct_overlap_matrix(x,h,N):
    ham = np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            psi =  psi_n(x,i) * psi_n(x,j)
            ham[i,j] = integrate(psi,h)
    
    return ham 

c = np.random.rand(7)

res = scipy.optimize.minimize(energy,c, method='nelder-mead',tol = 1e-6)

