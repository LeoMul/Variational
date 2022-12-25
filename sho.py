import numpy as np 
import scipy 

def basis_function(x,a):
    return np.exp(-a*x*x) 


def integrate(func,h):
    return scipy.integrate.simpson(func,dx = h)

def kinetic(x,a):
    return a * (1.0 - 2.0*a*x*x) * basis_function(x,a)


def energy(a):
    x,h = np.linspace(-5000,5000,1000000,retstep=True) 
    psi = basis_function(x,a)
    psisq = psi*psi
    kinetic =  a*(1.0 - 2.0*a*x*x) * psisq
    pot = 0.5*x*x*psisq
    kineticportion = integrate(kinetic,h)
    norm = integrate(psisq,h)
    potportion = integrate(pot,h)
    print("expect:",a/2.0,0.125/a)
    print(a,kineticportion/ norm ,potportion/ norm )

    energy = kineticportion + potportion
    #print(a,energy)
    energy = energy 
    return energy 

res = scipy.optimize.minimize(energy,0.51)

print(energy(res.x))
