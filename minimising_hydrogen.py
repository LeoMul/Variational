import numpy as np
import scipy
import matplotlib.pyplot as plt

a = [13.00773,1.962079,0.444529,0.1219492]


def psi_n(x,a):
    return np.exp(-a*x*x)

def secondderiv(x,a):
    return 2.0 * a * (2.0*a*x*x-3.0) * psi_n(x,a)

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



def construct_hamiltonian_matrix(x,h,a):
    N = len(a)
    ham = np.zeros([N,N])
    overlap = np.zeros([N,N])

    for i in range(0,N):
        for j in range(i,N):
            #basis1 = psi_n(x,a[i])
            #basis2 = psi_n(x,a[j]) 

            #minushalfsecondderivative = -1.0 * a[j] * (2.0*a[j]*x*x-3.0)*basis2*x*x
            #psisq = basis1*basis2

            #psi =  minushalfsecondderivative * basis1
            #psi2 = - x*psisq
            #psisq = x*x*psisq
            kin = (3.0*(np.pi**1.5)*a[i]*a[j]/((a[i]+a[j])**2.5))

            #print("expect kin:",kin,(3.0*(np.pi**1.5)*a[i]*a[j]/((a[i]+a[j])**2.5)))

            #pot = integrate(psi2,h)
            pot = - 2.0*np.pi*1.0/(a[i]+a[j])
            ham[i,j] = kin + pot
            overlap[i,j] = (np.pi/(a[i]+a[j]))**1.5
    return ham ,overlap


def construct_overlap_matrix(x,h,N):
    ham = np.zeros([N,N])
    for i in range(0,N):
        for j in range(i,N):
            #psi =  psi_n(x,a[i]) * psi_n(x,a[j])
            #overlap = integrate(psi*x*x,h)
            overlap = (np.pi/(a[i]+a[j]))**1.5
            #print("expect overlap:",overlap,(np.pi/(a[i]+a[j]))**1.5)

            ham[i,j] = overlap

    
    return ham 

def energy(ac):
    N = len(ac)
    #cvector = ac
    cvector = ac[0:int(N/2)]
    a = ac[int(N/2):]
    x,h = np.linspace(0.00,1000,100000,retstep=True)
    ham,overlap = construct_hamiltonian_matrix(x,h,a)
    #overlap = construct_overlap_matrix(x,h,N)
    norm = 0.0
    energy = 0.0
    for i in range(0,len(cvector)):
        for j in range(i+1,len(cvector)):
            norm = norm + 2.0*cvector[i] * cvector[j]*overlap[i,j]
            energy = energy + 2.0*cvector[i] * cvector[j] * ham[i,j]

    for i in range(0,len(cvector)):
        norm = norm + cvector[i] * cvector[i]*overlap[i,i]
        energy = energy + cvector[i] * cvector[i] * ham[i,i]


    energy = energy/norm
    return energy 

N = 12
c = np.ones(N)
#c[int(N/2)] = 15.0#

minimizer_kwargs = dict(method="L-BFGS-B")
res = scipy.optimize.basinhopping(energy, c, minimizer_kwargs=minimizer_kwargs)

#res = scipy.optimize.minimize(energy,c, method='nelder-mead',tol = 1e-10)
C = res.x
#print(res.x)
print(energy(res.x))

a = C[int(N/2):]
a.sort()
print(a)
x = np.linspace(0.0,10,400)
wf = np.zeros(len(x))
#for j in range(N):
#    wf = wf + C[j] * psi_n(x,a[j])

#plt.plot(x,x*wf)
#plt.show()

def min_search(starting_point):
    energy = 0.0
    minimum_point = starting_point




    return minimum_point,energy 