import numpy as np
import scipy
import matplotlib.pyplot as plt


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

def construct_t_matrix(a):
    N = len(a)
    capt = np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            capt[i,j] = (3.0*(np.pi**1.5)*a[i]*a[j]/((a[i]+a[j])**2.5)) 

    return capt

def construct_s_matrix(a):
    N = len(a)
    capt = np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            capt[i,j] = (np.pi/(a[i]+a[j]))**1.5

    return capt

def construct_a_matrix(a):
    N = len(a)
    capt = np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            capt[i,j] = - 2.0*np.pi*1.0/(a[i]+a[j])

    return capt


def construct_hamiltonian_matrix(x,h,a):
    N = len(a)
    ham = np.zeros([N,N])
    overlap = np.zeros([N,N])

    for i in range(0,N):
        for j in range(i,N):
        
            kin = (3.0*(np.pi**1.5)*a[i]*a[j]/((a[i]+a[j])**2.5))
            pot = - 4.0*np.pi*1.0/(a[i]+a[j]) #is 2 times that of Hydrogen
            ham[i,j] = kin + pot
            overlap[i,j] = (np.pi/(a[i]+a[j]))**1.5
    return ham ,overlap


def construct_q_matrix(a):
    N = len(a)
    qmatrix = np.zeros([N,N,N,N])
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
                for m in range(0,N):
                    fac = 1.0/((a[i]+a[k])*(a[j]+a[m]))
                    fac = fac * ((a[i]+a[j]+a[k]+a[m])**-0.5)
                    qmatrix[i,j,k,m] = fac
    qmatrix = qmatrix * 2.0 * (np.pi**2.5)
    return qmatrix 

def energy(ac):
    N = len(ac)
    cvector = ac
    cvector = ac[0:int(N/2)]
    a = ac[int(N/2):]

    tmatrix = construct_t_matrix(a)
    amatrix = construct_a_matrix(a)
    smatrix = construct_s_matrix(a)
    qmatrix = construct_q_matrix(a)

    norm = 0.0
    energy = 0.0

    N = len(cvector)

    for p in range(0,N):
        for q in range(0,N):
            for r in range(0,N):
                for s in range(0,N):
                    en_term = smatrix[s,q] * tmatrix[r,p]+ smatrix[r,p] * tmatrix[s,q]
                    en_term = en_term + qmatrix[p,r,q,s]
                    en_term = en_term +2.0* smatrix[s,q]*amatrix[r,p] + 2.0*smatrix[r,p]*amatrix[s,q]

                    
                    cterm =  cvector[p] * cvector[q] * cvector[r] * cvector[s]
                    energy +=  en_term * cterm
                    norm +=  cterm * smatrix[r,p] * smatrix[q,s]
    energy = energy/norm
    print(energy)
    #print(a)
    return energy 


#a = [0.298073,1.242567,5.782948,38.474970]

N = 8
c = np.ones(N)
#c[int(N/2)] = 15.0#
bounds = []

for i in range(0,int(N/2)):
    bounds.append((-np.inf,np.inf))
for i in range(int(N/2),N):
    bounds.append((0.01,np.inf))

    


minimizer_kwargs = dict(method="L-BFGS-B",bounds = bounds)
res = scipy.optimize.basinhopping(energy, c, minimizer_kwargs=minimizer_kwargs)

#res = scipy.optimize.minimize(energy,c, method='nelder-mead',tol = 1e-10)
C = res.x
print(res.x)
print(energy(res.x))

#a = C[int(N/2):]
#a.sort()
#print(a)
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