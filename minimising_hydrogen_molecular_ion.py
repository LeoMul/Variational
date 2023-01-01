import numpy as np
import scipy
import matplotlib.pyplot as plt

a = [13.00773,1.962079,0.444529,0.1219492]
pi3halfs = (np.pi ) **(1.5)

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



def construct_hamiltonian_matrix(x,h,a,R):
    N = len(a)
    ham = np.zeros([N,N])
    overlap = np.zeros([N,N])

    for i in range(0,N):
        for j in range(i,N):
        
            kin = (3.0*(np.pi**1.5)*a[i]*a[j]/((a[i]+a[j])**2.5))

            pot = - 2.0*np.pi*1.0/(a[i]+a[j])
            interaction = -pi3halfs / (R*(a[i]+a[j])**1.5)
            interaction*= scipy.special.erf(R*((a[i]+a[j])**0.5))
            #print("interaction",interaction)
            ham[i,j] = kin + pot + interaction
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
    R = ac[0]

    N = (N-1)/2
    cvector = ac[1:int((N+1))]
    
    a = ac[int((N+1)):]
    x,h = np.linspace(0.00,1000,100000,retstep=True)
    ham,overlap = construct_hamiltonian_matrix(x,h,a,R)
    #overlap = construct_overlap_matrix(x,h,N)
    norm = 0.0
    energy = 0.0
    for i in range(0,len(cvector)):
        for j in range(i+1,len(cvector)):
            norm = norm + 2.0*cvector[i] * cvector[j]*overlap[i,j]
            energy = energy + 2.0*cvector[i] * cvector[j] * (ham[i,j]+ overlap[i,j]/R )

    for i in range(0,len(cvector)):
        norm = norm + cvector[i] * cvector[i]*overlap[i,i]
        energy = energy + cvector[i] * cvector[i] * (ham[i,i]+ overlap[i,i]/R)


    energy = energy/norm
    #print(energy)
    return energy 

smallN = 6
c = np.ones(2*smallN+1)
#c[int(N/2)] = 15.0#

bounds = []
bounds.append((0.0001,np.inf))
for i in range(1,int(smallN+1)):
    bounds.append((-np.inf,np.inf))
for i in range(int(smallN+1),2*smallN+1):
    bounds.append((0.01,np.inf))

print((bounds))

minimizer_kwargs = dict(method="L-BFGS-B")
res = scipy.optimize.basinhopping(energy, c, minimizer_kwargs=minimizer_kwargs)

#res = scipy.optimize.minimize(energy,c, method='nelder-mead',tol = 1e-10)
C = res.x
#print(res.x)
print(energy(res.x))
print("R is ",C[0])
a = C[int((smallN+1)):]
a.sort()
print("a is",a)
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