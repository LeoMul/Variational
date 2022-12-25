import scipy as sci 
import numpy as np 

def psi_n(x,n):
    return (x**n) * (x-1.0) * (x+1.0)

def calc_s_mn(x,n,m,h):
    integral = sci.integrate.quad(lambda x: psi_n(x,n)*psi_n(x,m),-1,1)
    return integral[0]

def calc_H_mn(x,n,m,h):
    
    integral = sci.integrate.quad(lambda x: -psi_n(x,n)*find_second_derivative(x,m,h),-1,1)
    return integral[0]

def construct_s_matrix(x,h,N):
    s = np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            s[i,j] = calc_s_mn(x,i,j,h)

    return s 
def construct_H_matrix(x,h,N):
    s = np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            s[i,j] = calc_H_mn(x,i,j,h)

    return s 
def find_second_derivative(x,m,h):

    def wraps(x):
        return psi_n(x,m)
    return sci.misc.derivative(wraps,x,dx = h,n = 2)


N = 16
x,h = np.linspace(-1,1,10000,retstep=True)
H = construct_H_matrix(x,h,N)
S = construct_s_matrix(x,h,N)
eig = sci.linalg.eig(H,S)[0]
print(np.sort(eig))
