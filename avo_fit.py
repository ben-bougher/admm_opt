from simplex_projection import euclidean_proj_l1ball
from numpy.linalg import inv, solve

from matplotlib import pyplot as plt
import numpy as np

from numpy import linalg as LA

def Z_update(beta, Y, delta, X, Tk, Pk):
    
    # Go col by col
    Z = np.zeros(Tk.shape)

    # store the inverse
    A_inv = inv(beta * Y.T.dot(Y) + (delta * np.eye(Y.shape[1])))
    b = (beta * Y.T.dot(X)) + (delta * Tk - Pk)
    Z = A_inv.dot(b)
    return Z

def T_update(Z_kp1, Pk, Rw, delta, sigma, zeta):
    
    T1 = Z_kp1 +(Pk / delta) - (Rw.dot(sigma) / delta) 
    T2 = conv_proj(T1, delta, zeta)
    
    return T1 - T2

def P_update(Pk, delta, Z_kp1, T_kp1):
    
    return Pk + delta * (Z_kp1 - T_kp1)

def conv_proj(A, delta, zeta):
    
    # Copy the input
    A_out = np.array(A, copy=True)
    
    # Go row by row through A
    for i in range(A_out.shape[0]):
        
        a = A_out[i,:]
        
        pos_ind = a > 0
        b = a[pos_ind]
        proj_b = euclidean_proj_l1ball(b, zeta / delta)
        
        np.put(a, np.where(pos_ind), proj_b)

    return A_out        

def hilbert_prod(A,B):
    
    return np.trace(A.dot(B.T))


def minimizer(X, Y, beta=250.0, delta=1.0, zeta=1.0, 
              h=(1-np.cos(4.0 * np.pi/180.0)), v=50.0,
              tol=0.2, max_iter=500):
    
    Tk = np.ones(Y.shape[1])
    Pk = np.random.randn(*Tk.shape)
    
    Rw = np.eye(Tk.shape[0])
    
    sigma = v*(1-np.exp(-(1 / (2 * h**2))*(1-(Y.T.dot(X)))**2))
    
    def L_del(Z,T,P):
        
        if(np.all(T >= 0)):
            g = 0.0
        else:
            g = 1000000000.0
            print 'should not be here'
        
       
        l1_inf = zeta * np.sum(np.amax(T,1))
        l1_ = hilbert_prod(Rw.dot(sigma), T)
      
        fidelity = (beta / 2.0) * hilbert_prod(Y.dot(Z) - X, Y.dot(Z) - X)
        
        lagrange = hilbert_prod(P, Z-T) + (delta/2.0)*hilbert_prod(Z-T, Z-T)
        
        return l1_inf + l1_ + fidelity + lagrange
        
    n=0
    converged = 1000000000.0
    objective = []
    while (n < max_iter) and (converged > tol):
        
        Z_kp1 = Z_update(beta, Y, delta, X, Tk, Pk)
        
        T_kp1 = T_update(Z_kp1, Pk, Rw, delta, sigma, zeta)
        
        
        P_kp1 = P_update(Pk, delta, Z_kp1, T_kp1)

        L = L_del(Z_kp1, T_kp1, P_kp1)
        Tk = T_kp1
        Pk = P_kp1
        
        if n == 0:
            L_old = L
            n = n+1
            continue
        converged = np.sum((L - L_old)**2)
        n = n+1
        objective.append(converged)
        L_old = L

    return Tk, objective
