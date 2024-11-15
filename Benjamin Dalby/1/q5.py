#Benjamin Dalby
#This code covers question 5. It includes a solver for a(t), in the standard advective equation form for an upwind 1st order scheme.
from scipy import sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

def a_t(U : np.ndarray, j :int, del_t: float, del_x: float, a_val : float, theta : float) -> float:
    #Our a(t) function. Use the max/min function to select which version of u_x we are using
    #depending on the "direction" of the "wind". Note we multiply by -1 because the upwind method
    #has U^j+1 = U^j - (rest of method) where our main method has U^j+1 = U^j + (rest of method) 
    #NB: here theta can represent either theta or 1-theta depending on the caller. In practice
    #we are always calling with 1-theta as theta only exists on the unknown side of the equation
    mu_x = del_t / del_x
    return  mu_x*theta*(max(a_val,0)*(U[j]-U[j-1])+min(a_val,0)*(U[j+1]-U[j]))

def theta_solve(U : np.ndarray, t: int, epsilon : float, a_val : float, del_t  : float, del_x : float, theta : float, snapshots = list) -> list[np.ndarray]:
    mu = del_t / (del_x**2)
    mu_x = del_t / del_x

    results = []
    J = U.shape[0] #length of array in space
    A = np.zeros(shape=(J,J), dtype=np.float64)
    b = np.zeros(shape=(J), dtype=np.float64)
    c = np.zeros_like(b)

    for j in range(1, J-1):
        #The paramerts are fixed, so A is constant
        #Set up A at the boundarys
        A[0,0] = 1
        A[-1,-1] = 1

        #General case
        A[j, j-1] = (-1*max(a_val, 0)*theta*mu_x) - (epsilon*theta*mu)
        A[j,j] = 1 + (2*epsilon*theta*mu) + (max(a_val, 0)*theta*mu_x) - ((min(a_val, 0))*theta*mu_x)
        A[j, j+1] = (max(a_val, 0)*theta*mu_x) - (epsilon*theta*mu)

        A = sp.csr_matrix(A)

    for n in range(t):
        for j in range(1, J-1):
            b[j] = U[j] + epsilon*mu*(1-theta)*(U[j+1] - 2*U[j] + U[j-1]) + a_t(U, j, del_t, del_x, a_val, (1-theta))

        x = sp.linalg.spsolve(A,b)

        if n+1 in snapshots:
            results.append(copy(x))

        U = copy(x)

    return results

def main():
    J= 20
    U = np.zeros(J)
    del_x = 0.001
    T = 100
    del_t = 0.0000001
    a_val = 1           #Advection strength set to valus of a(t)
    epsilon = 1         #Diffusion strength
    theta = 0.5

    U_1 = np.linspace(0,1,11)
    U_2 = np.linspace(1,0,11)
    U = np.hstack((U_1[0:-1],U_2))

    snapshots = [1, 25, 50, 75]
    r_head = copy(U)
    results = theta_solve(U, T, epsilon, a_val, del_t, del_x, theta, snapshots)
    results.insert(0,r_head)

    x = np.linspace(0,1,J+1)

    for r,s in list(zip(results, snapshots)):
        plt.plot(x,r,label=f"t={s}")
        plt.xlabel("x")
        plt.ylabel("U")
        plt.title(f"$\Delta$t={del_t}")
        plt.legend(loc="upper right")
    plt.show()

main()
