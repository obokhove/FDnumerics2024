#Benjamin Dalby
from scipy import sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

def get_legendre(k, J):
    l = 0
    b_k = []
    for i in range(1,k+1):
        l_t, b_k_t = gen_legendre(i, J)
        l += l_t
        b_k.append(b_k_t)

    return l, b_k

def gen_legendre(k, J):
    '''Get the term L(x)b_k. Generates the legendre polynomail over {-1,1} and then shifts the range of L(x)b_k to > 0'''
    p = []
    for i in range(1,k+1):
        p.append(i)
    l = np.polynomial.legendre.Legendre(p, window=np.asarray([0,1])).linspace(J, domain=[-1,1])
    l = l[1]
    r = np.random.uniform(size=1)

    print(f'bk={r}')

    out = l*r
    C = np.min(out)
    if C < 0:
        out += abs(C)

    return out, r

def a_t(U : np.ndarray, j :int, del_t: float, del_x: float, a_val : float, theta : float) -> float:
    #Our a(t) function. Use the max/min function to select which version of u_x we are using
    #depending on the "direction" of the "wind". Note we multiply by -1 because the upwind method
    #has U^j+1 = U^j - (rest of method) where our main method has U^j+1 = U^j + (rest of method) 
    #NB: here theta can represent either theta or 1-theta depending on the caller. In practice
    #we are always calling with 1-theta as theta only exists on the unknown side of the equation
    mu_x = del_t / del_x
    return  mu_x*theta*(max(a_val,0)*(U[j]-U[j-1])+min(a_val,0)*(U[j+1]-U[j]))

def theta_solve(U : np.ndarray, t: int | float, epsilon : float, a_val : float, del_t  : float, del_x : float, theta : float, snapshots = list) -> list[np.ndarray]:
    #t has been calculated so tends to be a float, convert for loop control
    try:
        t = int(round(t))
    except Exception as e:
        raise e

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

        if (n+1)*del_t in snapshots:
            results.append(copy(x))

        U = copy(x)

    return results

def main():
    J = 40
    del_x = 0.001
    del_t = 0.0000005
    a_val = 1           #Advection strength set to values of a(t)
    epsilon = 0.001         #Diffusion strength
    theta = 1.0
    target_time = 1     #length of time to run for in "real time", seconds
    use_legendre = True

    T = target_time / del_t
    print(f'T= {T}')
    print(f'mu = {del_t/(del_x**2)}')

    x = np.linspace(-1,1,J)
    if use_legendre:
        l, b_k = get_legendre(3, J)
        U = ((1-x)**4)*(1+x)*l
    else:
        U = ((1-x)**4)*(1+x)

    snapshots = [0, 0.00025, 0.001, 0.0025, 0.005, 0.01, 0.1]
    r_head = copy(U)
    results = theta_solve(U, T, epsilon, a_val, del_t, del_x, theta, snapshots)
    results.insert(0,r_head)


    for r,s in list(zip(results, snapshots)):
        plt.plot(x,r,label=f"t={s} s")
        plt.ylabel("U")
        plt.title(f"$\mu$={round(del_t/(del_x**2),2)}, $\Delta x$ = {del_x}")
        plt.legend(loc="upper right")
        if use_legendre:
            plt.xlabel(f"x \n b_k={b_k}")
            plt.tight_layout()
        else:
            plt.xlabel("x")
    plt.show()

main()