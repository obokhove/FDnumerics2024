#Benjamin Dalby
import numpy as np
import matplotlib.pyplot as plt
import copy

def a_t(U : np.ndarray, j :int, del_t: float, del_x: float, a_val : float) -> float:
    #Our a(t) function. Use the max/min function to select which version of u_x we are using
    #depending on the "direction" of the "wind". Note we multiply by -1 because the upwind method
    #has U^j+1 = U^j - (rest of method) where our main method has U^j+1 = U^j + (rest of method) 
    mu_x = del_t/del_x
    return  -1*(mu_x*(max(a_val,0)*(U[j]-U[j-1])+min(a_val,0)*(U[j+1]-U[j])))

def explicit(U : np.ndarray, t: int, epsilon : float, a_val : float, del_t  : float, del_x : float, snapshots = list) -> list[np.ndarray]:
    mu = del_t / (del_x**2)
    U_p = np.zeros(shape=U.shape[0])
    results = []
    J = U.shape[0]      #length of the array in space

    for n in range(t):
        for j in range(1, J-1):
            U_p[j] = U[j] + epsilon*mu*(U[j+1] - 2*U[j] + U[j-1]) + a_t(U, j, del_t, del_x, a_val)

        if n+1 in snapshots:
            results.append(copy.copy(U_p))

        U = copy.copy(U_p)

    return results

def main():
    J= 20
    U = np.zeros(J)
    del_x = 1/J         #0.05 for J =20
    T = 50
    del_t = 0.0012
    a_val = -1           #Advection strength set to valus of a(t)
    epsilon = 1         #Diffusion strength

    U_1 = np.linspace(0,1,11)
    U_2 = np.linspace(1,0,11)
    U = np.hstack((U_1[0:-1],U_2))


    snapshots = [1, 25, 50]
    r_head = copy.copy(U)
    results = explicit(U, T, epsilon, a_val, del_t, del_x, snapshots)
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
