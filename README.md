# FDnumerics2024

## GitHub instructions
Please make a folder with your name and therein a readme.md : e.g. <yourname>/readme.md
Then add a subfolder for each exercise. Each subfolder should have clear simulation and read-through instructions in its readme.md

One creates a folder by making a file "onnob/readme" say (so "onnob" before the "/" takes care that readme.md is stored in the folder "onnob").

## Finite-difference exercise

  ```
    import matplotlib.pyplot as plt
    import time
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    from scipy.sparse import diags
    ...
    num = int(1)  # highest degree of Legendre
    DATA = int(5)
    a = (np.random.rand(num,1,DATA)) #coefficients for basis functions
    ...
    Lege[0,:] = 1+0*xj
    Lege[1,:] = xj
    Lege[2,:] = (3/2)*xj**2-1/2
    Lege[3,:] = (5/2)*xj**3-(3/2)*xj
    ... 
    lamb = 0.0*xj # u
    lambn = 0.0*lamb # u
    
    #
    # Initial condition
    #
    lamb = Amp*np.exp(-alphaa*(xj-0.5*L)**2)
    lamb = (1-xj)**4*(1+xj)    #lamb = lamb*(a[0,0,0]*Bern[0,:]+a[0,0,1]*Bern[1,:]+a[0,0,2]*Bern[2,:]+a[0,0,3]*Bern[3,:])
    lambn = (a[0,0,0]*Lege[0,:]+a[0,0,1]*Lege[1,:]+a[0,0,2]*Lege[2,:]+a[0,0,3]*Lege[3,:])
    lambn = lambn-min(lambn)
    lamb = lamb*lambn
  ```
