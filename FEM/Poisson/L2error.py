import numpy as np
import matplotlib.pyplot as plt

dx=[1/8,1/16,1/32,1/64,1/128]
L2=[0.006213900940245384,0.001593010773101212,0.00040075733646107597,0.0001003464039782477,2.5096426072489425e-05]
ref=[ele**2 for ele in dx]

plt.figure(num=1)

plt.loglog(dx,L2,label='$u$', marker='o', markersize=7, color='red')
plt.loglog(dx,ref,'b--')
plt.grid()
plt.xlabel('$\Delta x$',fontsize=14)
plt.ylabel('$L^2$ error',fontsize=14)
plt.legend(prop={'size':14})
plt.tight_layout()

plt.show()