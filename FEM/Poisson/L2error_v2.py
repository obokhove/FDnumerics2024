import numpy as np
import matplotlib.pyplot as plt

dx=[1/8,1/16,1/32,1/64,1/128]
# Method 1:
L2_1=[0.006213900940245384,
	  0.001593010773101212,
	  0.00040075733646107597,
	  0.0001003464039782477,
	  2.5096426072489425e-05]
# Method 2:
L2_2=[0.006213900940245383,
	  0.0015930107731022762,
	  0.0004007573364659739,
	  0.00010034640400416159,
	  2.509642615825676e-05]
ref=[ele**2 for ele in dx]

plt.figure(num=1)

plt.loglog(dx,L2_1,'^r-',label='Method 1')
plt.loglog(dx,L2_2,'vb--',label='Method 2')
plt.loglog(dx,ref,'g:',label='$(\Delta x)^2$')
plt.grid()
plt.xlabel('$\Delta x$',fontsize=14)
plt.ylabel('$L^2$ error',fontsize=14)
plt.legend(prop={'size':14})
plt.tight_layout()

plt.show()