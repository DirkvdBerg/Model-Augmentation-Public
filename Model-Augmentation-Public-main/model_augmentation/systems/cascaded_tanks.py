from deepSI import System_deriv

import numpy as np

class Cascaded_Tanks(System_deriv):
    def __init__(self, k1, k2, k3, k4, dt=4, sigma_n=[0, 0, 0]):
        super(Cascaded_Tanks, self).__init__(nx=2, dt=dt)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.sigma_n = sigma_n

    def deriv(self,x,u):
        x1,x2 = x
        dx1 = -self.k1*np.sqrt(x1) + self.k4*u# + np.random.normal(0, self.sigma_n[0])
        dx2 = self.k2*np.sqrt(x1) - self.k3*np.sqrt(x2)# + np.random.normal(0, self.sigma_n[1])

        # dx1 = max(0,dx1)
        # dx2 = max(0,dx2)

        return [dx1,dx2]

    def h(self,x,u):
        return x[1]# + np.random.normal(0, self.sigma_n[2])
    
class Extended_Cascaded_Tanks(System_deriv):
    def __init__(self, k1, k2, k3, k4, k5, dt=4, sigma_n=[0, 0, 0]):
        super(Extended_Cascaded_Tanks, self).__init__(nx=2, dt=dt)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.sigma_n = sigma_n

    def deriv(self,x,u):
        x1,x2 = x

        dx1 = -self.k1*np.sqrt(x1) + self.k4*u

        if x1 <= 10:
            dx2 = self.k2*np.sqrt(x1) - self.k3*np.sqrt(x2)
        else:
            dx2 = self.k2*np.sqrt(x1) - self.k3*np.sqrt(x2) + self.k5*u

        return [dx1,dx2]

    def h(self,x,u):
        return x[1]# + np.random.normal(0, self.sigma_n[2])