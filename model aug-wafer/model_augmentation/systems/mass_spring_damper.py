from deepSI import System_deriv

import numpy as np

class Mass_spring_damper(System_deriv):
    def __init__(self, k, c, m, a, dt=0.02, sigma_n=[0]):
        super(Mass_spring_damper, self).__init__(nx=2, dt=dt)
        self.k = k
        self.c = c
        self.m = m
        self.a = a
        self.sigma_n = sigma_n

    def deriv(self,x,u):
        z1,z2 = x
        dz1 = -self.k/self.m*z2 -self.a/self.m*np.power(z2,3) - self.c/self.m*z1 + 1/self.m*u
        dz2 = z1
        return [dz1,dz2]

    def h(self,x,u):
        return x[1] + np.random.normal(0, self.sigma_n[0])
    
class Msd_2dof(System_deriv):
    def __init__(self, m1, m2, k1, k2, c1, c2, a1, a2, dt=0.02):
        super(Msd_2dof, self).__init__(nx=4, dt=dt)
        self.m1 = m1; self.m2 = m2
        self.k1 = k1; self.k2 = k2
        self.c1 = c1; self.c2 = c2
        self.a1 = a1; self.a2 = a2

    def deriv(self,x,u):
        z1,z2,z3,z4 = x
        dz1 = z2
        dz2 = -(self.k1 + self.k2)/self.m1*z1 + self.k2/self.m1*z3 - (self.c1 + self.c2)/self.m1*z2 + self.c2/self.m1*z4 - self.a1/self.m1*np.power((z1),3) \
            + self.a2/self.m1*np.power((z3-z1), 3) + 1/self.m1*u[0]
        dz3 = z4
        dz4 = -self.k2/self.m2*(z3-z1) - self.c2/self.m2*(z4-z2) - self.a2/self.m2*np.power((z3-z1), 3) + 1/self.m2*u[1]
        return [dz1,dz2,dz3,dz4]

    def h(self,x,u):
        return x[0::2]
    
class Msd_ndof(System_deriv):
    def __init__(self, n, m, k, c, a, dt=0.02, output_ix=None, input_ix=None):
        super(Msd_ndof, self).__init__(nx=n*2, dt=dt)
        
        self.n = n
        self.output_ix = self.determine_mask(output_ix)
        self.input_ix = self.determine_mask(input_ix)
        
        self.m = self.cast_parameter_to_array(m, self.n)
        self.k = self.cast_parameter_to_array(k, self.n)
        self.c = self.cast_parameter_to_array(c, self.n)
        self.a = self.cast_parameter_to_array(a, self.n)

    def cast_parameter_to_array(self, p, n):
        if isinstance(p, (float, int)):
            array = np.repeat(p, n)
        else:
            array = p
        return array
    
    def determine_mask(self, ix):
        if ix == None:
            return np.arange(self.n)
        elif isinstance(ix, (np.ndarray, list)):
            return ix
        else:
            raise ValueError("Output_ix not valid value.")

    def deriv(self,x,u):
        u = self.expand_with_ix(u, self.input_ix, self.n)

        dx = np.zeros(self.nx)
        dx[0] = x[1]
        dx[1] = -(self.k[0] + self.k[1])/self.m[0]*x[0] + self.k[1]/self.m[0]*x[2] - (self.c[0] + self.c[1])/self.m[0]*x[1] + self.c[1]/self.m[0]*x[3] \
            - self.a[0]/self.m[0]*np.power((x[0]),3) + self.a[1]/self.m[0]*np.power((x[2]-x[0]), 3) + 1/self.m[0]*u[0]

        for i in range(1, self.n-1):
            d_i = x[2*i] - x[2*(i-1)]
            di_ = x[2*(i+1)] - x[2*i]

            der_d_i = x[2*i  + 1] - x[2*(i-1) + 1]
            der_di_ = x[2*(i+1) + 1] - x[2*i + 1]

            F_i = self.k[i]*d_i + self.c[i]*der_d_i + self.a[i]*np.power(d_i, 3)
            Fi_ = self.k[i+1]*di_ + self.c[i+1]*der_di_ + self.a[i+1]*np.power(di_, 3)

            dx[2*i] = x[2*i + 1]
            dx[2*i+1] = (-F_i + Fi_ + u[i])/self.m[i]

        xn = x[-2]; dxn = x[-1]; xn_ = x[-4]; dxn_ = x[-3]
        dx[-2] = dxn
        dx[-1] = -self.k[self.n-1]/self.m[self.n-1]*(xn-xn_) - self.c[self.n-1]/self.m[self.n-1]*(dxn-dxn_) \
            - self.a[self.n-1]/self.m[self.n-1]*np.power((xn-xn_), 3) + 1/self.m[-1]*u[-1]
        
        return dx

    def h(self,x,u):
        y = x[0::2]
        y = y[self.output_ix]
        return y
    
    def expand_with_ix(self, p, ix, n):
        q = np.zeros(n)
        q[ix] = p
        return q
    
class Msd_ndof_lpf_output(System_deriv):
    def __init__(self, n, m, k, c, a, dt=0.02, output_ix=None, input_ix=None):
        super(Msd_ndof_lpf_output, self).__init__(nx=n*2, dt=dt)
        
        self.n = n
        self.output_ix = self.determine_mask(output_ix)
        self.input_ix = self.determine_mask(input_ix)
        
        self.m = self.cast_parameter_to_array(m, self.n)
        self.k = self.cast_parameter_to_array(k, self.n)
        self.c = self.cast_parameter_to_array(c, self.n)
        self.a = self.cast_parameter_to_array(a, self.n)

        self.x_lpf = None

    def cast_parameter_to_array(self, p, n):
        if isinstance(p, (float, int)):
            array = np.repeat(p, n)
        else:
            array = p
        return array
    
    def determine_mask(self, ix):
        if ix == None:
            return np.arange(self.n)
        elif isinstance(ix, (np.ndarray, list)):
            return ix
        else:
            raise ValueError("Output_ix not valid value.")

    def deriv(self,x,u):
        u = self.expand_with_ix(u, self.input_ix, self.n)

        dx = np.zeros(self.nx)
        dx[0] = x[1]
        dx[1] = -(self.k[0] + self.k[1])/self.m[0]*x[0] + self.k[1]/self.m[0]*x[2] - (self.c[0] + self.c[1])/self.m[0]*x[1] + self.c[1]/self.m[0]*x[3] \
            - self.a[0]/self.m[0]*np.power((x[0]),3) + self.a[1]/self.m[0]*np.power((x[2]-x[0]), 3) + 1/self.m[0]*u[0]

        for i in range(1, self.n-1):
            d_i = x[2*i] - x[2*(i-1)]
            di_ = x[2*(i+1)] - x[2*i]

            der_d_i = x[2*i  + 1] - x[2*(i-1) + 1]
            der_di_ = x[2*(i+1) + 1] - x[2*i + 1]

            F_i = self.k[i]*d_i + self.c[i]*der_d_i + self.a[i]*np.power(d_i, 3)
            Fi_ = self.k[i+1]*di_ + self.c[i+1]*der_di_ + self.a[i+1]*np.power(di_, 3)

            dx[2*i] = x[2*i + 1]
            dx[2*i+1] = (-F_i + Fi_ + u[i])/self.m[i]

        xn = x[-2]; dxn = x[-1]; xn_ = x[-4]; dxn_ = x[-3]
        dx[-2] = dxn
        dx[-1] = -self.k[self.n-1]/self.m[self.n-1]*(xn-xn_) - self.c[self.n-1]/self.m[self.n-1]*(dxn-dxn_) \
            - self.a[self.n-1]/self.m[self.n-1]*np.power((xn-xn_), 3) + 1/self.m[-1]*u[-1]
        
        return dx

    def h(self,x,u):
        y = x[0::2]
        y = y[self.output_ix]

        if self.x_lpf is None:
            self.x_lpf = y
            y_lpf = y
        else:
            y_lpf = 6.25*self.x_lpf
            self.x_lpf = 0.3679*self.x_lpf + 0.1011*y 

        return y_lpf
    
    def expand_with_ix(self, p, ix, n):
        q = np.zeros(n)
        q[ix] = p
        return q

class Msd_ndof_input_saturation(System_deriv):
    def __init__(self, n, m, k, c, a, dt=0.02, output_ix=None, input_ix=None):
        super(Msd_ndof_input_saturation, self).__init__(nx=n*2, dt=dt)
        
        self.n = n
        self.output_ix = self.determine_mask(output_ix)
        self.input_ix = self.determine_mask(input_ix)
        
        self.m = self.cast_parameter_to_array(m, self.n)
        self.k = self.cast_parameter_to_array(k, self.n)
        self.c = self.cast_parameter_to_array(c, self.n)
        self.a = self.cast_parameter_to_array(a, self.n)

    def cast_parameter_to_array(self, p, n):
        if isinstance(p, (float, int)):
            array = np.repeat(p, n)
        else:
            array = p
        return array
    
    def determine_mask(self, ix):
        if ix == None:
            return np.arange(self.n)
        elif isinstance(ix, (np.ndarray, list)):
            return ix
        else:
            raise ValueError("Output_ix not valid value.")

    def deriv(self,x,u):
        
        u = self.expand_with_ix(u, self.input_ix, self.n)
        u = 30*np.tanh(u/30)

        dx = np.zeros(self.nx)
        dx[0] = x[1]
        dx[1] = -(self.k[0] + self.k[1])/self.m[0]*x[0] + self.k[1]/self.m[0]*x[2] - (self.c[0] + self.c[1])/self.m[0]*x[1] + self.c[1]/self.m[0]*x[3] \
            - self.a[0]/self.m[0]*np.power((x[0]),3) + self.a[1]/self.m[0]*np.power((x[2]-x[0]), 3) + 1/self.m[0]*u[0]

        for i in range(1, self.n-1):
            d_i = x[2*i] - x[2*(i-1)]
            di_ = x[2*(i+1)] - x[2*i]

            der_d_i = x[2*i  + 1] - x[2*(i-1) + 1]
            der_di_ = x[2*(i+1) + 1] - x[2*i + 1]

            F_i = self.k[i]*d_i + self.c[i]*der_d_i + self.a[i]*np.power(d_i, 3)
            Fi_ = self.k[i+1]*di_ + self.c[i+1]*der_di_ + self.a[i+1]*np.power(di_, 3)

            dx[2*i] = x[2*i + 1]
            dx[2*i+1] = (-F_i + Fi_ + u[i])/self.m[i]

        xn = x[-2]; dxn = x[-1]; xn_ = x[-4]; dxn_ = x[-3]
        dx[-2] = dxn
        dx[-1] = -self.k[self.n-1]/self.m[self.n-1]*(xn-xn_) - self.c[self.n-1]/self.m[self.n-1]*(dxn-dxn_) \
            - self.a[self.n-1]/self.m[self.n-1]*np.power((xn-xn_), 3) + 1/self.m[-1]*u[-1]
        
        return dx

    def h(self,x,u):
        y = x[0::2]
        y = y[self.output_ix]
        return y
    
    def expand_with_ix(self, p, ix, n):
        q = np.zeros(n)
        q[ix] = p
        return q