"""
© 2021 This work is licensed under a CC-BY-NC-SA license.
Title:
**Authors:** Cristiano Capone
"""

import numpy as np
import utils as ut
from tqdm import trange

#import _pickle as cPickle
import traceback


def sigm ( x, dv ):

	if dv < 1 / 30:
		return x > 0;
	y = x / dv;
    #y = x / 10.;

	out = 1.5*(1. / (1. + np.exp (-y*3. )) - .5);

	return out;

def gaussian(x, mu, sig):
	return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

class GOAL:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        #self.itau_m = self.dt / par['tau_m'];
        self.tau_m = np.logspace( np.log(par['tau_m_f']) ,np.log(par['tau_m_s']),self.N)
        self.itau_m = np.zeros(self.N,)#self.dt / par['tau_m'];
        self.itau_m[:int(self.N/2)] = self.dt / par['tau_m_f']
        self.itau_m[int(self.N/2):] = self.dt / par['tau_m_s']
        
        #self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        
        self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro']);

        self.dv = par['dv'];

        
        
        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec'], size = (self.N, self.N));#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I));
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O));
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N));#np.zeros ((self.O, self.N));
        self.JoutV = np.random.normal (0.0, par['sigma_output'], size = (1,self.N));#np.zeros ((self.O, self.N));
        self.Jsigmaout = np.random.normal (0.2, 0.5, size = (self.O,self.N))#np.zeros ((self.O, self.N));

        print('hello')

        self.dJfilt = np.zeros(np.shape(self.J))
        self.dJfilt_out = np.zeros(np.shape(self.Jout))
        self.dJfilt_sigma_out = np.zeros(np.shape(self.Jsigmaout))

        self.dJoutV_filt = np.zeros(np.shape(self.JoutV))
        self.dJoutV_aggregate = np.zeros(np.shape(self.JoutV))

        self.dJout_aggregate = np.zeros(np.shape(self.Jout))
        self.dJout_sigma_aggregate = np.zeros(np.shape(self.Jsigmaout))
        self.dJ_aggregate = np.zeros(np.shape(self.J))

        self.value=0
        self.r=0
        self.r_old=0
        
        self.y = np.zeros(self.O,)

        # Remove self-connections
        np.fill_diagonal (self.J, 0.)

        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo'];
        
        self.Vda = np.zeros(self.N,)

        self.Vo = par['Vo'];

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N);
        self.S_hat = np.zeros (self.N);
        self.S_ro = np.zeros (self.N);
        self.dH = np.zeros (self.N);

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N)
        self.state_out_p = np.zeros (self.N)

        # Here we save the params dictionary
        self.par = par
        

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv

        out = np.zeros (x.shape)
        mask = x > 0
        out [mask] = 1. / (1. + np.exp (-y [mask]))
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]))
        out = (out+1.)

        return out

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv))


    def step_rate (self, inp, t, probabilistic = False):
        self.alpha_m = np.exp(-self.dt/self.tau_m)
        itau_s = self.itau_s
        itau_ro = self.itau_ro
        #itau_m= np.exp (-self.dt / par['tau_s'])

        self.S_hat   = (self.S_hat   * itau_s + self.S   * (1. - itau_s))
        self.S_ro   = (self.S_ro   * itau_ro + self.S   * (1. - itau_ro))
        self.dH   = self.dH  *self.alpha_m  +  (1. - self.alpha_m) * self.S_hat
        self.mu = self.J @ self.S_hat   + self.Jin @ inp + self.h
        self.H   = self.H   * self.alpha_m  + (1. - self.alpha_m)* (self.J @ self.S_hat   + self.Jin @ inp + self.h)#\
                                                          #+ self.Jreset @ self.S  ;
        self.Vda = self.Vda  * (self.alpha_m) + (self.H - self.mu)
        #

        self.S   = self._sigm (self.H  , dv = self.dv)
        self.y = self.Jout@ self.S_ro#np.tanh()

        return self.H


    def reset (self, init = None):
        self.S   = init if init else np.zeros (self.N)
        self.S_hat   = self.S   * self.itau_s if init else np.zeros (self.N)
        self.S_ro   = self.S   * self.itau_s if init else np.zeros (self.N)
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  *= 0.
        self.H  += self.Vo
        self.y = 0
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1
            self.I_clock = I_clock

class GOAL_SIMPLE:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        #self.itau_m = self.dt / par['tau_m'];
        self.tau_m = np.logspace( np.log(par['tau_m_f']) ,np.log(par['tau_m_s']),self.N)
        self.itau_m = np.zeros(self.N,)#self.dt / par['tau_m'];
        self.itau_m[:int(self.N/2)] = self.dt / par['tau_m_f']
        self.itau_m[int(self.N/2):] = self.dt / par['tau_m_s']
        
        #self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        
        self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro']);

        self.dv = par['dv'];

    
        
        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec'], size = (self.N, self.N));#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I));
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O));
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N));#np.zeros ((self.O, self.N));
    
        self.y = np.zeros(self.O,)
        np.fill_diagonal (self.J, 0.);

        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo'];
        
        self.Vda = np.zeros(self.N,)

        self.Vo = par['Vo'];

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N);
        self.S_hat = np.zeros (self.N);
        self.S_ro = np.zeros (self.N);
        self.dH = np.zeros (self.N);
        self.inp_filt= np.zeros (self.I);

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N)
        self.state_out_p = np.zeros (self.N)

        # Here we save the params dictionary
        self.par = par
        

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv

        out = np.zeros (x.shape)
        mask = x > 0
        out [mask] = 1. / (1. + np.exp (-y [mask]))
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]))

        return out;

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv))


    def step_rate (self, inp, t, probabilistic = False):
        self.alpha_m = np.exp(-self.dt/self.tau_m)
        itau_s = self.itau_s
        itau_ro = self.itau_ro
        #itau_m= np.exp (-self.dt / par['tau_s'])

        self.S_hat   = (self.S_hat   * itau_s + self.S   * (1. - itau_s))
        self.S_ro   = (self.S_ro   * itau_ro + self.S   * (1. - itau_ro))
        self.dH   = self.dH  *self.alpha_m  +  (1. - self.alpha_m) * self.S_hat
        self.mu = self.J @ self.S_hat   + self.Jin @ inp + self.h
        self.H   = self.H   * self.alpha_m  + (1. - self.alpha_m)* (self.J @ self.S_hat   + self.Jin @ inp + self.h)#\
                                                          #+ self.Jreset @ self.S  ;
        self.Vda = self.Vda  * (self.alpha_m) + (self.H - self.mu)
        #

        self.S   = self._sigm (self.H  , dv = self.dv)
        self.y = self.Jout@ self.S_ro#np.tanh(self.Jout@ self.S_ro)

        return self.H


    def reset (self, init = None):
        self.S   = init if init else np.zeros (self.N)
        self.S_hat   = self.S   * self.itau_s if init else np.zeros (self.N)
        self.S_ro   = self.S   * self.itau_s if init else np.zeros (self.N)
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  *= 0.
        self.H  += self.Vo
        self.y = 0
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1
            self.I_clock = I_clock
            
class GOAL_TAKENS:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape']
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        #self.itau_m = self.dt / par['tau_m'];
        self.tau_m = np.logspace( np.log(par['tau_m_f']) ,np.log(par['tau_m_s']),self.N)
        self.itau_m = np.zeros(self.N,)#self.dt / par['tau_m'];
        self.itau_m[:int(self.N/2)] = self.dt / par['tau_m_f']
        self.itau_m[int(self.N/2):] = self.dt / par['tau_m_s']
        
        #self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        
        #self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro'])

        self.dv = par['dv']
        
        self.n_tau = 10
        #self.itau_s
        
        #self.tau_s = np.logspace(-3,0,self.n_tau)#np.array([1.,10.])*0.001
        self.tau_s = np.linspace(1.,1000.,self.n_tau)*0.001#np.array([1.,10.])*0.001
        
        self.itau_s = np.copy(np.exp ( -self.dt /  self.tau_s ) )
        
        print(self.itau_s)
        
        #### definisci bene sti tau
        
        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec']*0., size = (self.N, self.N, self.n_tau) )#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I))
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O))
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N))#np.zeros ((self.O, self.N));
        self.JoutV = np.random.normal (0.0, par['sigma_output'], size = (1,self.N))#np.zeros ((self.O, self.N));
        self.Jsigmaout = np.random.normal (0.2, 0.5, size = (self.O,self.N))#np.zeros ((self.O, self.N));
        
        self.bias_ro = 0.

        print('hello')

        self.dJfilt = np.zeros(np.shape(self.J))
        self.dJfilt_out = np.zeros(np.shape(self.Jout))
        self.dJfilt_sigma_out = np.zeros(np.shape(self.Jsigmaout))

        self.dJoutV_filt = np.zeros(np.shape(self.JoutV))
        self.dJoutV_aggregate = np.zeros(np.shape(self.JoutV))

        self.dJout_aggregate = np.zeros(np.shape(self.Jout))
        self.dJout_sigma_aggregate = np.zeros(np.shape(self.Jsigmaout))
        self.dJ_aggregate = np.zeros(np.shape(self.J))

        self.value=0
        self.r=0
        self.r_old=0
        
        self.y = 0

        # Remove self-connections
        for k in range(self.n_tau):
            np.fill_diagonal (np.reshape( self.J[:,:,k] , ( self.N,self.N )) , 0.);

        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo']
        
        self.Vda = np.zeros(self.N,)

        self.Vo = par['Vo']

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N)
        self.S_hat = np.zeros ((self.N,self.n_tau))
        self.S_ro = np.zeros (self.N)
        self.dH = np.zeros ((self.N,self.n_tau))
        self.inp_filt= np.zeros (self.I)

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N);
        self.state_out_p = np.zeros (self.N);

        # Here we save the params dictionary
        self.par = par;

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));

        return out;

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv));


    def step_rate (self, inp, t, probabilistic = False):
        self.alpha_m = np.exp(-self.dt/self.tau_m);
        #itau_s = self.itau_s;
        itau_ro = self.itau_ro;
        #itau_m= np.exp (-self.dt / par['tau_s'])
        
        for k in range(self.n_tau):
            
            itau_s = self.itau_s[k]
            #print(np.shape(self.S_hat))
            self.S_hat[:,k]   = (self.S_hat[:,k]   * itau_s + self.S   * (1. - itau_s))
        
        self.S_ro   = (self.S_ro   * itau_ro + self.S   * (1. - itau_ro))
        
        for k in range(self.n_tau):
            self.dH[:,k]   = self.dH[:,k]  *self.alpha_m  +  (1. - self.alpha_m) * self.S_hat[:,k]

        alpha_inp = self.dt/self.tau_inp
        
        self.inp_filt = self.inp_filt * (1.- alpha_inp) + inp#*alpha_inp
        
        self.mu = self.Jin @ self.inp_filt + self.h
        
        
        for k in range(self.n_tau):
            self.mu += self.J[:,:,k] @ self.S_hat[:,k]
        
        self.H   = self.H   * self.alpha_m  + (1. - self.alpha_m)* self.mu
        
        self.Vda = self.Vda  * (self.alpha_m) + (self.H - self.mu)
        #

        self.S   = self._sigm (self.H  , dv = self.dv)
        self.y = self.Jout@ self.S_ro+self.bias_ro#np.tanh(self.Jout@ self.S_ro)

        return self.H


    def reset (self, init = None):
        
        self.S   = init if init else np.zeros (self.N)
        self.S_hat  = np.copy(self.S_hat*0.)
        
        self.S_ro   = np.copy(self.S*0.)
                                                                    
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  *= 0.;
        self.H  += self.Vo
        self.y = 0
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1;
            self.I_clock = I_clock

class GOAL_TAKENS_LINEAR:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape']
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        #self.itau_m = self.dt / par['tau_m'];
        self.tau_m = np.logspace( np.log(par['tau_m_f']) ,np.log(par['tau_m_s']),self.N)
        self.itau_m = np.zeros(self.N,)#self.dt / par['tau_m'];
        self.itau_m[:int(self.N/2)] = self.dt / par['tau_m_f']
        self.itau_m[int(self.N/2):] = self.dt / par['tau_m_s']
        
        #self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        
        #self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro'])

        self.dv = par['dv']
        
        self.n_tau = 10
        #self.itau_s
        
        #self.tau_s = np.logspace(-3,0,self.n_tau)#np.array([1.,10.])*0.001
        self.tau_s = np.linspace(1.,1000.,self.n_tau)*0.001#np.array([1.,10.])*0.001
        
        self.itau_s = np.copy(np.exp ( -self.dt /  self.tau_s ) )
        
        print(self.itau_s)
        
        #### definisci bene sti tau
        
        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec']*0., size = (self.N, self.N, self.n_tau) )#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I))
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O))
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N))#np.zeros ((self.O, self.N));
        self.JoutV = np.random.normal (0.0, par['sigma_output'], size = (1,self.N))#np.zeros ((self.O, self.N));
        self.Jsigmaout = np.random.normal (0.2, 0.5, size = (self.O,self.N))#np.zeros ((self.O, self.N));
        
        self.bias_ro = 0.

        print('hello')

        self.dJfilt = np.zeros(np.shape(self.J))
        self.dJfilt_out = np.zeros(np.shape(self.Jout))
        self.dJfilt_sigma_out = np.zeros(np.shape(self.Jsigmaout))

        self.dJoutV_filt = np.zeros(np.shape(self.JoutV))
        self.dJoutV_aggregate = np.zeros(np.shape(self.JoutV))

        self.dJout_aggregate = np.zeros(np.shape(self.Jout))
        self.dJout_sigma_aggregate = np.zeros(np.shape(self.Jsigmaout))
        self.dJ_aggregate = np.zeros(np.shape(self.J))

        self.value=0
        self.r=0
        self.r_old=0
        
        self.y = 0

        # Remove self-connections
        for k in range(self.n_tau):
            np.fill_diagonal (np.reshape( self.J[:,:,k] , ( self.N,self.N )) , 0.);

        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo']
        
        self.Vda = np.zeros(self.N,)

        self.Vo = par['Vo']

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N)
        self.S_hat = np.zeros ((self.N,self.n_tau))
        self.S_ro = np.zeros (self.N)
        self.dH = np.zeros ((self.N,self.n_tau))
        self.inp_filt= np.zeros (self.I)

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N);
        self.state_out_p = np.zeros (self.N);

        # Here we save the params dictionary
        self.par = par;

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));

        return out;

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv));


    def step_rate (self, inp, t, probabilistic = False):
        self.alpha_m = np.exp(-self.dt/self.tau_m);
        #itau_s = self.itau_s;
        itau_ro = self.itau_ro;
        #itau_m= np.exp (-self.dt / par['tau_s'])
        
        for k in range(self.n_tau):
            
            itau_s = self.itau_s[k]
            #print(np.shape(self.S_hat))
            self.S_hat[:,k]   = (self.S_hat[:,k]   * itau_s + self.S   * (1. - itau_s))
        
        self.S_ro   = (self.S_ro   * itau_ro + self.S   * (1. - itau_ro))
        
        for k in range(self.n_tau):
            self.dH[:,k]   = self.dH[:,k]  *self.alpha_m  +  (1. - self.alpha_m) * self.S_hat[:,k]

        alpha_inp = self.dt/self.tau_inp
        
        self.inp_filt = self.inp_filt * (1.- alpha_inp) + inp#*alpha_inp
        
        self.mu = self.Jin @ self.inp_filt + self.h
        
        
        for k in range(self.n_tau):
            self.mu += self.J[:,:,k] @ self.S_hat[:,k]
        
        self.H   = self.H   * self.alpha_m  + (1. - self.alpha_m)* self.mu
        
        self.Vda = self.Vda  * (self.alpha_m) + (self.H - self.mu)
        #

        self.S  = np.copy(self.H )#self._sigm (self.H  , dv = self.dv)
        self.y = self.Jout@ self.S_ro+self.bias_ro#np.tanh(self.Jout@ self.S_ro)

        return self.H


    def reset (self, init = None):
        
        self.S   = init if init else np.zeros (self.N)
        self.S_hat  = np.copy(self.S_hat*0.)
        
        self.S_ro   = np.copy(self.S*0.)
                                                                    
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  *= 0.;
        self.H  += self.Vo
        self.y = 0
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1;
            self.I_clock = I_clock

class RESERVOIRE:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        self.itau_m = np.zeros(self.N,)#self.dt / par['tau_m'];
        #self.itau_m[:int(self.N/2)] = self.dt / par['tau_m_f']
        #self.itau_m[int(self.N/2):] = self.dt / par['tau_m_s']
        #self.itau_m = np.linspace(self.dt / par['tau_m_f'],self.dt / par['tau_m_s'],self.N)
        self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro']);

        self.dv = par['dv']

        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec'], size = (self.N, self.N));#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I));
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O));
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N));#np.zeros ((self.O, self.N));
        self.JoutV = np.random.normal (0.0, par['sigma_output'], size = (1,self.N));#np.zeros ((self.O, self.N));
        self.Jsigmaout = np.random.normal (0.2, 0.5, size = (self.O,self.N))#np.zeros ((self.O, self.N));

        print('hello')

        self.dJfilt = np.zeros(np.shape(self.J))
        self.dJfilt_out = np.zeros(np.shape(self.Jout))
        self.dJfilt_sigma_out = np.zeros(np.shape(self.Jsigmaout))

        self.dJoutV_filt = np.zeros(np.shape(self.JoutV))
        self.dJoutV_aggregate = np.zeros(np.shape(self.JoutV))

        self.dJout_aggregate = np.zeros(np.shape(self.Jout))
        self.dJout_sigma_aggregate = np.zeros(np.shape(self.Jsigmaout))
        self.dJ_aggregate = np.zeros(np.shape(self.J))

        self.value=0
        self.r=0
        self.r_old=0

        self.h_Jout = np.zeros((self.O,))
        
        self.y = np.zeros((self.O,))

        # Remove self-connections
        
        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo']*0.;

        self.Vo = par['Vo'];

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N);
        self.S_hat = np.zeros (self.N);
        self.S_ro = np.zeros (self.N);
        self.dH = np.zeros (self.N);
        self.Vda = np.zeros (self.N);

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N);
        self.state_out_p = np.zeros (self.N);

        # Here we save the params dictionary
        self.par = par;

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));

        return out;

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv));

    def step (self, inp, t, probabilistic = False):
        itau_m = self.itau_m;
        itau_s = self.itau_s;
        itau_ro = self.itau_ro;

        self.S_hat   = (self.S_hat   * itau_s + self.S   * (1. - itau_s))
        self.dH   = self.dH  * (1. - itau_m) + itau_m * self.S_hat;
            
        self.H   = self.H   * (1. - itau_m) + itau_m * np.tanh ( self.J @ self.S_hat   + self.Jin @ inp ) ;
        #

        self.S   = self.H#self._sigm (self.H  , dv = self.dv) - 0.5 > 0.

        return self.H
    
    def step_rate (self, inp, t, probabilistic = False):
        itau_m = np.copy(1.-np.exp(-self.itau_m))
        itau_s = self.itau_s;
        itau_ro = self.itau_ro;

        self.S_hat   = np.copy(self.S)# (self.S_hat   * itau_s + self.S   * (1. - itau_s))
        self.S_ro   = (self.S_ro   * itau_ro + self.S   * (1. - itau_ro))
        self.dH   = self.dH  * (1. - itau_m) + itau_m * self.S_hat
        #self.H   = self.H   * (1. - itau_m) + itau_m * np.tanh ( self.J @ self.S_hat   + self.Jin @ inp )
        self.H   = self.H   * (1-self.dt/self.tau_m) + (self.dt/self.tau_m)  * (self.J @ self.S_hat   + self.Jin @ inp )
        
            
        #self.H   = self.H   * (1. - itau_m) + itau_m * (self.J @ self.S_hat   + self.Jin @ inp + self.h)#\
                                                          #+ self.Jreset @ self.S  ;
        #

        self.S  = np.copy(self.H) #self._sigm (self.H  , dv = self.dv)
        self.y = np.tanh(self.Jout@ self.S_ro + self.h_Jout)
        #self.y = self.Jout@ self.S_ro + self.h_Jout

        return self.H


    def reset (self, init = None):
        self.S   = init if init else np.zeros (self.N);
        self.S_hat   = self.S   * self.itau_s if init else np.zeros (self.N);
        self.S_ro   = self.S   * self.itau_s if init else np.zeros (self.N);
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  *= 0.;
        #self.H  += self.Vo
        self.y = np.zeros((self.O,))
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1;
            self.I_clock = I_clock


class RESERVOIRE_SIMPLE:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        #self.itau_m = self.dt / par['tau_m_f'];
        #self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        self.tau_m = np.linspace(  par['tau_m_f'],  par['tau_m_s'] ,self.N)
        #self.tau_m = np.logspace(  np.log(par['tau_m_f']) , np.log( par['tau_m_s']) ,self.N)
        #self.itau_m = np.linspace( self.dt / par['tau_m_f'], self.dt / par['tau_m_s'] ,self.N)

        self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro']);

        self.dv = par['dv']

        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec'], size = (self.N, self.N));#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices

        print(par['sigma_input'])

        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I))

        #print(self.Jin)

        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O));
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N));#np.zeros ((self.O, self.N));
        self.h_Jout = np.zeros((self.O,))
        
        self.y = np.zeros((self.O,))

        # Remove self-connections
        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh']
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h']

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo']*0.

        self.Vo = par['Vo']

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N)
        self.S_hat = np.zeros (self.N)
        self.S_ro = np.zeros (self.N)
        self.dH = np.zeros (self.N)
        self.Vda = np.zeros (self.N)

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N)
        self.state_out_p = np.zeros (self.N)

        # Here we save the params dictionary
        self.par = par

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));

        return out;

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv));

    def step_rate (self, inp, t, probabilistic = False):
        #itau_m = self.itau_m;
        itau_s = self.itau_s;
        itau_ro = self.itau_ro;

        self.S_hat   = np.copy(self.S)#(self.S_hat   * itau_s + self.S   * (1. - itau_s))
        #(self.S_ro   * itau_ro + self.S   * (1. - itau_ro))
        #self.dH   = self.dH  * (1. - itau_m) + itau_m * self.S_hat
        #self.H   = self.H   * (1. - itau_m) + itau_m * np.tanh ( self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * (1. - itau_m) + itau_m * (self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * np.exp(-self.dt/self.tau_m) + (1-np.exp(-self.dt/self.tau_m) ) * (self.J @ self.S_hat   + self.Jin @ inp )
        self.H   = self.H   * (1-self.dt/self.tau_m) + (self.dt/self.tau_m)  * (self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * np.exp(-self.dt/self.tau_m) + (1-np.exp(-self.dt/self.tau_m) ) * np.tanh (self.J @ self.S_hat   + self.Jin @ inp )
        
        #print((1. - itau_m))

        self.S  = np.copy(self.H)
        self.S_ro   = np.copy(self.S)
        self.y = np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)

        return self.H


    def reset (self, init = None):
        self.S   = init if init else np.zeros (self.N);
        self.S_hat   = self.S   * self.itau_s if init else np.zeros (self.N);
        self.S_ro   = self.S   * self.itau_s if init else np.zeros (self.N);
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  *= 0.;
        #self.H  += self.Vo
        self.y = np.zeros((self.O,))
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1;
            self.I_clock = I_clock

class GOAL_TAKENS_SIMPLE:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        #self.itau_m = self.dt / par['tau_m'];
        self.tau_m = np.logspace( np.log(par['tau_m_f']) ,np.log(par['tau_m_s']),self.N)
        self.itau_m = np.zeros(self.N,)#self.dt / par['tau_m'];
        self.itau_m[:int(self.N/2)] = self.dt / par['tau_m_f']
        self.itau_m[int(self.N/2):] = self.dt / par['tau_m_s']
        
        #self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        
        #self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro']);

        self.dv = par['dv'];

        self.inp_filt = 0.

        self.n_tau = 200
        #self.itau_s
        
        #self.tau_s = np.logspace(-3,0,self.n_tau)#np.array([1.,10.])*0.001
        #self.tau_s = np.logspace(-2.,0.,self.n_tau)#np.array([1.,10.])*0.001
        self.tau_s = np.linspace(1.,1000.,self.n_tau)*0.001#np.array([1.,10.])*0.001
        
        self.itau_s = np.copy(np.exp ( -self.dt /  self.tau_s ) )
        
        print(self.itau_s)
        
        #### definisci bene sti tau
        
        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec']*0., size = (self.N, self.N, self.n_tau) );#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I))
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O))
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N))#np.zeros ((self.O, self.N));
        
        self.bias_ro = 0.

        self.value=0
        self.r=0
        self.r_old=0
        
        self.y = 0

        # Remove self-connections
        for k in range(self.n_tau):
            np.fill_diagonal (np.reshape( self.J[:,:,k] , ( self.N,self.N )) , 0.);

        self.name = 'model'

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo']   
        self.Vda = np.zeros(self.N,)
        self.Vo = par['Vo']

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N)
        self.S_hat = np.zeros ((self.N,self.n_tau))
        self.S_ro = np.zeros (self.N)
        self.dH = np.zeros ((self.N,self.n_tau))

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N);
        self.state_out_p = np.zeros (self.N);

        # Here we save the params dictionary
        self.par = par;

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of sigmoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));

        return out;

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv));


    def step_rate (self, inp, t, probabilistic = False):
        self.alpha_m = np.exp(-self.dt/self.tau_m)
        #itau_s = self.itau_s;
        itau_ro = self.itau_ro
        #itau_m= np.exp (-self.dt / par['tau_s'])
        
        for k in range(self.n_tau):
            
            itau_s = self.itau_s[k]
            #print(np.shape(self.S_hat))
            self.S_hat[:,k]   = self.S_hat[:,k]   * itau_s + self.S #* (1. - itau_s)
        
        self.S_ro   = (self.S_ro   * itau_ro + self.S   * (1. -   itau_ro))
        
        for k in range(self.n_tau):
            self.dH[:,k]   = self.dH[:,k]  *self.alpha_m  +   self.S_hat[:,k]#(1. - self.alpha_m) *
        
        #self.mu = self.Jin @ inp + self.h
        alpha_inp = self.dt/self.tau_inp#(2.*0.005)
        self.inp_filt = self.inp_filt * (1.- alpha_inp) + inp#*alpha_inp
        self.mu = self.Jin @ self.inp_filt + self.h
        
        #self.mu = self.h
        #for k in range(self.n_tau):
        #    self.mu += self.J[:,:,k] @ self.S_hat[:,k]
        
        for k in range(self.n_tau):
            self.mu += self.J[:,:,k] @ self.S_hat[:,k]
        
        self.alpha_m = 0.
        self.H   = np.copy(self.mu)#self.H   * self.alpha_m  + (1. - self.alpha_m)* self.mu
        
        self.Vda = self.Vda  * (self.alpha_m) + (self.H - self.mu)
        #
        self.S   = np.copy(self.H)#np.tanh(self.H)) #self._sigm (self.H  , dv = self.dv)
        self.y = self.Jout@ self.S_ro+self.bias_ro#np.tanh(self.Jout@ self.S_ro)

        #return self.H


    def reset (self, init = None):
        
        self.S   = init if init else np.zeros (self.N)
        self.S_hat  = np.copy(self.S_hat*0.)
        
        self.S_ro   = np.copy(self.S*0.)
                                                                    
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  *= 0.;
        self.H  += self.Vo
        self.y = 0
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1;
            self.I_clock = I_clock





class GOAL_TAKENS_INP_SIMPLE:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        #self.itau_m = self.dt / par['tau_m'];
        self.tau_m = np.logspace( np.log(par['tau_m_f']) ,np.log(par['tau_m_s']),self.N)
        self.itau_m = np.zeros(self.N,)#self.dt / par['tau_m'];
        self.itau_m[:int(self.N/2)] = self.dt / par['tau_m_f']
        self.itau_m[int(self.N/2):] = self.dt / par['tau_m_s']
        
        #self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        
        #self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro']);

        self.dv = par['dv'];

        #self.inp_filt = 0.

        self.n_tau = par['n_tau'];
        #self.itau_s
        
        #self.tau_s = np.logspace(-3,0,self.n_tau)#np.array([1.,10.])*0.001
        #self.tau_s = np.logspace(-2.,0.,self.n_tau)#np.array([1.,10.])*0.001
        self.tau_s = np.linspace(1.,1000.,self.n_tau)*0.001#np.array([1.,10.])*0.001
        
        self.itau_s = np.copy(np.exp ( -self.dt /  self.tau_s ) )
        
        print(self.itau_s)
        
        #### definisci bene sti tau
        
        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec']*0., size = (self.N, self.N, self.n_tau) );#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I,self.n_tau));
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O));
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N));#np.zeros ((self.O, self.N));
        
        self.bias_ro = 0.

        self.value=0
        self.r=0
        self.r_old=0
        
        self.y = 0

        # Remove self-connections
        for k in range(self.n_tau):
            np.fill_diagonal (np.reshape( self.J[:,:,k] , ( self.N,self.N )) , 0.);

        self.name = 'model'

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo']   
        self.Vda = np.zeros(self.N,)
        self.Vo = par['Vo']

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N)
        self.S_hat = np.zeros ((self.N,self.n_tau))
        self.inp_filt = np.zeros ((self.I,self.n_tau))
        self.S_ro = np.zeros (self.N)
        self.dH = np.zeros ((self.N,self.n_tau))

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N);
        self.state_out_p = np.zeros (self.N);

        # Here we save the params dictionary
        self.par = par;

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of sigmoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));

        return out;

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv));


    def step_rate (self, inp, t, probabilistic = False):
        self.alpha_m = np.exp(-self.dt/self.tau_m)
        #itau_s = self.itau_s;
        itau_ro = self.itau_ro
        #itau_m= np.exp (-self.dt / par['tau_s'])
        
        for k in range(self.n_tau):
            
            itau_s = self.itau_s[k]
            #print(np.shape(self.S_hat))
            self.S_hat[:,k]   = self.S_hat[:,k]   * itau_s + self.S# * (1. - itau_s)
            self.inp_filt[:,k] = self.inp_filt[:,k] * itau_s + inp#* (1. - itau_s)
        
        self.S_ro   = (self.S_ro   * itau_ro + self.S   * (1. -   itau_ro))
        
        for k in range(self.n_tau):
            self.dH[:,k]   = self.dH[:,k]  *self.alpha_m  +   self.S_hat[:,k]#(1. - self.alpha_m) *
        
        #self.mu = self.Jin @ inp + self.h
        #alpha_inp = self.dt/(2.*0.005)
        #self.inp_filt = self.inp_filt * (1.- alpha_inp) + inp#*alpha_inp

        #self.mu = self.Jin @ self.inp_filt + self.h
        
        self.mu = np.copy(self.h)
        
        #for k in range(self.n_tau):
        #    self.mu += self.J[:,:,k] @ self.S_hat[:,k]
        
        
        for k in range(self.n_tau):
            self.mu += self.Jin[:,:,k] @ self.inp_filt[:,k]#*self.dt/0.025
            self.mu += self.J[:,:,k] @ self.S_hat[:,k]#*self.dt/0.025
            

        self.alpha_m = 0.
        self.H   = np.copy(self.mu)#self.H   * self.alpha_m  + (1. - self.alpha_m)* self.mu
        
        self.Vda = self.Vda  * (self.alpha_m) + (self.H - self.mu)
        #
        self.S   = np.copy(np.tanh(self.H)) #self._sigm (self.H  , dv = self.dv)
        self.y = self.Jout@ self.S_ro+self.bias_ro#np.tanh(self.Jout@ self.S_ro)

        #return self.H


    def reset (self, init = None):
        
        self.S   = init if init else np.zeros (self.N)
        self.S_hat  = np.copy(self.S_hat*0.)
        
        self.S_ro   = np.copy(self.S*0.)
                                                                    
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  *= 0.;
        self.H  += self.Vo
        self.y = 0
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1;
            self.I_clock = I_clock



class RESERVOIRE_SIMPLE_NL:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        #self.itau_m = self.dt / par['tau_m_f'];
        #self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        self.tau_m = np.linspace(  par['tau_m_f'],  par['tau_m_s'] ,self.N)
        #self.tau_m = np.logspace(  np.log(par['tau_m_f']) , np.log( par['tau_m_s']) ,self.N)
        #self.itau_m = np.linspace( self.dt / par['tau_m_f'], self.dt / par['tau_m_s'] ,self.N)

        self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro']);

        self.dv = par['dv']

        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec'], size = (self.N, self.N));#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I));
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O));
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N));#np.zeros ((self.O, self.N));
        self.h_Jout = np.zeros((self.O,))
        
        self.y = np.zeros((self.O,))

        # Remove self-connections
        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N,) * par['Vo']*0.;

        self.Vo = par['Vo'];

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N,);
        self.S_hat = np.zeros (self.N,);
        self.S_ro = np.zeros (self.N,);
        self.dH = np.zeros (self.N,);
        self.Vda = np.zeros (self.N,);

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N,);
        self.state_out_p = np.zeros (self.N,);

        # Here we save the params dictionary
        self.par = par;

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));
        out = out+1.
        return out

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv));

    def step_rate (self, inp, sigma_S, probabilistic = False,if_tanh=True):
        #itau_m = self.itau_m;
        itau_s = self.itau_s
        itau_ro = self.itau_ro

        self.S_hat   = np.copy(self.S)#(self.S_hat   * itau_s + self.S   * (1. - itau_s))
        #(self.S_ro   * itau_ro + self.S   * (1. - itau_ro))
        #self.dH   = self.dH  * (1. - itau_m) + itau_m * self.S_hat
        #self.H   = self.H   * (1. - itau_m) + itau_m * np.tanh ( self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * (1. - itau_m) + itau_m * (self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * np.exp(-self.dt/self.tau_m) + (1-np.exp(-self.dt/self.tau_m) ) * (self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * (1-self.dt/self.tau_m) + (self.dt/self.tau_m)  * (self.J @ self.S_hat   + self.Jin @ inp )

        self.H   = self.H   * np.exp(-self.dt/self.tau_m) + (1-np.exp(-self.dt/self.tau_m) ) * np.tanh ( self.J @ self.S_hat   + self.Jin @ inp )

        self.S  = np.copy(self.H + np.random.normal (0., sigma_S, size = np.shape(self.H) ))#(np.tanh(self.H)+1)/2
        self.S_ro   = np.copy(self.S)
        if if_tanh:
            self.y = np.tanh(self.Jout@ self.S_ro + self.h_Jout)# np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)
        else:
            self.y = (self.Jout@ self.S_ro + self.h_Jout)# np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)

        return self.H


    def reset (self, init = None):
        self.S   = np.zeros ((self.N,))#init if init else np.zeros (self.N,)
        self.S_hat   = np.zeros ((self.N,)) #  * self.itau_s if init else np.zeros (self.N,)
        self.S_ro   = np.zeros ((self.N,))#   * self.itau_s if init else np.zeros (self.N,)
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  = np.zeros ((self.N,))
        #self.H  += self.Vo
        self.y = np.zeros((self.O,))
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1
            self.I_clock = I_clock

    def learn_error (self, r , g1 ):

            ch = 0.#0.002

            alpha_J = 0.002

            #dJ = np.outer ((self.S [:] - self.lam), self.dH)
            ac_vector = np.zeros((self.O,))
            ac_vector[self.action] = 1
            
            dJ_ent_out = - np.outer (  self.prob*np.log(self.prob ) + self.entropy , self.state_out.T)

            dJ_out =  g1*np.outer((ac_vector - self.prob), self.S.T)#np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)
            
            self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out

            self.dJout_aggregate += (r*self.dJfilt_out + ch*dJ_ent_out)

    def learn_error_hpg (self, r , g , g_coeff):

                ch = 0.001#0.002

                alpha_J = 0.001

                #dJ = np.outer ((self.S [:] - self.lam), self.dH)
                ac_vector = np.zeros((3,))
                ac_vector[self.action] = 1
                dJ_out = np.zeros((self.O,self.N))
                dJ_ent_out = np.zeros((self.O,self.N))

                dJ_ent_out[0:3,:] = - g[0]*np.outer (  self.prob*np.log(self.prob ) + self.entropy , self.state_out.T)
                dJ_ent_out[3:6,:] = - g[1]*np.outer (  self.prob*np.log(self.prob ) + self.entropy , self.state_out.T)

                dJ_out[0:3,:] =  g[0]*np.outer((ac_vector - self.prob), self.S.T)#+dJ_ent_out_1#np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)
                dJ_out[3:6,:] =  g[1]*np.outer((ac_vector - self.prob), self.S.T)#+dJ_ent_out_2#np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)
                dJ_out[6,:] =  np.sum(np.outer((ac_vector - self.prob)*(self.y[0:3]-self.y[3:6]), self.S.T) *g_coeff,axis=0)#*(1-np.tanh(self.y[-1])**2)

                self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out
                self.grad = np.copy(r*self.dJfilt_out)
                self.dJout_aggregate += (r*self.dJfilt_out)+ ch*dJ_ent_out

    def learn_error_hpg_multig (self, r , g , g_coeff,num_g):

                ch = .01#0.002
                alpha_J = 0.001

                #dJ = np.outer ((self.S [:] - self.lam), self.dH)
                ac_vector = np.zeros((3,))
                ac_vector[self.action] = 1
                dJ_out = np.zeros((self.O,self.N))
                dJ_ent_out = np.zeros((self.O,self.N))

                for gi in range(num_g):
                    dJ_out[3*gi :3*gi + 3,:] =  g[gi]*np.outer((ac_vector - self.prob), self.S.T)#+dJ_ent_out_1#np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)
                    dJ_ent_out[3*gi :3*gi + 3,:] = - g[gi]*np.outer (  self.prob* ( np.log(self.prob ) + self.entropy  ), self.S.T)
                    dJ_out[3*num_g + gi ,:] =  np.sum(np.outer((ac_vector - self.prob)*(self.y[3*gi :3*gi + 3]), self.S.T* g[gi]*(1-g[gi])  ) *g_coeff,axis=0)#*(1-np.tanh(self.y[-1])**2)

#                dJ_out[6,:] =  np.sum(np.outer((ac_vector - self.prob)*(self.y[0:3]-self.y[3:6]), self.S.T) *g_coeff,axis=0)#*(1-np.tanh(self.y[-1])**2)

                self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out
                self.grad = np.copy(r*self.dJfilt_out)
                self.dJout_aggregate += (r*self.dJfilt_out)+ ch*dJ_ent_out


    def update_J (self, r):

            #self.J = self.adam_rec.step (self.J, self.dJ_aggregate)#
            #self.J = self.J + self.par["alpha"]*self.dJ_aggregate#0.1*10*.5*.5*.5 - self.J*0.0000001
            np.fill_diagonal (self.J, 0.)

            self.Jout = self.adam_out.step (self.Jout, self.dJout_aggregate)#self.Jout + self.par["alpha_rout"]*self.dJout_aggregate### 0.01*.5*.5*.5 - self.Jout*0.0000001
            #self.Jout = self.Jout + self.par["alpha_rout"]*self.dJout_aggregate### 0.01*.5*.5*.5 - self.Jout*0.0000001
            #self.JoutV = self.adam_outV.step (self.JoutV, self.dJoutV_aggregate)

            #self.JoutV = self.JoutV + self.par["alpha_rout"]*self.dJoutV_aggregate
            #print(np.std(self.dJout_aggregate))
            self.dJout_aggregate=0




class RESERVOIRE_SIMPLE_NL_MULT:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        #self.itau_m = self.dt / par['tau_m_f'];
        #self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        self.tau_m = np.linspace(  par['tau_m_f'],  par['tau_m_s'] ,self.N)
        #self.tau_m = np.logspace(  np.log(par['tau_m_f']) , np.log( par['tau_m_s']) ,self.N)
        #self.itau_m = np.linspace( self.dt / par['tau_m_f'], self.dt / par['tau_m_s'] ,self.N)

        self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro']);

        self.dv = par['dv']

        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec'], size = (self.N, self.N));#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I));
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O));
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N));#np.zeros ((self.O, self.N));
        self.h_Jout = np.zeros((self.O,))
        
        self.y = np.zeros((self.O,))

        # Remove self-connections
        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N,) * par['Vo']*0.;

        self.Vo = par['Vo'];

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N,);
        self.S_hat = np.zeros (self.N,);
        self.S_ro = np.zeros (self.N,);
        self.dH = np.zeros (self.N,);
        self.Vda = np.zeros (self.N,);

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N,);
        self.state_out_p = np.zeros (self.N,);

        # Here we save the params dictionary
        self.par = par;

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));
        out = out+1.
        return out

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv));

    def step_rate (self, inp, inp_modulation,sigma_S, if_tanh=True):
        #itau_m = self.itau_m;
        itau_s = self.itau_s
        itau_ro = self.itau_ro

        self.S_hat   = np.copy(self.S)#(self.S_hat   * itau_s + self.S   * (1. - itau_s))
        #(self.S_ro   * itau_ro + self.S   * (1. - itau_ro))
        #self.dH   = self.dH  * (1. - itau_m) + itau_m * self.S_hat
        #self.H   = self.H   * (1. - itau_m) + itau_m * np.tanh ( self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * (1. - itau_m) + itau_m * (self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * np.exp(-self.dt/self.tau_m) + (1-np.exp(-self.dt/self.tau_m) ) * (self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * (1-self.dt/self.tau_m) + (self.dt/self.tau_m)  * (self.J @ self.S_hat   + self.Jin @ inp )

        self.H   = self.H   * np.exp(-self.dt/self.tau_m) + (1-np.exp(-self.dt/self.tau_m) ) * np.tanh( ( self.J @ self.S_hat   + self.Jin @ inp )* (inp_modulation) )

        self.S  = np.copy(self.H + np.random.normal (0., sigma_S, size = np.shape(self.H) ))#(np.tanh(self.H)+1)/2
        self.S_ro   = np.copy(self.S)
        #self.y = np.tanh(self.Jout@ self.S_ro + self.h_Jout)# np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)
        if if_tanh:
            self.y = np.tanh(self.Jout@ self.S_ro + self.h_Jout)# np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)
        else:
            self.y = (self.Jout@ self.S_ro + self.h_Jout)# np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)


        return self.H


    def reset (self, init = None):
        self.S   = np.zeros ((self.N,))#init if init else np.zeros (self.N,)
        self.S_hat   = np.zeros ((self.N,)) #  * self.itau_s if init else np.zeros (self.N,)
        self.S_ro   = np.zeros ((self.N,))#   * self.itau_s if init else np.zeros (self.N,)
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  = np.zeros ((self.N,))
        #self.H  += self.Vo
        self.y = np.zeros((self.O,))
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1
            self.I_clock = I_clock

class RESERVOIRE_SIMPLE_NL_MULT_OUT:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        #self.itau_m = self.dt / par['tau_m_f'];
        #self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        self.tau_m = np.linspace(  par['tau_m_f'],  par['tau_m_s'] ,self.N)
        #self.tau_m = np.logspace(  np.log(par['tau_m_f']) , np.log( par['tau_m_s']) ,self.N)
        #self.itau_m = np.linspace( self.dt / par['tau_m_f'], self.dt / par['tau_m_s'] ,self.N)

        self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro']);

        self.dv = par['dv']

        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec'], size = (self.N, self.N));#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I));
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O));
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N));#np.zeros ((self.O, self.N));
        self.h_Jout = np.zeros((self.O,))
        
        self.y = np.zeros((self.O,))

        # Remove self-connections
        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N,) * par['Vo']*0.;

        self.Vo = par['Vo'];

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N,);
        self.S_hat = np.zeros (self.N,);
        self.S_ro = np.zeros (self.N,);
        self.dH = np.zeros (self.N,);
        self.Vda = np.zeros (self.N,);

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N,);
        self.state_out_p = np.zeros (self.N,);

        # Here we save the params dictionary
        self.par = par;

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));
        out = out+1.
        return out

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv));

    def step_rate (self, inp, inp_modulation,sigma_S, if_tanh=True):
        #itau_m = self.itau_m;
        itau_s = self.itau_s
        itau_ro = self.itau_ro

        self.S_hat   = np.copy(self.S)
        self.H   = self.H   * np.exp(-self.dt/self.tau_m) + (1-np.exp(-self.dt/self.tau_m) ) * np.tanh( ( self.J @ self.S_hat   + self.Jin @ inp ) )*np.tanh( (inp_modulation) )

        self.S  = np.copy(self.H + np.random.normal (0., sigma_S, size = np.shape(self.H) ))#(np.tanh(self.H)+1)/2
        self.S_ro   = np.copy(self.S)
        #self.y = np.tanh(self.Jout@ self.S_ro + self.h_Jout)# np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)
        if if_tanh:
            self.y = np.tanh(self.Jout@ self.S_ro + self.h_Jout)# np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)
        else:
            self.y = (self.Jout@ self.S_ro + self.h_Jout)# np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)


        return self.H


    def reset (self, init = None):
        self.S   = np.zeros ((self.N,))#init if init else np.zeros (self.N,)
        self.S_hat   = np.zeros ((self.N,)) #  * self.itau_s if init else np.zeros (self.N,)
        self.S_ro   = np.zeros ((self.N,))#   * self.itau_s if init else np.zeros (self.N,)
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  = np.zeros ((self.N,))
        #self.H  += self.Vo
        self.y = np.zeros((self.O,))
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1
            self.I_clock = I_clock
