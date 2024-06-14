"""
© 2024 This work is licensed under a CC-BY-NC-SA license.
Title:
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


class RESERVOIRE_SIMPLE_NL:

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
