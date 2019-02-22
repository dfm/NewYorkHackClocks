
# coding: utf-8

# In[1]
import numpy as np
import pandas as pd
import tensorflow as tf
import corner
import matplotlib.pyplot as plt

import hemcee
from hemcee.sampler import TFModel
from maelstrom.kepler import kepler

# In[2]
kicid=5709664 # PB1/SB1
rv = False
Hemcee = True
td=True

times, dmag = np.loadtxt("kic5709664_appended-msMAP_Q99_llc.txt",usecols=(0,1)).T
time_mid = (times[0] + times[-1]) / 2.
times -= time_mid
dmmags = dmag * 1000. 

nu_arr = [19.44005582, 16.25960082, 22.55802495, 19.123847  , 27.87541656,
       22.07540612]

if rv:
    # Read in radial velocity data
    rv_JD, rv_RV, rv_err = np.loadtxt('kic5709664b_JDrv.txt',delimiter=",", usecols=(0,1,2)).T
    rv_JD -= time_mid

porb = 95.4
a1 = 114.
tp = -220
e = 0.483
varpi = 0.388
a1d = a1#/86400.0


# In[4]:
class BoundParam(object):
    def __init__(self, name, value, min_value, max_value, dtype=tf.float64):
        self.name = name
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        
        # Bound
        self.param = tf.Variable(self.get_bounded_for_value(self.value, self.min_value, self.max_value), dtype=dtype, name=name + "_param")
        self.var = self.min_value + (self.max_value - self.min_value) / (1.0 + tf.exp(-self.param))
        
        self.log_jacobian = tf.log(self.var - self.min_value) + tf.log(self.max_value - self.var) - np.log(self.max_value - self.min_value)
        # Add this to the log prior
        self.log_prior = tf.reduce_sum(self.log_jacobian)
    def get_bounded_for_value(self, value, min_val, max_val):
        # Check if correct bounds
        if np.any(value <= min_val) or np.any(value >= max_val):
            raise ValueError("Value must be within the given bounds")
        return np.log(value-min_val)-np.log(max_val-value)
    
    def get_value_for_bounded(self,param):
        return self.min_value + (self.max_value - self.min_value) / (1.0 + np.exp(-param))
    
# In[5]: Setup tensorflow variables with bounds
sess = tf.InteractiveSession()
T = tf.float64

# Unbound tensors
nu_tensor = tf.Variable(nu_arr, dtype=T)

# Bound tensors
porb_tensor = BoundParam('Porb', porb, 1, 500)  # Orbital period
varpi_tensor = BoundParam('Varpi', varpi, 0, 5)   # Angle of the ascending node
tp_tensor = BoundParam('t_p', tp, -1000, 0) # Time of periastron
e_tensor = BoundParam('e', e, 1e-10, 0.99) # Eccentricity
log_sigma2_tensor = BoundParam('log_sigma2', -1.14, -5, 5) # Known value
a1d_tensor = BoundParam('a_1d', a1d, -300, 300.) # Projected semimajor axis

if rv:
    # Tensors specific to SB1/2
    
    # Some notes on gammav: 
    # If gammav is specified as gammav/c then scipy fit will work fine
    gammav_tensor = BoundParam('gammav',np.mean(rv_RV),-100,100)
    log_rv_sigma2_tensor = BoundParam('logrv', 0., -0.1,0.1)

times_tensor = tf.placeholder(T, times.shape)
dmmags_tensor = tf.placeholder(T, dmmags.shape)

if td:
    
    # Solve Kepler's equation
    mean_anom = 2.0 * np.pi * (times_tensor - tp_tensor.var) / porb_tensor.var
    ecc_anom = kepler(mean_anom, e_tensor.var)
    true_anom = 2.0 * tf.atan2(tf.sqrt(1.0+e_tensor.var)*tf.tan(0.5*ecc_anom),tf.sqrt(1.0-e_tensor.var) + tf.zeros_like(times_tensor))
    
    # Here we define how the time delay will be calculated:
    tau_tensor = -(a1d_tensor.var / 86400) * (1.0 - tf.square(e_tensor.var)) * tf.sin(true_anom + varpi_tensor.var) / (1.0 + e_tensor.var*tf.cos(true_anom))
    
    # And the design matrix:
    arg_tensor = 2.0 * np.pi * nu_tensor[None, :] * (times_tensor - tau_tensor)[:, None]
    D_tensor = tf.concat([tf.cos(arg_tensor), tf.sin(arg_tensor)], axis=1)
    
    # Define the linear solve for W_hat:
    DTD_tensor = tf.matmul(D_tensor, D_tensor, transpose_a=True)
    DTy_tensor = tf.matmul(D_tensor, dmmags_tensor[:, None], transpose_a=True)
    W_hat_tensor = tf.linalg.solve(DTD_tensor, DTy_tensor)
    
    # Finally, the model and the chi^2 objective:
    model_tensor = tf.squeeze(tf.matmul(D_tensor, W_hat_tensor)) # Removes dimensions of size 1 from the shape of a tensor.
    # Sometimes faster with switched signs on log_sigma2 here:
    chi2_tensor = tf.reduce_sum(tf.square(dmmags_tensor - model_tensor)) * tf.exp(-log_sigma2_tensor.var)
    chi2_tensor += len(times) * (log_sigma2_tensor.var)

if rv:
    
    # Equations specific to RV
    rv_time_tensor = tf.placeholder(T)
    rv_tensor = tf.placeholder(T)
    rv_err_tensor = tf.placeholder(T)
    
    # Solve Kepler's equation for the RVs
    rv_mean_anom = 2.0 * np.pi * (rv_time_tensor - tp_tensor.var) / porb_tensor.var
    rv_ecc_anom = kepler(rv_mean_anom, e_tensor.var)
    rv_true_anom = 2.0 * tf.atan2(tf.sqrt(1.0+e_tensor.var)*tf.tan(0.5*rv_ecc_anom), tf.sqrt(1.0-e_tensor.var) + tf.zeros_like(rv_time_tensor))
    
    # Here we define how the RV will be calculated:
    vrad_tensor = -2.0 * np.pi * ((a1d_tensor.var /86400) / porb_tensor.var) * (1/tf.sqrt(1.0 - tf.square(e_tensor.var))) * (tf.cos(rv_true_anom + varpi_tensor.var) + e_tensor.var*tf.cos(varpi_tensor.var))
    vrad_tensor *= 299792.458  # c in km/s
    vrad_tensor += gammav_tensor.var
    
    
    rv_sig2 = tf.square(rv_err_tensor) + tf.exp(log_rv_sigma2_tensor.var)
    chi = tf.square(rv_tensor - vrad_tensor) / rv_sig2 + tf.log(rv_sig2)
    
    if not td:
        print("RESETTING CHI2")
        chi2_tensor = tf.Variable(0., dtype=tf.float64)
    chi2_tensor += tf.reduce_sum(chi)

init = tf.global_variables_initializer()
sess.run(init)


feed_dict = {
    times_tensor: times,
    dmmags_tensor: dmmags
}

var = [
    porb_tensor,
    varpi_tensor, 
    tp_tensor,
    a1d_tensor,
    e_tensor, 
    log_sigma2_tensor,
]

if rv:
    feed_dict.update({
            rv_time_tensor: rv_JD,
            rv_tensor: rv_RV,
            rv_err_tensor: rv_err,
                    })
    var+=[
            log_rv_sigma2_tensor,
            gammav_tensor  #  
        ]
    
var_list = [tensors.param for tensors in var]
# In[8]:
if Hemcee:
    # Calculate prior
    log_prior = tf.constant(0.0, dtype=tf.float64)
    # Add the jacobian to the prior
    for tensor in var:
        if tensor.log_prior is not None:
            log_prior += tensor.log_prior
            
    log_prob = - 0.5 * chi2_tensor + log_prior
    
    model = TFModel(log_prob, var_list=var_list, feed_dict=feed_dict)
    model.setup()
    coords = model.current_vector()

    metric = hemcee.metric.DenseMetric(np.eye(len(coords)))
    step = hemcee.step_size.VariableStepSize()
    sampler = hemcee.NoUTurnSampler(model.value, model.gradient, step_size=step, metric=metric)
    
    # Burn-in
    results = sampler.run_warmup(coords, 1500, tune_metric=True)
    
    # Run the sampler
    coords_chain, logprob_chain = sampler.run_mcmc(
            results[0], 
            5000, 
            initial_log_prob=results[1], 
            var_names='', 
            plot=False, update_interval=100, 
            tune=False
            )
    
    plt.plot([coord[0] for coord in coords_chain])
    plt.title('$P_{orb}$ trace')
    
    
    for i,tensor in enumerate(var):
        tensor.real = tensor.get_value_for_bounded(coords_chain[:,i])
        
    ndim = len(coords)
    var_real = [tensor.real for tensor in var]
    figure = corner.corner(list(zip(*var_real)),
                           labels=[tensor.name for tensor in var],
                           #quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 12})
    
    
    true_vars = [porb, varpi, tp, e, 0, a1d]
    true_vars = [tensor.value for tensor in var]
    sample_vars = [np.median(tensor.real) for tensor in var]
    
    axes = np.array(figure.axes).reshape((ndim, ndim))
    for i in range(len(var_list)):   
        ax = axes[i, i]
        ax.axvline(true_vars[i], color="b")
        ax.axvline(sample_vars[i], color="r")
        
    for yi in range(len(var_list)):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(sample_vars[xi], color="r")
            ax.axvline(true_vars[xi], color="b")
            ax.axhline(sample_vars[yi], color="r")
            ax.axhline(true_vars[yi], color="b")
            
    if rv:
        fig = plt.figure()
        rv_phi_test = np.sort(np.linspace(0, np.mean(porb_tensor.real), 5000) % np.mean(porb_tensor.real))
        vrad_test = sess.run(vrad_tensor, feed_dict={rv_time_tensor: rv_phi_test})
        plt.errorbar((rv_JD % np.mean(porb_tensor.real))/np.mean(porb_tensor.real),rv_RV,rv_err,fmt=".",label='RV obs')
        plt.plot(rv_phi_test/np.mean(porb_tensor.real), vrad_test,label='RV th')
        plt.xlabel("Orbital phase")
        plt.ylabel("RV (km/s)")
        plt.legend()
        plt.show()
        

else:
    # Use Scipy to minimise
    for i in var:
        print(i.name, ":", i.value, ':', i.get_value_for_bounded(sess.run(i.param)))
    
    opt = tf.contrib.opt.ScipyOptimizerInterface(chi2_tensor, var_list=var_list)
    for i in range(10):
        opt.minimize(sess, feed_dict=feed_dict)
    
    for i,tensor in enumerate(var):
        tensor.real = tensor.get_value_for_bounded(sess.run(tensor.param))
        
    for i in var:
        print(i.name, ":", np.round(i.value,5), ':', i.get_value_for_bounded(sess.run(i.param)))
    if rv:
        rv_phi_test = np.sort(np.linspace(0, porb_tensor.real, 5000) % porb_tensor.real)
        vrad_test = sess.run(vrad_tensor, feed_dict={rv_time_tensor: rv_phi_test})
        plt.errorbar((rv_JD % porb_tensor.real)/porb_tensor.real,rv_RV,rv_err,fmt=".",label='RV obs')
        plt.plot(rv_phi_test/porb_tensor.real, vrad_test,label='RV th')
        plt.xlabel("Orbital phase")
        plt.ylabel("RV (km/s)")
        plt.legend()
        plt.show()
    
#sess.close()
tf.InteractiveSession.close(sess)