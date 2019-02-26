# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ['Maelstrom', 'RadialVelocity']

from multiprocessing import Pool, cpu_count

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
# import hemcee
# from hemcee.sampler import TFModel

from .kepler import kepler
from .estimator import estimate_frequencies

class Maelstrom(object):
    """The real deal

    Args:
        time (array-like): 
        mag (array-like):
        nu (optional): 
        log_sigma2:
        session:
        max_peaks:
        **kwargs:
        """

    T = tf.float64
    C = 299792.458 # Speed of light km/s

    def __init__(self, time, mag, nu=None, rvs=None, log_sigma2=None, 
                session=None, max_peaks=9, **kwargs):
        self.time_data = np.atleast_1d(time)
        self.mag_data = np.atleast_1d(mag)

        self.rvs = rvs or []

        # Estimate frequencies if none are supplied
        if nu is None:
            nu = estimate_frequencies(time, mag, max_peaks=max_peaks)
        self.nu_data = np.atleast_1d(nu)

        # Placeholder tensors for time and mag data
        self.time = tf.constant(self.time_data, dtype=self.T)
        self.mag = tf.constant(self.mag_data, dtype=self.T)

        self._session = session

        # Parameters
        if log_sigma2 is None:
            log_sigma2 = 0.0
        self.log_sigma2 = tf.Variable(log_sigma2, dtype=self.T,
                                      name="log_sigma2")
        self.nu = tf.Variable(self.nu_data, dtype=self.T, name="frequencies")

        self.setup_orbit_model(**kwargs)

        arg = 2.0*np.pi*self.nu[None, :] * (self.time[:, None] - self.tau)
        D = tf.concat([tf.cos(arg), tf.sin(arg),
                       tf.ones((len(self.time_data), 1), dtype=self.T)],
                      axis=1)

        # Solve for the amplitudes and phases of the oscillations
        DTD = tf.matmul(D, D, transpose_a=True)
        DTy = tf.matmul(D, self.mag[:, None], transpose_a=True)
        W_hat = tf.linalg.solve(DTD, DTy)

        # Model and the chi^2 objective:
        self.model_tensor = tf.squeeze(tf.matmul(D, W_hat))
        self.chi2 = tf.reduce_sum(tf.square(self.mag - self.model_tensor))
        self.chi2 *= tf.exp(-self.log_sigma2)
        self.chi2 += len(self.time_data) * self.log_sigma2

        # Add radial velocity to objective
        for rv in self.rvs:
            self.chi2 += tf.reduce_sum(rv.chi)
        
        # Initialize all the variables
        self.run(tf.global_variables_initializer())

        # Minimal working feed dict for lightcurve
        self._feed_dict = {
            self.time: self.time_data,
            self.mag: self.mag_data
        }

        # Update feed dict with RV data
        for rv in self.rvs:
            self._feed_dict.update({
                rv.time: rv.time_data,
                rv.vel: rv.vel_data,
                rv.err: rv.err_data
            })

        # Wrap tensorflow in hemcee model
        self._tf_model = TFModel(self.ln_prob, self.params, 
                        self.feed_dict, session=self._session)

        self._step = hemcee.step_size.VariableStepSize()

        self._sampler = hemcee.NoUTurnSampler(self.tf_model.value, 
                            self.tf_model.gradient, step_size=self._step)

    @property
    def sampler(self):
        self._sampler = hemcee.NoUTurnSampler(self.tf_model.value, 
                            self.tf_model.gradient, step_size=self._step)
        return self._sampler

    @property
    def tf_model(self):
        #self._packed_params = self._pack(self.params)
        self._tf_model = TFModel(self.ln_prob, self.params, 
                        self.feed_dict, session=self._session)
        self._tf_model.setup()
        return self._tf_model

    @property
    def ln_prob(self):
        return - 0.5 * self.chi2 # + self.ln_prior

    @property
    def feed_dict(self):
        return self._feed_dict

    @feed_dict.setter
    def feed_dict(self, feed_dict):
        self._feed_dict = feed_dict

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params
        # Reorganize ln prior here

    @property
    def session(self):
        if self._session is None:
            self._session = tf.Session()
        return self._session

    def setup_orbit_model(self, with_eccen=True):
        self.with_eccen = with_eccen

        self.period = tf.Variable(1.0, dtype=self.T, name="period")
        self.lighttime = tf.Variable(np.zeros_like(self.nu_data), dtype=self.T,
                                     name="lighttime")
        self.lighttime_inds = tf.Variable(
            np.arange(len(self.nu_data)).astype(np.int32), dtype=tf.int32,
            name="lighttime_inds")
        self.tref = tf.Variable(0.0, dtype=self.T, name="tref")
        if self.with_eccen:
            self.eccen_param = tf.Variable(-5.0, dtype=self.T,
                                           name="eccen_param")
            self.varpi = tf.Variable(0.0, dtype=self.T, name="varpi")
            self.eccen = 1.0 / (1.0 + tf.exp(-self.eccen_param))

        # Which parameters do we fit for?
        self.params = [
            self.log_sigma2, self.period, self.lighttime, self.tref
        ]
        if self.with_eccen:
            self.params += [self.eccen_param, self.varpi]

        # If radial velocity data is supplied, generate gammav, logsigma2
        if self.rvs:
            self.gamma_v = tf.Variable(np.mean(self.rvs[0].vel_data) / self.C, 
                                    dtype=self.T, name="gamma_v")
            self.log_rv_sigma2 = tf.Variable(0.0, dtype=self.T, 
                                            name="log_rv_sigma2")
            self.params += [self.gamma_v, self.log_sigma2]

        self._lighttime_per_mode = tf.gather(self.lighttime,
                                             self.lighttime_inds)

        # Incorporating RV data
        for rv in self.rvs:
            # Keplers equations
            rv.mean_anom = 2.0 * np.pi * (rv.time - self.tref) / self.period
            rv.ecc_anom = kepler(rv.mean_anom, self.eccen)
            rv.true_anom = (2.0 * tf.atan2(tf.sqrt(1.0+self.eccen) *
                            tf.tan(0.5*rv.ecc_anom), tf.sqrt(1.0-self.eccen)
                            + tf.zeros_like(rv.time)))
            
            # Here we define how the RV will be calculated
            # This gives (J,N)
            rv.vrad = ((self._lighttime_per_mode / 86400)[None,:] * (-2.0 * np.pi * (1 / self.period) * 
                (1/tf.sqrt(1.0 - tf.square(self.eccen))) * 
                (tf.cos(rv.true_anom + self.varpi) + 
                self.eccen*tf.cos(self.varpi)))[:,None]
                )
            rv.vrad *= self.C
            rv.vrad += self.gamma_v

            # Account for uncertainty in RV
            rv.sig2 = tf.square(rv.err) + tf.exp(-self.log_rv_sigma2)

            # Add objective to RV object
            rv.chi = (tf.square(rv.vel[:,None] - rv.vrad) / rv.sig2[:,None]
                   + tf.log(rv.sig2[:,None]))

        # Set up the model
        self.mean_anom = 2.0*np.pi*(self.time-self.tref)/self.period
        if self.with_eccen:
            self.ecc_anom = kepler(self.mean_anom, self.eccen)
            self.true_anom = 2.0*tf.atan2(
                tf.sqrt(1.0+self.eccen) * tf.tan(0.5*self.ecc_anom),
                tf.sqrt(1.0-self.eccen) + tf.zeros_like(self.time))
            factor = 1.0 - tf.square(self.eccen)
            factor /= 1.0 + self.eccen*tf.cos(self.true_anom)
            self.psi = -factor * tf.sin(self.true_anom + self.varpi)
        else:
            self.psi = -tf.sin(self.mean_anom)

        # Build the design matrix
        self.tau = (self._lighttime_per_mode / 86400)[None, :] * self.psi[:, None]

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def __del__(self):
        if self._session is not None:
            self._session.close()

    def init_from_orbit(self, period, lighttime, tref=0.0, eccen=1e-5,
                        varpi=0.0):
        """Initialize the parameters based on an orbit estimate

        Args:
            period: The orbital period in units of ``time``.
            lighttime: The projected light travel time in units of ``time``
                (:math:`a_1\,\sin(i)/c`).
            tref: The reference time in units of ``time``.
            eccen: The orbital eccentricity.
            varpi: The angle of the ascending node in radians.

        """
        ops = []
        ops.append(tf.assign(self.period, period))
        ops.append(tf.assign(self.lighttime,
                             lighttime + tf.zeros_like(self.lighttime)))
        ops.append(tf.assign(self.tref, tref))
        if self.with_eccen:
            ops.append(tf.assign(self.eccen_param,
                                 np.log(eccen) - np.log(1.0 - eccen)))
            ops.append(tf.assign(self.varpi, varpi))
        self.run(ops)

    def optimize(self, params=None, **kwargs):
        """Optimizes the TensorFlow model using Scipy interface
 
        Args:
            params (optional) : List of TensorFlow variables
            **kwargs (optional) : Args to pass to opt.minimize()
        """
        if params is None:
            params = self.params
        opt = tf.contrib.opt.ScipyOptimizerInterface(self.chi2, params,
                                                     **kwargs)
        return opt.minimize(self.session, feed_dict=self.feed_dict) 

    def get_lighttime_estimates(self):
        """Estimates lighttime values """
        ivar = -tf.diag_part(tf.hessians(-0.5*self.chi2,
                                         self._lighttime_per_mode)[0])
        return self.run([self._lighttime_per_mode, np.abs(ivar)])

    def pin_lighttime_values(self):
        """Pins estimated lighttime values to positive or negative. """
        lt, lt_ivar = self.get_lighttime_estimates()
        chi = lt * np.sqrt(lt_ivar)
        mask_lower = chi < -1.0
        # Upper mask for case where lighttime is always negative.
        # Otherwise there's a div 0 in lt
        mask_upper = chi > 1.0
        if np.any(mask_lower) and np.any(mask_upper):
            m1 = lt >= 0
            m2 = ~m1
            lt = np.array([
                np.sum(lt_ivar[m1]*lt[m1]) / np.sum(lt_ivar[m1]),
                np.sum(lt_ivar[m2]*lt[m2]) / np.sum(lt_ivar[m2]),
            ])
            inds = 1 - m1.astype(np.int32)
        else:
            inds = np.zeros(len(lt), dtype=np.int32)
            lt = np.array([np.sum(lt_ivar*lt) / np.sum(lt_ivar)])

        self.run([
            tf.assign(self.lighttime_inds, inds),
            tf.assign(self.lighttime[:len(lt)], lt),
            tf.assign(self.lighttime[len(lt):],
                      np.zeros(len(lt_ivar)-len(lt))),
        ])
        return inds, lt

    def run_mcmc(self, samples=1000):
        """Runs model using Hemcee, storing samples and lnprob in chain attrib
 
        Args:
            samples (optional, default 1000) 
            parallel (bool): Whether to run multiple chains in parallel
        """
        def sampler_wrap(sample):
            np.random.seed()
            return self.sampler.run_mcmc(
                    [self.run(param) for param in self.params], 
                    sample
                    )

        # Apparently Pool can not pickle class methods in python ...
        parallel=False
        if parallel:
            num_cpu = cpu_count()
            pool = Pool(processes=num_cpu)

            sample_split = [int(np.ceil(samples/num_cpu))]*num_cpu
            pool_results = pool.map(sampler_wrap, sample_split)

            self.chain = [res for res in pool_results]
        else:
            self.chain = sampler_wrap(samples)

    def run_warmup(self, samples=1000):
        """Runs warmup of model using Hemcee, replaces tensor values
 
        Args:
            samples (optional, default 1000)
            **kwargs : args to pass to Hemcee
        """
        
        results = self.sampler.run_warmup(self.tf_model.current_vector(),
                                    samples,
                                    tune_metric=True)

        # Set optimization params to their new values
        self._update_placeholders = [
            array_ops.placeholder(param.dtype) for param in self.params
        ]
        self._param_updates = [
                param.assign(array_ops.reshape(placeholder, self._get_shape_tuple(param)))
                for param, placeholder in zip(self.params, self._update_placeholders)
        ]
        self.run(
            self._param_updates,
            feed_dict=dict(zip(self._update_placeholders, results[0]))
            )

    def get_bounded_for_value(self, value, min_value, max_value):
        if np.any(value <= min_value) or np.any(value >= max_value):
            raise ValueError("value must be in the range (min_value,\
             max_value)")
        return np.log(value - min_value) - np.log(max_value - value)

    def get_value_for_bounded(self, param, min_value, max_value):
        return min_value + (max_value - min_value) / (1.0 + np.exp(-param))

    def get_bounded_variable(self, name, value, min_value, max_value, 
                            dtype=tf.float64):
        param = tf.Variable(self.get_bounded_for_value(value, min_value, 
                            max_value), dtype=dtype, name=name + "_param")
        var = min_value + (max_value - min_value) / (1.0 + tf.exp(-param))
        log_jacobian = (tf.log(var - min_value) + tf.log(max_value - var) -
                        np.log(max_value - min_value))
        return param, var, tf.reduce_sum(log_jacobian), (min_value, max_value)

    def _pack(self, tensors):
        """Pack a list of `Tensor`s into a single rank-1 `Tensor`."""
        if not tensors:
            return None
        elif len(tensors) == 1:
            return array_ops.reshape(tensors[0], [-1])
        else:
            flattened = [array_ops.reshape(tensor, [-1]) for tensor in tensors]
            return array_ops.concat(flattened, 0)

    def _get_shape_tuple(self, tensor):
        """ Exactly what it says on the tin. """
        return tuple(dim.value for dim in tensor.get_shape())

class RadialVelocity(object):
    """Radial velocity class for handling RV data

    Args:
        time: The array of timestamps.
        vel: The velocities measured at ``time``.
        err (Optional): An array of errors of vel (in units of vel).
            (default: ``None``)
        meta (Optional): Extra information about the RV data
            (default: ``{}``)
    """
    T = tf.float64

    def __init__(self, time, vel, err=None, meta={}):
        self.time_data = time
        self.vel_data = vel
        if err is not None:    
            self.err_data = err
        self.meta = meta

        self.time = tf.constant(self.time_data, dtype=self.T)
        self.vel = tf.constant(self.vel_data, dtype=self.T)
        self.err = tf.constant(self.err_data, dtype=self.T)