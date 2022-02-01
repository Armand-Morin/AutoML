import numpy as np
from utils import get_maturities_relative, get_maturities_absolute, hours
import pandas as pd

class Bachelier(object):
    
    def __init__(self,  init, calendar, volatility, drift, \
                 correlation):
        self.init = init
        self.volatility = volatility   
        self.drift = drift
        self.correlation = correlation
        self.calendar = calendar
        self.nb_dates = np.size(self.calendar)
        self.nb_diffusions = np.size(self.volatility)
        self.init_aleas_utils()

    def init_aleas_utils(self):
        self.vol_step = self.volatility.reshape(-1, 1) * \
            np.sqrt(np.diff(self.calendar)).reshape(1, -1)
        self.vol_step = self.vol_step.reshape(self.nb_diffusions, self.nb_dates-1, 1)
        self.drift_time = (self.drift.reshape(-1,1) * \
                 self.calendar.reshape(1, -1)).reshape(\
            self.nb_diffusions, self.nb_dates, 1)
    
    def get_prices(self, nb_simulations):
        returns = np.random.multivariate_normal(\
            np.zeros(self.nb_diffusions), \
            self.correlation, (self.nb_dates-1, nb_simulations)).transpose(2, 0, 1)
    
        brownian = np.zeros((self.nb_diffusions, self.nb_dates, nb_simulations))
        brownian[:,1:self.nb_dates,:] = np.cumsum(self.vol_step * returns, axis = 1)

        return brownian + self.drift_time + self.init.reshape(-1, 1, 1)

class BlackScholes(object):
    
    def __init__(self,  init, calendar, volatility, drift, \
                 correlation):
        self.init = init
        self.volatility = volatility   
        self.drift = drift
        self.correlation = correlation
        self.calendar = calendar
        self.nb_dates = np.size(self.calendar)
        self.nb_diffusions = np.size(self.volatility)
        self.init_aleas_utils()
        self.init_martingale_correction()
        
    def init_martingale_correction(self):
        self.martingale_correction = \
            (self.init.reshape(-1,1) * np.exp((self.drift - \
            self.volatility**2/2).reshape(-1,1) * \
                 self.calendar.reshape(1, -1))).reshape(\
            self.nb_diffusions, self.nb_dates, 1)

    def init_aleas_utils(self):
        self.vol_step = self.volatility.reshape(-1, 1) * \
            np.sqrt(np.diff(self.calendar)).reshape(1, -1)
        self.vol_step = self.vol_step.reshape(self.nb_diffusions, self.nb_dates-1, 1)

    def get_prices(self, nb_simulations):
        returns = np.random.multivariate_normal(\
            np.zeros(self.nb_diffusions), \
            self.correlation, (self.nb_dates-1, nb_simulations)).transpose(2, 0, 1)
    
        brownian = np.zeros((self.nb_diffusions, self.nb_dates, nb_simulations))
        brownian[:,1:self.nb_dates,:] = np.cumsum(self.vol_step * returns, axis = 1)
        return np.exp(brownian) * self.martingale_correction
    
class NFactorsModel(object):
    
    def __init__(self, products, calendar, volatility, mean_reverting, \
                 correlation, values_init):
        self.type = 'NFactors'
        self.products = products
        self.calendar = calendar   
        self.nb_dates = self.calendar.size
        self.nb_products = len(products)
        self.init_maturities()
        
        self.mean_reverting = mean_reverting
        self.correlation = correlation
        self.volatility = volatility
        self.values_init = values_init
        self.nb_diffusions = np.size(self.mean_reverting)
        self.nb_prices = self.mean_reverting.shape[0]
        self.nb_factors = self.mean_reverting.shape[1]
        self.init_aleas_utils()
        self.init_martingale_correction()
        
    def init_aleas_utils(self):
        time = np.array(hours(self.calendar-self.calendar[0])).reshape(1,-1,1)

        ind_reverting_null = self.mean_reverting.reshape(-1,) == 0.0

        mean_reverting = self.mean_reverting.reshape(-1, 1, 1)
        volatility = self.volatility.reshape(-1, 1, 1)
        mean_reverting_sum = mean_reverting.reshape(-1,1) + \
                        mean_reverting.reshape(1,-1)
        mean_reverting_sum_zero = np.copy(mean_reverting_sum)
        mean_reverting_sum_zero[mean_reverting_sum_zero == 0] = 1

        #computing of the term (1-exp(-mean_revergin * delivery_periods))/ \
        #(mean_reverting * delivery_period) which is of shape (nb_diffusions, 
        #nb_dates, nb_products)        
        product_mean_reverting_delivery = mean_reverting * \
            self.delivery_periods.reshape(1, self.nb_dates, self.nb_products)

        self.correction_shaping = 1 - np.exp(-product_mean_reverting_delivery)

        product_mean_reverting_delivery_zero = product_mean_reverting_delivery
        product_mean_reverting_delivery_zero[ind_reverting_null] = 1
        self.correction_shaping /= product_mean_reverting_delivery_zero
        self.correction_shaping[ind_reverting_null] = 1

        #computing of the covariance matrix of 
        #int_t_i^t_{i+1} e^{mean_reverting_i s} dW_s:
        #time_volatility[ind_diffusion, ind_diffusion2, ind_time] is the 
        #covariance matrix between int_t_ind_time^t_{ind_time+1} 
        #e^{mean_reverting_ind_diffusion * s} dW^{ind_diffusion_s} and 
        #int_t_ind_time^t_{ind_time+1} 
        #e^{mean_reverting_ind_diffusion2 * s} dW^{ind_diffusion2_s}
        
        time_volatility_temp = np.exp(mean_reverting_sum.reshape(\
                        self.nb_diffusions, self.nb_diffusions, 1) * \
                time.transpose(0,2,1))

        time_volatility_temp /= mean_reverting_sum_zero.reshape(self.nb_diffusions, \
                                                      self.nb_diffusions, 1)

        #index where mean_reverting[i] + mean_reverting[j] is null
        ind_reverting_null_sum = np.where(mean_reverting_sum == 0)

        time_volatility_temp[ind_reverting_null_sum[0], \
                    ind_reverting_null_sum[1], :] = time[0,:,0]
        
        time_volatility_temp = self.correlation.reshape(self.nb_diffusions, \
                                                    self.nb_diffusions, 1) * \
                                                    time_volatility_temp
        time_volatility = np.diff(time_volatility_temp, axis=2)  

        self.time_volatility_squared = np.array([self.chol(\
                time_volatility[:, :, ind_time])\
                for ind_time in range(self.nb_dates-1)]).transpose(2,1,0)

        self.time_volatility_squared = self.time_volatility_squared.reshape(\
                                            self.nb_diffusions, \
                            self.nb_diffusions, self.nb_dates-1, 1)
        
        self.volatility_to_multiply = volatility.reshape(-1,1,1) 
        
        
        self.exp_maturities = np.exp(- (self.time_to_maturity.reshape(1, \
                    self.nb_dates,self.nb_products) + time) * \
        self.mean_reverting.reshape(self.nb_diffusions, 1, 1))        

        self.volatility_to_multiply = self.correction_shaping.reshape(\
                self.nb_diffusions, self.nb_dates, self.nb_products) * \
                    self.volatility_to_multiply
        
        
        self.volatility_to_multiply = self.volatility_to_multiply * self.exp_maturities 

        self.volatility_to_multiply = self.volatility_to_multiply.reshape(\
                self.nb_diffusions, self.nb_dates, self.nb_products, 1)    
        
        #covariance matrix of int_t_i^t_{i+1} e^{mean_reverting_i s} dW_s:
        #time_volatility[ind_diffusion, ind_diffusion2, ind_time] is the 
        #covariance matrix between int_t_ind_time^t_{ind_time+1} 
        #e^{mean_reverting_ind_diffusion * s} dW^{ind_diffusion_s} and 
        #int_t_ind_time^t_{ind_time+1} 
        #e^{mean_reverting_ind_diffusion2 * s} dW^{ind_diffusion2_s} 

        self.time_variance = time_volatility_temp - \
                        time_volatility_temp[:,:,0].reshape(\
                                    self.nb_diffusions, self.nb_diffusions,1)

        self.exp_maturities_variance = np.exp(- (self.time_to_maturity.reshape(1, 1, \
            self.nb_dates, self.nb_products) + \
            time.reshape(1, 1, self.nb_dates, 1)) * \
            mean_reverting_sum.reshape(self.nb_diffusions, \
                                               self.nb_diffusions, 1, 1))
        
        
        self.time_variance = self.time_variance.reshape(\
                    self.nb_diffusions, self.nb_diffusions, self.nb_dates, 1)
        
        self.time_variance = self.exp_maturities_variance *  self.time_variance

        self.time_variance = (self.volatility.reshape(-1,1) * \
                self.volatility.reshape(1,-1)).reshape(self.nb_diffusions, \
                            self.nb_diffusions, 1, 1) * self.time_variance
        
        self.time_variance = self.correction_shaping.reshape(\
                self.nb_diffusions, 1, self.nb_dates, self.nb_products) * \
                self.correction_shaping.reshape(\
                1, self.nb_diffusions, self.nb_dates, self.nb_products) * \
                            self.time_variance
    def chol(self,matrix):

        return np.linalg.cholesky(matrix)

    def init_martingale_correction(self):
        self.correction_martingale = np.array([self.time_variance[ind_price * \
                    self.nb_factors:(ind_price + 1) * self.nb_factors, \
                    ind_price * self.nb_factors:(ind_price + 1) * \
                    self.nb_factors,: ,:] \
                    for ind_price in range(self.nb_prices)])

        self.correction_martingale = np.sum(self.correction_martingale, axis=1)
        self.correction_martingale = np.sum(self.correction_martingale, axis=1)
        self.correction_martingale = self.correction_martingale.reshape(\
                            self.nb_prices, self.nb_dates, self.nb_products, 1)

        self.correction_martingale = np.exp(-self.correction_martingale / 2)
        
    def init_maturities(self):
            
        self.maturities = []
        self.time_to_maturity = np.zeros((self.calendar.size,len(self.products)))
        self.delivery_periods = np.zeros((self.calendar.size,len(self.products)))
    
        for ind_product, product in enumerate(self.products):
            if product[-1].lower() == 'h' or product.lower() == 'spot':
                maturities = get_maturities_relative(\
                          pd.DataFrame(self.calendar, columns=['date']), product)
            else:
                maturities = get_maturities_absolute(product)
            
            self.delivery_periods[:,ind_product] = hours(maturities['end']-maturities['begin'])
            self.time_to_maturity[:,ind_product] = hours(maturities['begin'] - self.calendar)
            self.maturities.append(maturities)        
        
    def generate_aleas(self, nb_simulations):

        returns = np.random.multivariate_normal(\
                                    np.zeros(self.nb_diffusions), \
                                    np.eye(self.nb_diffusions), \
                                    (self.nb_dates-1, nb_simulations)).transpose(\
                                    2, 0, 1)

        alea = np.zeros((self.nb_diffusions, self.nb_dates, nb_simulations))
      
        alea[:, 1:self.nb_dates, :] = np.cumsum(\
        np.sum((self.time_volatility_squared * returns.reshape(\
        self.nb_diffusions, 1, self.nb_dates-1, nb_simulations)), axis=0), axis=1)

        return alea
    
    def get_prices(self, nb_simulations):
        diffusions = self.generate_aleas(nb_simulations)

        diffusions = self.volatility_to_multiply * \
            diffusions.reshape(\
            self.nb_diffusions, self.nb_dates, 1, nb_simulations)

        diffusions = diffusions.reshape(self.nb_prices, self.nb_factors, \
                            self.nb_dates, self.nb_products, nb_simulations) 

        diffusions = np.sum(diffusions, axis=1)

        #warning only one product
        return (np.exp(diffusions) * self.correction_martingale)[:,:,0,:] * \
            self.values_init.reshape(-1,1,1)