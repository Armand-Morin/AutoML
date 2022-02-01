import numpy as np
from datetime import timedelta
import pandas as pd
import re
from scipy.stats import norm


MONTH_TO_INT = {'jan': 1,
                'feb': 2,
                'mar': 3,
                'apr': 4,
                'may': 5,
                'jun': 6,
                'jul': 7,
                'aug': 8,
                'sep': 9,
                'oct': 10,
                'nov': 11,
                'dec': 12}


def hours(delta):
    '''param delta: a timedelta (i.e. a difference between two dates)
    return delta converted in hours
    '''
    return (delta / np.timedelta64(1, 'h'))


def get_maturities_relative(calendar, product_name):
    '''param calendar: pd DataFrame with columns date containing dates
    param product_name: string of the form 'nxAH' with n a int and x in
    {D, W, Q, Y} or equal to 'Spot'
    return a pd DataFrame with column begin and end corresponding to the begin
    and end of delivery
    '''
    maturities = pd.DataFrame()

    if (product_name == 'Spot'):
        maturities['begin'] = calendar['date'] + timedelta(hours=0)
        maturities['end'] = calendar['date'] + timedelta(hours=0)
        return maturities

    # n is integer present in the name of the product
    n = int(re.findall('\d+', product_name)[0])

    # size of n and then index of the first letter of the product
    begin = int(np.log10(n) + 1)

    if product_name[begin] == 'W':
        maturities['begin'] = calendar['date'] + pd.offsets.Week(n, weekday=0)
        maturities['end'] = calendar['date'] + pd.offsets.Week(n + 1, weekday=0)

    if product_name[begin] == 'M':
        maturities['begin'] = calendar['date'] + \
                              pd.offsets.MonthBegin(n)
        maturities['end'] = calendar['date'] + \
                            pd.offsets.MonthBegin(n + 1)

    if product_name[begin] == 'Q':
        maturities['begin'] = calendar['date'] + \
                              pd.offsets.QuarterEnd(n) + pd.offsets.Day(1)
        maturities['end'] = calendar['date'] + \
                            pd.offsets.QuarterEnd(n + 1) + pd.offsets.Day(1)

    if product_name[begin] == 'Y':
        maturities['begin'] = calendar['date'] + \
                              pd.offsets.YearBegin(n)
        maturities['end'] = calendar['date'] + \
                            pd.offsets.YearBegin(n + 1)

    if product_name[begin] == 'D':
        maturities['begin'] = calendar['date'] + \
                              pd.offsets.Day(n)
        maturities['end'] = calendar['date'] + \
                            pd.offsets.Day(n + 1)

    maturities['begin'] = maturities['begin'].apply(lambda x: x.replace(hour=0))
    maturities['end'] = maturities['end'].apply(lambda x: x.replace(hour=0))

    return maturities


def get_maturities_absolute(product_name):
    '''param product_name: string which is can be of the form 
    -'year' with year an int for yearly products,
    -'monthyear' with month the name of the month in english, year an int, for
    monthly products
    -'Qnyear' with n an int between 1 and 4 and year an int for the quarter 
    product
    return two datetimes corresponding to the begining and end of delivery of 
    the product
    '''
    year_begin = int(product_name[-4:])
    day = 1
    if len(product_name) == 4:
        month_begin = 1
        month_end = 1
        year_end = year_begin + 1
    elif product_name[0] == 'Q':
        if product_name[1] == '1':
            month_begin = 1
            month_end = 4
            year_end = year_begin
        elif product_name[1] == '2':
            month_begin = 4
            month_begin = 7
            year_end = year_begin
        elif product_name[1] == '3':
            month_begin = 7
            month_end = 10
            year_end = year_begin
        elif product_name[1] == '4':
            month_begin = 10
            month_end = 1
            year_end = year_begin + 1
        else:
            raise Exception('Q' + product_name[1] + 'does not exist. ' + \
                            'Only Q1, Q2, Q3 and Q4 exist')
    elif product_name[:3].lower() in list(MONTH_TO_INT.keys()):
        month_begin = MONTH_TO_INT[product_name[:3].lower()]
        month_end = 1 if month_begin == 12 else month_begin + 1
        year_end = year_begin + 1 if month_begin == 12 else year_begin
    else:
        raise Exception('Product does not exist')
    return {'begin': pd.to_datetime(str(day) + '-' + str(month_begin) + '-' + \
                                    str(year_begin), format='%d-%m-%Y'), \
            'end': pd.to_datetime(str(day) + '-' + str(month_end) + '-' + \
                                  str(year_end), format='%d-%m-%Y')}




def integral_exp(dates1, dates2, alpha):
    '''
    date1 : numpy array
    date2 : numpy array
    alpha : int
    return: integral of exp(-alpha*t) between dates1 and dates2
    '''

    ind_alpha_null = alpha == 0
    integral = np.zeros(alpha.shape)
    integral[np.logical_not(ind_alpha_null)] = ( \
                        np.exp(- alpha[np.logical_not(ind_alpha_null)] * dates1) - \
                        np.exp(-alpha[np.logical_not(ind_alpha_null)] * dates2)) / \
                                               alpha[np.logical_not(ind_alpha_null)]
    integral[ind_alpha_null] = dates2 - dates1

    return integral



def spread_option_kirk_2f_3m(init, delivery_period, time, maturity, \
                    mean_reverting, volatility, correl, maturity_option, strike=0):
    volatility_shaped = volatility * \
        integral_exp(0, 1, mean_reverting * delivery_period)
    time_to_maturity = maturity - time
    
    variance_1 = np.sum(correl[:2,:2] * volatility_shaped[0,:].reshape(-1,1) * \
        volatility_shaped[0,:].reshape(1,-1) * integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[0,:].reshape(-1,1) + mean_reverting[0,:].reshape(1,-1))) 

    variance_2 = np.sum(correl[2:4,2:4] * volatility_shaped[1,:].reshape(-1,1) * \
        volatility_shaped[1,:].reshape(1,-1) * integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[1,:].reshape(-1,1) + mean_reverting[1,:].reshape(1,-1))) 

    variance_3 = np.sum(correl[4:,4:] * volatility_shaped[2,:].reshape(-1,1) * \
        volatility_shaped[2,:].reshape(1,-1) * integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[2,:].reshape(-1,1) + mean_reverting[2,:].reshape(1,-1)))

    variance_23 = np.sum(correl[2:4,4:] * volatility_shaped[1,:].reshape(-1,1) * \
        volatility_shaped[2,:].reshape(1,-1) * \
            integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[1,:].reshape(-1,1) + mean_reverting[2,:].reshape(1,-1)))
    
    variance_12 = np.sum(correl[:2,2:4] * volatility_shaped[0,:].reshape(-1,1) * \
        volatility_shaped[1,:].reshape(1,-1) * \
            integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[0,:].reshape(-1,1) + mean_reverting[1,:].reshape(1,-1)))
    
    variance_13 = np.sum(correl[:2,4:] * volatility_shaped[0,:].reshape(-1,1) * \
        volatility_shaped[2,:].reshape(1,-1) * \
            integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[0,:].reshape(-1,1) + mean_reverting[2,:].reshape(1,-1)))

    strike_eq = strike + init[1] + init[2]
    
    variance_eq = np.log(1 + (init[1]**2*(np.exp(variance_2)-1) + \
        init[2]**2*(np.exp(variance_3)-1) + 2 * init[1] * init[2] * \
            (np.exp(variance_23) - 1))/ strike_eq**2)                       
    cov_eq = np.log((strike + init[1] * np.exp(variance_12) + init[2] * \
                     np.exp(variance_13))/strike_eq)
  
    sigma2 = variance_1 + variance_eq -2 * cov_eq
    d1 = (np.log(init[0] / strike_eq) + sigma2 / 2) / np.sqrt(sigma2)
    d2 = d1 - np.sqrt(sigma2)
    return init[0] * norm.cdf(d1) - strike_eq * norm.cdf(d2)



def delta_option_kirk_2f_3m(init, delivery_period, time, maturity, \
                    mean_reverting, volatility, correl, maturity_option, strike=0):
    
    volatility_shaped = volatility * \
        integral_exp(0, 1, mean_reverting * delivery_period)
    time_to_maturity = maturity - time
        
    variance_1 = np.sum(correl[:2,:2] * volatility_shaped[0,:].reshape(-1,1) * \
        volatility_shaped[0,:].reshape(1,-1) * integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[0,:].reshape(-1,1) + mean_reverting[0,:].reshape(1,-1))) 

    variance_2 = np.sum(correl[2:4,2:4] * volatility_shaped[1,:].reshape(-1,1) * \
        volatility_shaped[1,:].reshape(1,-1) * integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[1,:].reshape(-1,1) + mean_reverting[1,:].reshape(1,-1))) 

    variance_3 = np.sum(correl[4:,4:] * volatility_shaped[2,:].reshape(-1,1) * \
        volatility_shaped[2,:].reshape(1,-1) * integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[2,:].reshape(-1,1) + mean_reverting[2,:].reshape(1,-1)))

    variance_23 = np.sum(correl[2:4,4:] * volatility_shaped[1,:].reshape(-1,1) * \
        volatility_shaped[2,:].reshape(1,-1) * \
            integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[1,:].reshape(-1,1) + mean_reverting[2,:].reshape(1,-1)))
    
    variance_12 = np.sum(correl[:2,2:4] * volatility_shaped[0,:].reshape(-1,1) * \
        volatility_shaped[1,:].reshape(1,-1) * \
            integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[0,:].reshape(-1,1) + mean_reverting[1,:].reshape(1,-1)))
    
    variance_13 = np.sum(correl[:2,4:] * volatility_shaped[0,:].reshape(-1,1) * \
        volatility_shaped[2,:].reshape(1,-1) * \
            integral_exp(maturity - maturity_option, time_to_maturity, \
        mean_reverting[0,:].reshape(-1,1) + mean_reverting[2,:].reshape(1,-1)))

    strike_eq = strike + init[1] + init[2]
    
    variance_eq = np.log(1 + (init[1]**2*(np.exp(variance_2)-1) + \
        init[2]**2*(np.exp(variance_3)-1) + 2 * init[1] * init[2] * \
            (np.exp(variance_23) - 1))/ strike_eq**2)                       
    cov_eq = np.log((strike + init[1] * np.exp(variance_12) + init[2] * \
                     np.exp(variance_13))/strike_eq)
  
    sigma2 = variance_1 + variance_eq -2 * cov_eq
    d1 = (np.log(init[0] / strike_eq) + sigma2 / 2) / np.sqrt(sigma2)
    d2 = d1 - np.sqrt(sigma2)

    return np.concatenate((np.expand_dims(norm.cdf(d1), axis=0), \
                          np.expand_dims(-norm.cdf(d2), axis=0), \
                              np.expand_dims(- norm.cdf(d2),axis=0)),axis=0)