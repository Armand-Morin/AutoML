import os
import json
import numpy as np
import pandas as pd
from price_model import BlackScholes, NFactorsModel
from payoff import PutOption, MaxCall, StrangleSpread, PutMultiplyOption
from control import *

YEAR_IN_HOURS = 365 * 24


def read_study(folder, file):
    '''
    Return the different parameters of case study from json file
    '''
    with open(os.path.join(folder, file)) as fin:
        try:
            input_dict = json.loads(fin.read())
        except:
            raise ValueError( \
                "Problem while loading json file {} located in {}".format(file, \
                                                                          folder))
    nb_simulations_for_normalization = 100000
    parameters = {}

    # time parameters
    param_time = input_dict['my_time']['Time']
    T = param_time['maturity']
    nb_dates = param_time['nbDates'] + 1
    time = np.linspace(0, T, nb_dates)
    parameters['time'] = time

    # model parameters
    param_simulation = input_dict['my_simulator']["SimulatorBlackScholes"]
    model = read_bs_model(time, param_simulation)
    parameters['model'] = model
    param_control = list(input_dict['my_hedger'].values())[0]
    nb_test_size = param_control['nbDataForTests']
    batch_size = param_control['mini_batch_size']
    parameters['batch_size'] = batch_size
    parameters['nb_test_size'] = nb_test_size

    nb_neurons = param_control['layerSizes']
    max_actions = param_control.get('maxActions', 1)
    delay = param_control.get('delay', None)

    simulations_for_normalization = model.get_prices(nb_simulations_for_normalization)
    parameters['control'] = ControlFeedForward(time, simulations_for_normalization, delay, max_actions, nb_neurons)

    interest_rate = param_control['riskFreeRate']
    parameters['interest_rate'] = interest_rate

    # optim params
    parameters['learning_rate'] = 0.001
    parameters['nb_iterations'] = input_dict['my_trainer']['Learner']['nbIter']
    parameters['print_every'] = input_dict['my_trainer']['Learner']['printEvery']
    parameters['nb_validation_size'] = input_dict['my_predictor']['PredictorAmerican']['size']

    # payoff
    payoff_name = list(input_dict['my_put'].keys())[0]
    params_payoff = input_dict['my_put'][payoff_name]
    params_payoff.update({'r': interest_rate})

    if 'riskFactor' in params_payoff.keys():
        del params_payoff['riskFactor']
    if 'riskFactors' in params_payoff.keys():
        del params_payoff['riskFactors']
    payoff = eval(payoff_name)(**params_payoff)
    parameters['payoff'] = payoff
    return parameters


def read_bs_model(time, params):
    '''create BlackScholes model from params'''
    futures = params['riskFactors']

    volatility = np.zeros((len(futures),))
    drift = np.zeros((len(futures),))
    init = np.zeros((len(futures),))

    for ind_future, future in enumerate(futures):
        volatility[ind_future] = params["volatility"][future]
        init[ind_future] = params["initialValue"][future]
        drift[ind_future] = params["drift"][future]

    correlation = np.eye(len(futures))
    correlation_input = params.get("correlation", [])
    for correl in correlation_input:
        ind_price_1 = np.where(np.array(futures) == correl[0])
        ind_price_2 = np.where(np.array(futures) == correl[1])

        correlation[ind_price_1, ind_price_2] = correl[2]
        correlation[ind_price_2, ind_price_1] = correl[2]

    model_simulation = BlackScholes(init, \
                                    time, volatility, drift, correlation)

    return model_simulation


def read_n_factors_model(params):
    '''create NFactorsModel from params'''

    risk_factors_names = params["risk_factors"]

    init = params["initial_value"]
    init = np.array([init[key] for key in risk_factors_names])
    volatility = params["volatility"]
    volatility = np.array([volatility[key] for key in \
                           risk_factors_names]) / np.sqrt(YEAR_IN_HOURS)
    mean_reverting = params["mean_reverting"]
    mean_reverting = np.array([mean_reverting[key] for key in \
                               risk_factors_names]) / YEAR_IN_HOURS

    nb_factors = np.shape(mean_reverting)[1]
    keys_factors = list([name_market + '_' + str(i) \
                         for name_market in risk_factors_names \
                         for i in range(1, nb_factors + 1)])
    positions_factors = dict({key: i for (i, key) in \
                              enumerate(keys_factors)})

    correlation_input = params.get("correlation", [])
    nb_diffusions = np.size(mean_reverting)
    correlation = np.eye(nb_diffusions)
    for correl in correlation_input:
        correlation[positions_factors[correl[0]], \
                    positions_factors[correl[1]]] = correl[2]
        correlation[positions_factors[correl[1]], \
                    positions_factors[correl[0]]] = correl[2]

    product = params["product"]

    calendar = pd.date_range(params['time']['begin'], \
                             params['time']['end'], freq=params['time']['step'])

    model = NFactorsModel([product], calendar, \
                          volatility, mean_reverting, correlation, init)

    return model