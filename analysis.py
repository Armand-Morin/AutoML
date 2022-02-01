import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from read_input import read_study
########################American#############################################
def main_analysis():
    output_folder = './output_ref/'
    studies = ['bermudean', 'maxcall2', 'maxcall10', 'strangle']

    losses_ref = {'maxcall10': 38.278, \
               'maxcall2': 13.899, \
               'strangle': 11.794,\
               'bermudean': 0.06031}
    name_studies = {'maxcall10': 'Max-call, d = 10', \
                    'maxcall2': 'Max-call, d = 2',\
                    'bermudean': 'Bermudean put', \
                    'strangle': 'Strangle spread'}

    losses = {}
    error = {}
    times = {}
    for study in studies:
        results = pd.read_csv(output_folder + study + '/results.csv', sep=';')
        losses[study] = results['price'].values[0]
        error[study] = np.abs((losses[study]-losses_ref[study])/ losses_ref[study])
        times[study] = round(results['time'].values[0], 1)

    round_n = 4
    results_df = pd.DataFrame()
    results_df['Use case / Method'] = [name_studies[study] for study in studies]

    results_df['Algorithm \ref{algo:algoGlobal}'] = [round(losses[study], round_n) \
              for study in studies]

    results_df['Reference'] = [np.around(losses_ref[study],round_n) for \
              study in studies]
    results_df['Difference'] = [str(np.around(100 * error[study],2)) + '\%' for \
              study in studies]

    results_df['Time (s)'] = [times[study] for \
              study in studies]
    print(results_df.to_latex(index=False, escape=False))

    #plot learning curve
    for study in studies:
        learning_curve = pd.read_csv(output_folder + \
                                study + '/learning_curve.csv', sep=';')
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.plot(learning_curve['iter'], learning_curve['loss_train'], label='loss train')
        ax.plot(learning_curve['iter'], learning_curve['loss_test'], label='loss test')
        ax.plot(learning_curve['iter'], \
                -np.ones((len(learning_curve['iter']),))*losses_ref[study],\
                    label='reference value')
        ax.grid()
        ax.legend(loc='best', fontsize=20)
        ax.set_xlabel('Number of iterations', fontsize=25)
        ax.set_ylabel('Loss', fontsize=25)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        fig.savefig(output_folder + study + '/learningcurve_' + study + '.png',\
                    bbox_inches='tight')
        plt.show()
        plt.close()


    #########################Bermudean analysis
    parameters = read_study('./input', 'bermudean.json')
    control = parameters['control']
    control.load_weights(output_folder + 'bermudean/' + 'optimal_weights')
    range_x = np.linspace(0.6, 1.4, 1000)
    times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    proba = np.zeros((len(range_x), len(times)))
    fig, ax = plt.subplots(figsize=(20, 12))
    for ind_t, t in enumerate(times):
        state = (range_x.reshape(-1,1) - control.normalisation_dict['mean'].numpy()[0,0,0]) \
            / control.normalisation_dict['std'].numpy()[0,0,0]
        state = np.concatenate((t*np.ones((len(range_x), 1)), state), axis=1)
        state = np.concatenate((state, np.zeros((len(range_x), 1))), axis=1)
        output = 10 * np.tanh(control.nn(state))
        proba[:,ind_t] = 1/(1+np.exp(-output[:,0]))
        ax.plot(range_x, proba[:,ind_t], label='time to maturity = ' + str(np.round(1-t,1)))

    ax.grid()
    ax.legend(loc='best', fontsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    ax.set_xlabel('Price', fontsize=25)
    ax.set_ylabel('Probability of exercise', fontsize=25)
    fig.savefig(output_folder + 'bermudean/proba_exercise.png', bbox_inches='tight')
    plt.show()
    plt.close()

    range_t = np.linspace(0, 1, 1000)
    price_limit = np.zeros((len(range_t), ))
    for ind_t, t in enumerate(range_t):
        state = (range_x.reshape(-1,1) - control.normalisation_dict['mean'].numpy()[0,0,0]) \
            / control.normalisation_dict['std'].numpy()[0,0,0]
        state = np.concatenate((t*np.ones((len(range_x), 1)), state), axis=1)
        state = np.concatenate((state, np.zeros((len(range_x), 1))), axis=1)
        output = 10 * np.tanh(control.nn(state))
        proba = 1/(1+np.exp(-output[:,0]))
        p_l = range_x[np.where(proba < 0.5)[0][0]]
        price_limit[ind_t] = p_l
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.plot((1-range_t), price_limit, label='Neural network')
    ax.grid()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    ax.set_ylabel('Price', fontsize=25)
    ax.set_xlabel('Time to maturity', fontsize=25)
    fig.savefig(output_folder + 'bermudean/region_exercise.png', bbox_inches='tight')
    plt.show()
    plt.close()

    ##########################Swing Ibanez#######################################
    strikes = [35, 40, 45]
    nb_exercises = range(1, 7)
    round_n = 3
    ibanez = {35: [5.114, 10.195, 15.230, 20.230, 25.200, 30.121], \
              40: [1.774, 3.480, 5.111, 6.661, 8.124, 9.502], \
              45: [0.411, 0.772, 1.089, 1.358, 1.582, 1.756]}

    nn = dict()
    txt_dict = dict()
    error_dict = dict()
    times_dict = dict()
    for strike in strikes:
        price = []
        error = []
        txt = []
        times = []
        for ind_ex, ex in enumerate(nb_exercises):
            results = pd.read_csv(output_folder + 'swingIbanez' + str(strike) + \
                                  str(ex) + '/results.csv', sep=';')
            price.append(results['price'].values[0])
            error.append(np.around(100 * np.abs((price[-1] - ibanez[strike][ind_ex])/ \
                                ibanez[strike][ind_ex]),2))
            txt.append('(' + str(round(price[-1], round_n)) + ', ' + \
                         str(round(ibanez[strike][ind_ex], round_n)) + ', ' + \
                         str(round(error[-1], round_n)) + '\%)')
            times.append(round(results['time'].values[0],1))
        nn[strike] = price
        times_dict[strike] = times
        error_dict[strike] = error
        txt_dict[strike] = txt

    results_df = pd.DataFrame()
    times_df = pd.DataFrame()
    results_df['$l$ / $S_0$'] = nb_exercises
    times_df['$l$ / $S_0$'] = nb_exercises

    for strike in strikes:
        results_df[strike] = txt_dict[strike]
        times_df[strike] = times_dict[strike]

    print(results_df.to_latex(index=False, escape=False))
    print(times_df.to_latex(index=False, escape=False))


    parameters = read_study('./input', 'swingIbanez406.json')
    control = parameters['control']
    control.load_weights(output_folder + 'swingIbanez406/' + 'optimal_weights')
    range_x = np.linspace(30, 50, 1000)
    time = 0.5
    range_l = [0,1,2,3,4,5]
    proba = np.zeros((len(range_x), len(range_l)))
    fig, ax = plt.subplots(figsize=(20, 12))
    for ind_l, l in enumerate(range_l):
        state = (range_x.reshape(-1,1) - control.normalisation_dict['mean'].numpy()[0,0,0]) \
            / control.normalisation_dict['std'].numpy()[0,0,0]
        state = np.concatenate((time*np.ones((len(range_x), 1)), state), axis=1)
        state = np.concatenate((state, l*np.ones((len(range_x), 1))), axis=1)
        output = 10 * np.tanh(control.nn(state))
        proba[:,ind_l] = 1/(1+np.exp(-output[:,0]))
        ax.plot(range_x, proba[:,ind_l], label='Remaining exercises = ' + str(6-l))

    ax.grid()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel('Price', fontsize=25)
    ax.set_ylabel('Probability of exercise', fontsize=25)
    fig.savefig(output_folder + 'swingIbanez406/proba_exercise_swing.png', bbox_inches='tight')
    plt.show()
    plt.close()

    range_t = np.linspace(0, 1, 1000)
    range_l = [0,1,2,3,4,5]
    price_limit = np.zeros((len(range_t), len(range_l)))
    for ind_l, l in enumerate(range_l):
        for ind_t, t in enumerate(range_t):
            state = (range_x.reshape(-1,1) - control.normalisation_dict['mean'].numpy()[0,0,0]) \
            / control.normalisation_dict['std'].numpy()[0,0,0]
            state = np.concatenate((t*np.ones((len(range_x), 1)), state), axis=1)
            state = np.concatenate((state, l*np.ones((len(range_x), 1))), axis=1)
            output = 10 * np.tanh(control.nn(state))
            proba = 1/(1+np.exp(-output[:,0]))
            p_l = range_x[np.where(proba < 0.5)[0][0]]
            price_limit[ind_t,ind_l] = p_l
    fig, ax = plt.subplots(figsize=(20, 12))
    maturity = control.time[-1]
    time_true = np.linspace(0, maturity, 1000)
    for ind_l, l in enumerate(range_l):
        ax.plot(maturity-time_true, price_limit[:,ind_l], label='Remaining exercises = ' + str(6-l))
    ax.grid()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    ax.set_ylabel('Price', fontsize=25)
    ax.set_xlabel('Time to maturity', fontsize=25)
    ax.legend(loc='best', fontsize=20)

    fig.savefig(output_folder + 'swingIbanez406/region_exercise_swing.png', bbox_inches='tight')
    plt.show()
    plt.close()
    #########################Swing Ibanez 5d######################################
    nn = dict()
    txt_dict = dict()
    error_dict = dict()
    times_dict = dict()
    n_exercise = 6
    price = []
    error = []
    txt = []
    times = []
    for ex in range(1, n_exercise+1):
        results = pd.read_csv(output_folder + 'swingIbanez40'  + \
                              str(ex) + '5d/results.csv', sep=';')
        price.append(results['price'].values[0])
        error.append(np.abs((price[-1] - ibanez[40][ex-1])/ \
                            ibanez[40][ex-1]))
        times.append(round(results['time'].values[0],1))

    results_df = pd.DataFrame()

    round_n = 3
    results_df = pd.DataFrame()
    results_df['Use case / Method'] = ['l = ' + str(l) for l in \
              range(1, n_exercise+1)]

    results_df['Algorithm \ref{algo:algoGlobal}'] = [round(price[l], round_n) \
              for l in range(n_exercise)]

    results_df['Reference'] = [round(ibanez[40][l], round_n) \
              for l in range(n_exercise)]

    results_df['Difference'] = [str(np.around(100 * error[l],2)) + '\%' for \
              l in range(n_exercise)]

    results_df['Time (s)'] = [times[l] for \
              l in range(n_exercise)]

    print(results_df.to_latex(index=False, escape=False))

    #########################Swing Ibanez 5d light######################################
    nn = dict()
    txt_dict = dict()
    error_dict = dict()
    times_dict = dict()
    n_exercise = 6
    price = []
    error = []
    txt = []
    times = []
    for ex in range(1, n_exercise+1):
        results = pd.read_csv(output_folder + 'swingIbanez40'  + \
                              str(ex) + '5dlight/results.csv', sep=';')
        price.append(results['price'].values[0])
        error.append(np.abs((price[-1] - ibanez[40][ex-1])/ \
                            ibanez[40][ex-1]))
        times.append(round(results['time'].values[0],1))

    results_df = pd.DataFrame()

    round_n = 3
    results_df = pd.DataFrame()
    results_df['Use case / Method'] = ['l = ' + str(l) for l in \
              range(1, n_exercise+1)]

    results_df['Algorithm \ref{algo:algoGlobal}'] = [round(price[l], round_n) \
              for l in range(n_exercise)]

    results_df['Reference'] = [round(ibanez[40][l], round_n) \
              for l in range(n_exercise)]

    results_df['Difference'] = [str(np.around(100 * error[l],2)) + '\%' for \
              l in range(n_exercise)]

    results_df['Time (s)'] = [times[l] for \
              l in range(n_exercise)]

    print(results_df.to_latex(index=False, escape=False))


    #############################Swing Carmona#####################################
    carmona_prices = {1: 9.85, \
               2: 19.26, \
               3: 28.80,\
               4: 38.48,
               5: 48.32}

    prices = {}
    error = {}
    times = {}
    range_actions = range(1, 6)
    for l in range_actions:
        results = pd.read_csv(output_folder + 'swingCarmona' + str(l) + '/results.csv', sep=';')
        prices[l] = results['price'].values[0]
        error[l] = np.abs((prices[l]-carmona_prices[l])/ carmona_prices[l])
        times[l] = round(results['time'].values[0], 1)
    #Write results in latex
    round_n = 3
    results_df = pd.DataFrame()
    results_df['Use case / Method'] = ['l = ' + str(l) for l in range_actions]

    results_df['Algorithm \ref{algo:algoGlobal}'] = [round(prices[l], round_n) \
              for l in range_actions]

    results_df['Reference'] = [round(carmona_prices[l], round_n) \
              for l in range_actions]

    results_df['Difference'] = [str(np.around(100 * error[l],2)) + '\%' for \
              l in range_actions]

    results_df['Time (s)'] = [times[l] for \
              l in range_actions]
    print(results_df.to_latex(index=False, escape=False))



    parameters = read_study('./input', 'swingCarmona5.json')
    control = parameters['control']
    control.load_weights(output_folder + 'swingCarmona5/' + 'optimal_weights')
    range_x = np.linspace(70, 120, 1000)
    time = 0.5
    range_l = [0,1,2,3,4]
    proba = np.zeros((len(range_x), len(range_l)))
    fig, ax = plt.subplots(figsize=(20, 12))
    for ind_l, l in enumerate(range_l):
        state = (range_x.reshape(-1,1) - control.normalisation_dict['mean'].numpy()[0,0,0]) \
            / control.normalisation_dict['std'].numpy()[0,0,0]
        state = np.concatenate((time*np.ones((len(range_x), 1)), state), axis=1)
        state = np.concatenate((state, l*np.ones((len(range_x), 1))), axis=1)
        state = np.concatenate((state, 0.1*np.ones((len(range_x), 1))), axis=1)

        output = 10 * np.tanh(control.nn(state))
        proba[:,ind_l] = 1/(1+np.exp(-output[:,0]))
        ax.plot(range_x, proba[:,ind_l], label='Remaining exercises = ' + str(6-l))
    ax.grid()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel('Price', fontsize=25)
    ax.set_ylabel('Probability of exercise', fontsize=25)
    fig.savefig(output_folder + 'swingCarmona5/proba_exercise_swingdelay.png', bbox_inches='tight')
    plt.show()
    plt.close()

    range_t = np.linspace(0, 1, 1000)
    range_l = [0,1,2,3,4,5]
    price_limit = np.zeros((len(range_t), len(range_l)))
    for ind_l, l in enumerate(range_l):
        for ind_t, t in enumerate(range_t):
            state = (range_x.reshape(-1,1) - control.normalisation_dict['mean'].numpy()[0,0,0]) \
            / control.normalisation_dict['std'].numpy()[0,0,0]
            state = np.concatenate((t*np.ones((len(range_x), 1)), state), axis=1)
            state = np.concatenate((state, l*np.ones((len(range_x), 1))), axis=1)
            state = np.concatenate((state, 0.1*np.ones((len(range_x), 1))), axis=1)
            output = 10 * np.tanh(control.nn(state))
            proba = 1/(1+np.exp(-output[:,0]))
            p_l = range_x[np.where(proba < 0.5)[0][0]]
            price_limit[ind_t,ind_l] = p_l

    fig, ax = plt.subplots(figsize=(20, 12))
    maturity = control.time[-1]
    time_true = np.linspace(0, maturity, 1000)
    for ind_l, l in enumerate(range_l):
        ax.plot(maturity-time_true, price_limit[:,ind_l], label='Remaining exercises = ' + str(6-l))
    ax.grid()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    ax.set_ylabel('Price', fontsize=25)
    ax.set_xlabel('Time to maturity', fontsize=25)
    ax.legend(loc='best', fontsize=20)
    fig.savefig(output_folder + 'swingCarmona5/region_exercise.png', bbox_inches='tight')
    plt.show()
    plt.close()


    time = 0.5
    l = 2
    range_delay = [0.1,0.2,0.3,0.4]
    proba = np.zeros((len(range_x), len(range_delay)))
    fig, ax = plt.subplots(figsize=(20, 12))
    for ind_delay, delay in enumerate(range_delay):
        state = (range_x.reshape(-1,1) - control.normalisation_dict['mean'].numpy()[0,0,0]) \
            / control.normalisation_dict['std'].numpy()[0,0,0]
        state = np.concatenate((time*np.ones((len(range_x), 1)), state), axis=1)
        state = np.concatenate((state, l*np.ones((len(range_x), 1))), axis=1)
        state = np.concatenate((state, delay*np.ones((len(range_x), 1))), axis=1)
        output = 10 * np.tanh(control.nn(state))
        proba[:,ind_delay] = 1/(1+np.exp(-output[:,0]))
        ax.plot(range_x, proba[:,ind_delay], label='Delay since last exercise = ' +\
                 str(delay))

    ax.grid()
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel('Price', fontsize=25)
    ax.set_ylabel('Probability of exercise', fontsize=25)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    fig.savefig(output_folder + 'swingCarmona5/proba_exercise_delay.png', bbox_inches='tight')
    plt.show()
    plt.close()

