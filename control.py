import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
import torch
import torch.nn.functional as F
import torch.distributions as torchp
import numpy as np
from nni.retiarii import model_wrapper
from aleat import model_quant



depth=4

@model_wrapper
class ControlFeedForward(nn.Module):

    def __init__(self, time, simulations_for_normalisation, delay, max_actions, nb_neurons):
        super().__init__()
        self.time = time
        self.create_normalisation_dict(simulations_for_normalisation)
        self.delay = delay
        self.max_actions = max_actions
        self.nb_neurons = nb_neurons
        #self.create_neural_network(delay)

        self._ops = nn.ModuleList()

        self.in_size = 3
        if delay is not None:
            self.in_size = 4

        L=genrateurneurones(depth)
        self._ops.append(aleat(self.in_size ,L[0],bias=True))

        for x in range(1,depth):
            op = aleat(L[x-1] , L[x] , bias=True)
            self._ops.append(op)

        self._ops.append(aleat(L[-1] ,1,bias=True))

    def create_normalisation_dict(self, simulations_for_normalisation):
        normalisation = dict()
        normalisation['mean'] = np.mean(simulations_for_normalisation, axis=(1, 2)).astype('float32')
        normalisation['std'] = np.mean(np.std(simulations_for_normalisation, axis=2), axis=1).astype('float32')
        normalisation['mean'] = torch.from_numpy(normalisation['mean'])
        normalisation['mean'] = torch.unsqueeze(normalisation['mean'], dim=0)
        normalisation['mean'] = torch.unsqueeze(normalisation['mean'], dim=0)
        normalisation['std'] = torch.from_numpy(normalisation['std'])
        normalisation['std'] = torch.unsqueeze(normalisation['std'], dim=0)
        normalisation['std'] = torch.unsqueeze(normalisation['std'], dim=0)

        self.normalisation_dict = normalisation

    def forward(self , x):
        for f in self._ops:
            x = f(x)
        return x


    def get_output(self, ind_time, prices_normed, time_normed, sum_actions, delay):

        state = torch.cat((torch.unsqueeze(time_normed[:, ind_time], dim=1), prices_normed[:, ind_time, :]), dim=1)
        if self.max_actions is not None:
            state = torch.cat((state, sum_actions),dim=1)
        if self.delay is not None:
            state = torch.cat((state, delay), dim=1)

        for f in self._ops:
            state = f(state)

        return state

    def compute_action(self, prices, time, is_train):
        #positions = torch.TensorArray(torch.float32, size=len(self.time[:-1]))
        actions = []
        sum_actions = torch.zeros((prices.shape[0], 1))
        log_softmax = []
        proba_list = []
        if self.delay is not None:
            delays = []
            delay_time = torch.ones((prices.shape[0], 1)) * self.delay
        else: 
            delay_time = None
        time_normed = time / self.time[-1] 

        prices_normed = (prices - self.normalisation_dict['mean']) / \
            self.normalisation_dict['std']
        
        for ind_time in range(len(self.time)):
            print(ind_time)
            output = self.get_output(ind_time, prices_normed, time_normed, sum_actions, delay_time)
            #output = torch.minimum(output, 4)
            #output = torch.maximum(output, -4)
            output = 10 * torch.tanh(output)
            #temperature = torch.cond(is_train, lambda: 3.0, lambda: 1.0)
            #output /= temperature      
            c = torch.ones((prices.shape[0],1))
                
            if self.max_actions is not None:
                c = c *  \
                    (1-(
                        torch.ge(sum_actions, self.max_actions-0.1)
                    ).type(torch.FloatTensor))
            if self.delay is not None:
                c = c * (1-(
                    torch.le(delay_time, self.delay-self.time[1]/2)
                ).type(torch.FloatTensor))
            prob_value_1 = torch.clamp(c/(1+torch.exp(-output)), min=1e-12)
            
            #prob = torchp.distributions.Bernoulli(logits=output)
            prob = torchp.bernoulli.Bernoulli(probs=prob_value_1)

            action = torch.where(is_train, (prob.sample()).type(torch.FloatTensor), (torch.gt(prob_value_1, 0.5)).type(torch.FloatTensor))
            
            #action = action * c
            proba_list.append(prob_value_1[:,0])
            
        
            actions.append(action[:,0])
            log_softmax.append(prob.log_prob(action)[:,0])

            sum_actions += action
            if self.delay is not None:
                delays.append(delay_time[:,0])
                if ind_time < prices.shape[1] - 1:
                    delay_time = self.delay + time[:,ind_time + 1] - \
                    torch.sum(torch.stack(actions, dim=-1) * torch.stack(delays, dim=-1) , dim=1)
                    delay_time = torch.unsqueeze(delay_time, dim=-1)
    
        actions = torch.stack(actions, dim = -1)
        log_softmax = torch.stack(log_softmax, dim = -1)
        proba_list = torch.stack(proba_list, dim = -1)
        
        if self.delay is not None: 
            delays = torch.stack(delays, dim=-1)
        else:
            delays = None
    
        return actions, log_softmax, delays, proba_list

class aleat(nn.Module):
    def __init__(self , dim_in , dim_out , bias=True):
        super().__init__()
        self.Activation = [nn.ReLU(),nn.LeakyReLU(),nn.Softplus(),nn.Tanh(),nn.Sigmoid()]
        self.linear = nn.Linear(dim_in , dim_out , bias=bias)
        self.activation = nn.LayerChoice(self.Activation)
    
    def forward(self , x):
        out = self.linear(x)
        out = self.activation(out)
        return out


def genrateurneurones(depth):
    # fonction qui renvoi la liste du nombre de neurones choisis aleatoirement pour chaque couche
    L=[]
    for i in range(depth):
        L.append(nn.ValueChoice([16,32,64,128,256]))
    return L
