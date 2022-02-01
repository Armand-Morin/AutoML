import torch
import numpy as np


class Payoff(object):

    def compute(self, prices, time, r):
        raise Exception('compute method in Payoff not implemented')


class PutOption(Payoff):

    def __init__(self, strike, r):
        self.strike = strike
        self.r = r


    def compute(self, prices, time):
        return torch.nn.functional.relu(self.strike - prices[:, :, 0]) *  torch.exp(-self.r * time)


class CallOption(Payoff):

    def __init__(self, strike, r):
        self.strike = strike
        self.r = r


    def compute(self, prices, time):
        return torch.nn.functional.relu(-self.strike + prices[:, :, 0]) * \
               torch.exp(-self.r * time)


class MaxCall(Payoff):

    def __init__(self, strike, r):
        self.strike = strike
        self.r = r


    def compute(self, prices, time):
        return torch.nn.functional.relu(torch.max(prices, dim=2) - self.strike) * torch.exp(-self.r * time)


class PutMultiplyOption(Payoff):

    def __init__(self, strike, r):
        self.strike = strike
        self.r = r


    def compute(self, prices, time):
        payoff_right = prices[:, :, 0]

        for ind_price in range(1, prices.shape[2]):
            payoff_right *= prices[:, :, ind_price]

        payoff = torch.nn.functional.relu(self.strike - payoff_right)

        return payoff * torch.exp(-self.r * time)


class StrangleSpread(Payoff):

    def __init__(self, strikes, r):
        self.strikes = np.array(strikes).astype(float)
        self.r = r

    def compute(self, prices, time):
        mean_asset = prices[:, :, 0]

        for ind_price in range(1, prices.shape[2]):
            mean_asset = mean_asset + prices[:, :, ind_price]

        mean_asset = mean_asset / 5.0

        payoff = -torch.nn.functional.relu(self.strikes[0] - mean_asset)
        payoff = payoff + torch.nn.functional.relu(self.strikes[1] - mean_asset)
        payoff = payoff + torch.nn.functional.relu(mean_asset - self.strikes[2])
        payoff = payoff - torch.nn.functional.relu(mean_asset - self.strikes[3])
        return payoff * torch.exp(-self.r * time)

