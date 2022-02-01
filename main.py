from read_input import read_study
import os
import numpy as np
import time as t
import pandas as pd
import torch
import json
import logging
import nni.retiarii.nn.pytorch as nn
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from dartsPerso import OurDartsTrainer
from control import ControlFeedForward
import nni.retiarii.strategy as strategy

input_folder = './input/'
input_file = 'swingIbanez351.json'
name_study = input_file.split('.')[0]
output_folder = './outputTemp/' + name_study + '/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

parameters = read_study(input_folder, input_file)

price_model = parameters['model']
control = parameters['control']
payoff = parameters['payoff']
calendar = parameters['time']
batch_size = parameters['batch_size']
nb_iterations = parameters['nb_iterations']
nb_test_size = parameters['nb_test_size']
nb_validation_size = parameters['nb_validation_size']
learning_rate = parameters['learning_rate']
print_every = parameters['print_every']

optimizer = torch.optim.Adam(control.parameters(), lr=learning_rate)

results_train_test = []
t_init = t.time()


def generator(simulator, calendar, batch_size=1):
    """
    Yields the next training batch.
    """
    while True:
        x_train = simulator.get_prices(batch_size)
        time_vec = calendar.reshape(1, -1, 1) * \
                   np.ones((1, 1, batch_size))
        x_train = np.concatenate((x_train, time_vec), axis=0)
        yield x_train.transpose((2, 1, 0)).astype('float32')


def train_step(simulations):
    prices = simulations[:, :, :-1]
    time = simulations[:, :, -1]

    actions, log_softmax, delays, proba_list = control.compute_action(prices, time, torch.tensor([True]))

    payoff_batch = payoff.compute(prices, time)
    cost_batch = payoff_batch * actions
    # reward_stopped = tf.reduce_sum(tf.stop_gradient(-cost_batch) * tf.cumsum(log_softmax,axis=1), axis=1)
    reward_stopped = torch.sum((-cost_batch).detach(), dim=1) * torch.sum(log_softmax, dim=1)

    loss_actions = torch.mean(reward_stopped)

    optimizer.zero_grad()
    loss_actions.backward()
    optimizer.step()

    return -torch.mean(torch.sum(cost_batch, dim=1))


def test_step(simulations):
    prices = simulations[:, :, :-1]
    time = simulations[:, :, -1]
    actions, log_softmax, delays, p_l = control.compute_action(prices, time, torch.tensor([False]))
    payoff_batch = payoff.compute(prices, time)
    loss = -torch.mean(torch.sum(actions * payoff_batch, dim=1))

    return actions, log_softmax, delays, loss, actions * payoff_batch, p_l

def loss_function(logits, simulations, stopped=False):
    prices = simulations[:, :, :-1]
    time = simulations[:, :, -1]
    actions, log_softmax, delays, proba_list = logits
    payoff_batch = payoff.compute(prices, time)
    cost_batch = payoff_batch * actions
    if stopped:
        loss = -torch.mean(torch.sum(cost_batch, dim=1))
    else:
        reward_stopped = torch.sum((-cost_batch).detach(), dim=1) * torch.sum(log_softmax, dim=1)
        loss = torch.mean(reward_stopped)
    return loss

test_data = next(generator(price_model, calendar, nb_test_size))
test_data = torch.from_numpy(test_data)

loss_max = np.inf


if __name__ == "__main__":
    model = control
    batch = torch.Tensor(next(generator(price_model, calendar, batch_size)))
    loss_batch = train_step(batch)
    loss = loss_batch
    validation_data = next(generator(price_model, calendar, nb_validation_size))
    dataset = next(generator(price_model, calendar, batch_size + nb_validation_size))
    # metrics=lambda output, target: accuracy(output, target, topk=(1,)),
    trainer = OurDartsTrainer(model=model,
                              loss=loss_function,
                              metrics=lambda output, target: [],
                              optimizer=optimizer,
                              num_epochs=3,
                              dataset=torch.Tensor(test_data),
                              nb_iter=nb_iterations,
                              price_model=price_model,
                              generator=generator,
                              batch_size=1,
                              calendar=calendar,
                              child_batch_size=1,
                              validation_size=nb_validation_size,
                              log_frequency=1,
                              unrolled=False)
    trainer.fit()

    final_architecture = trainer.export()
    print(final_architecture)