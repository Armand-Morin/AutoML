# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
from collections import OrderedDict
from read_input import read_study
import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.retiarii.oneshot.pytorch.darts import DartsLayerChoice, DartsInputChoice
from nni.retiarii.oneshot.interface import BaseOneShotTrainer
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, replace_layer_choice, replace_input_choice, to_device

input_folder = './input/'
input_file = 'swingIbanez351.json'

_logger = logging.getLogger(__name__)


class OurDartsTrainer(BaseOneShotTrainer):

    def __init__(self, model, loss, metrics, optimizer,
                 num_epochs, dataset, nb_iter, generator, price_model, calendar, child_batch_size, validation_size,
                 grad_clip=5.,
                 learning_rate=2.5E-3, batch_size=64, workers=4,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False):
        self.model = model
        self.nb_iter = nb_iter
        self.generator = generator
        self.price_model = price_model
        self.calendar = calendar
        self.child_batch_size = child_batch_size
        self.validation_size = validation_size
        self.loss = loss
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.optimizer = optimizer
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # self.device = torch.device('cuda:0')
        self.log_frequency = log_frequency
        # self.model.to(self.device)

        parameters = read_study(input_folder, input_file)
        self.control = parameters['control']
        self.payoff = parameters['payoff']

        self.nas_modules = []
        replace_layer_choice(self.model, DartsLayerChoice, self.nas_modules)
        replace_input_choice(self.model, DartsInputChoice, self.nas_modules)
        # for _, module in self.nas_modules:
        #    module.to(self.device)

        self.model_optim = optimizer
        # use the same architecture weight for modules with duplicated names
        ctrl_params = {}
        for _, m in self.nas_modules:
            if m.name in ctrl_params:
                assert m.alpha.size() == ctrl_params[
                    m.name].size(), 'Size of parameters with the same label should be same.'
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        self.ctrl_optim = torch.optim.Adam(list(ctrl_params.values()), arc_learning_rate, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
        self.unrolled = unrolled
        self.grad_clip = 5.

        self._init_dataloader()

    def fit(self):
        for i in range(self.num_epochs):
            self._train_one_epoch(i)

    @torch.no_grad()
    def export(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result

    def _init_dataloader(self):
        n_train = len(self.dataset)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=self.workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=self.workers)

    def _train_one_epoch(self, epoch):

        self.model.train()

        meters = AverageMeterGroup()

        for step in range(1, self.nb_iter + 1):
            trn_X = torch.from_numpy(
                next(self.generator(self.price_model, self.calendar, self.batch_size)))  # !!!!!!!!!!!!!!!!
            val_X = torch.from_numpy(
                next(self.generator(self.price_model, self.calendar, self.validation_size)))  # !!!!!!!!!!!!!!!!
            # (trn_X, val_X) in enumerate(zip(self.train_loader, self.valid_loader)):
            # trn_X = to_device(trn_X, self.device)
            # val_X = to_device(val_X, self.device)

            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            if self.unrolled:
                try:
                    self._unrolled_backward(trn_X, val_X)
                except:
                    print("UROLLED MUST BE FALSE")
            else:
                self._backward(val_X)
            self.ctrl_optim.step()

            # phase 2: child network step
            self.model_optim.zero_grad()
            logits_train, loss_train = self._logits_and_loss(trn_X, stopped=True)
            loss_train.requires_grad = True
            loss_train.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clipping
            self.model_optim.step()

            # metrics = self.metrics(logits)
            # metrics['loss'] = loss.item()
            # meters.update(metrics)

            logits_test, loss_test = self._logits_and_loss(val_X, stopped=False)

            if self.log_frequency is not None and step % self.log_frequency == 0:

                print('Epoch [%s/%s] Step [%s/%s] ', epoch + 1,
                             self.num_epochs, step, self.nb_iter)
                print("Loss train : ", loss_train.item())
                print("Loss test : ", loss_test.item())
                print(torch.mean(logits_test[0], axis=0))
                print('-' *  30)
                print("")

    def _logits_and_loss(self, simulations, stopped=False):
        prices = simulations[:, :, :-1]
        time = simulations[:, :, -1]
        logits = self.model.compute_action(prices, time, torch.tensor([True]))
        loss = self.loss(logits, simulations, stopped)
        return logits, loss

    def _backward(self, val_X):
        """
        Simple backward with gradient descent
        """
        _, loss = self._logits_and_loss(val_X)
        loss.backward()
