#!/usr/bin/python
# -*- encoding: utf-8 -*-

from hedge_defense import HedgeDefense
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from wideresnet import WideResNet
from autoattack import AutoAttack


def cw_loss(logits, y, confidence=0):
    onehot_y = torch.nn.functional.one_hot(y, num_classes=logits.shape[1]).float()
    self_loss = F.nll_loss(-logits, y, reduction='none')
    other_loss = torch.max((1 - onehot_y) * logits, dim=1)[0]
    return -torch.mean(torch.clamp(self_loss - other_loss + confidence, 0))


def bce_loss(logits, y):
    p = F.softmax(logits).detach()
    _, ind = torch.topk(p, 2, dim=1)
    mask = (ind[:, 0] == y).long()
    new_y = mask * ind[:, 1] + (1 - mask) * ind[:, 0]
    return F.cross_entropy(logits, y) + F.nll_loss(torch.log(1 - F.softmax(logits) + 1e-8), new_y)


def dlr_loss(x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        loss_indiv =  -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
        return loss_indiv.sum()


class PGDAttack():
    def __init__(
        self,
        attack_steps=20,
        step_size=0.007,
        epsilon=0.031,
        temperature=1,
        attack_loss='ce',
        random=True,
        target=None,
    ):
        # assert attack_loss in ['ce', 'bce', 'dlr', 'cw']
        self.attack_steps = attack_steps
        self.step_size    = step_size
        self.epsilon      = epsilon
        self.temperature  = temperature
        self.attack_loss  = attack_loss
        self.random       = random
        self.target       = target

    def pgd_attack(self, model, X, y):
        epsilon = self.epsilon
        X_pgd = Variable(X.data, requires_grad=True)

        if self.random:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(X.device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(self.attack_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                if self.attack_loss == 'ce':
                    if self.target is None:
                        loss = nn.CrossEntropyLoss()(model(X_pgd)/self.temperature, y)
                    else:
                        loss = -F.cross_entropy(
                            model(X_pgd) / self.temperature,
                            torch.ones_like(y) * self.target,
                        )
                elif self.attack_loss == 'bce':
                    loss = bce_loss(model(X_pgd)/self.temperature, y)
                elif self.attack_loss == 'dlr':
                    loss = dlr_loss(model(X_pgd)/self.temperature, y)
                else:
                    loss = cw_loss(model(X_pgd)/self.temperature, y)
            loss.backward()
            eta = self.step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        return X_pgd

    def __call__(self, model, data, target, epsilons=0.031):
        X_pgd = self.pgd_attack(model, data, target)
        pred = model(X_pgd).data.max(1)[1]
        is_adv = (pred != target.data).float()
        return data, X_pgd, is_adv


class MultitargetedPGDAttack(nn.Module):
    def __init__(self, *args, pgd_attack_class=PGDAttack, **kwargs):
        super().__init__()
        self.attacks = [
            pgd_attack_class(*args, **kwargs, target=target) for target in range(10)
        ]
        self.hedge_defense = HedgeDefense()

    def forward(self, model, inputs, labels, epsilons):
        batch_size = inputs.size()[0]
        adv_inputs = torch.stack([
            attack.pgd_attack(model, inputs, labels).detach() for attack in self.attacks
        ], dim=1)
        expanded_labels = labels[:, None].expand(-1, 10).reshape(-1)

        hedge_incorrect = torch.ones((batch_size, 10), device=inputs.device, dtype=torch.bool)
        hedge_logits = torch.zeros((batch_size, 10, 10), device=inputs.device)
        for _ in range(3):
            for target in range(10):
                hedge_inputs = self.hedge_defense.hedge_defense(
                    model, adv_inputs[:, target])
                hedge_logits[:, target] = model(hedge_inputs)
                hedge_incorrect[:, target] &= (hedge_logits[:, target].argmax(1) != labels)

        # adv_logits = model(adv_inputs.flatten(end_dim=1))
        # adv_incorrect = (adv_logits.argmax(1) != expanded_labels).reshape(-1, 10)

        wc_indices = (torch.arange(batch_size), hedge_incorrect.float().argmax(1))
        wc_inputs = adv_inputs[wc_indices]
        # wc_logits = hedge_logits[wc_indices]
        wc_logits = model(wc_inputs)
        wc_incorrect = wc_logits.argmax(1) != labels

        return inputs, wc_inputs, wc_incorrect


class RestartAttack(nn.Module):
    def __init__(self, attack, restarts=1):
        super().__init__()
        self.attack = attack
        self.restarts = restarts

    def forward(self, model, inputs, labels):
        wc_inputs = inputs.clone().detach()
        wc_incorrect = torch.zeros_like(labels, dtype=torch.bool)

        for _ in range(self.restarts):
            _, adv_inputs, incorrect = self.attack(
                model,
                inputs[~wc_incorrect],
                labels[~wc_incorrect],
            )
            wc_inputs[~wc_incorrect] = adv_inputs
            wc_incorrect[~wc_incorrect] = incorrect

            print("".join("CI"[x] for x in wc_incorrect.long()))

            if torch.all(wc_incorrect):
                break

        return inputs, wc_inputs, wc_incorrect



class DAAWhite():
    def __init__(
        self,
        attack_steps=20,
        step_size=0.007,
        epsilon=0.031,
        temperature=1,
        attack_loss='ce',
        random=True,
        target=None,
    ):
        # assert attack_loss in ['ce', 'bce', 'dlr', 'cw']
        self.attack_steps = attack_steps
        self.step_size    = step_size
        self.epsilon      = epsilon
        self.temperature  = temperature
        self.attack_loss  = attack_loss
        self.random       = random
        self.target       = target

    def pgd_attack(self, model, X, y):
        epsilon = self.epsilon
        noise_radius = 0.093
        X_pgd = Variable(X.data, requires_grad=True)

        if self.random:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(X.device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
        
        for _ in range(10):
            opt = optim.SGD([X_pgd], lr=1e-3)
            X_inv = Variable(X_pgd.data, requires_grad=True)
            all_eta = 0.

            for _ in range(10):
                opt.zero_grad()
                
                logits = model(X_inv)
                num_classes = list(logits.size())[-1]
                with torch.enable_grad():
                    loss = F.kl_div(F.log_softmax(logits, dim=1),
                                    torch.ones_like(logits) * (1./num_classes),
                                    reduction='batchmean')
                loss.backward()
                eta = self.step_size * X_inv.grad.data.sign()
                X_inv = Variable(X_inv.data + eta, requires_grad=True)
                eta = torch.clamp(X_inv.data - X_pgd.data, -noise_radius, noise_radius)
                X_inv = Variable(X_pgd.data + eta, requires_grad=True)
                X_inv = Variable(torch.clamp(X_inv, 0, 1.0), requires_grad=True)
                opt.zero_grad()

                with torch.enable_grad():
                    loss = nn.CrossEntropyLoss()(model(X_inv), y)
                loss.backward()
                all_eta = all_eta + 0.007 * X_inv.grad.data.sign()

            X_pgd = Variable(X_pgd.data + all_eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


        return X_pgd

    def __call__(self, model, data, target, epsilons=0.031):
        X_pgd = self.pgd_attack(model, data, target)
        pred = model(X_pgd).data.max(1)[1]
        is_adv = (pred != target.data).float()
        return data, X_pgd, is_adv




class DAABlack():
    def __init__(
        self,
        attack_steps=20,
        step_size=0.007,
        epsilon=0.031,
    ):
        self.attack_steps = attack_steps
        self.step_size    = step_size
        self.epsilon      = epsilon

    def pgd_attack(self, model, X, y):
        epsilon = self.epsilon
        model = InternalDefense(model)
        X_pgd = Variable(X.data, requires_grad=True)

        adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='rand')
        adversary.attacks_to_run = ['square']
        X_pgd = adversary.run_standard_evaluation(X, y, bs=128)


        return X_pgd

    def __call__(self, model, data, target, epsilons=0.031):
        X_pgd = self.pgd_attack(model, data, target)
        pred = model(X_pgd).data.max(1)[1]
        is_adv = (pred != target.data).float()
        return data, X_pgd, is_adv


class InternalDefense(nn.Module):
    def __init__(
        self,
        model,
        attack_steps=1,
        step_size=0.045,
        epsilon=0.09,
        random=True
    ):
        super().__init__()
        self.model = model
        self.add_module("deterministic", model)

        self.attack_steps = attack_steps
        self.step_size    = step_size
        self.epsilon      = epsilon
        self.random       = random
        
    def hedge_defense(self, model, X):

        X_hedge = Variable(X.data, requires_grad=True)

        random_noise = torch.FloatTensor(*X_hedge.shape).uniform_(-self.epsilon, self.epsilon).to(X.device)
        X_hedge = Variable(X_hedge.data + random_noise, requires_grad=True)

        for k in range(self.attack_steps):
            opt = optim.SGD([X_hedge], lr=1e-3)
            opt.zero_grad()

            logits = model(X_hedge)
            
            loss = F.kl_div(F.log_softmax(logits, dim=1),
                            torch.ones_like(logits) * (1./10),
                            reduction='batchmean')
            loss.backward()
            eta = self.step_size * X_hedge.grad.data.sign()
            X_hedge = Variable(X_hedge.data + eta, requires_grad=True)
            eta = torch.clamp(X_hedge.data - X.data, -self.epsilon, self.epsilon)
            X_hedge = Variable(X.data + eta, requires_grad=True)
            X_hedge = Variable(torch.clamp(X_hedge, 0, 1.0), requires_grad=True)
        
        return X_hedge.detach()
    
    def forward(self, data):
        with torch.enable_grad():
            X_hedge = self.hedge_defense(self.model, data)
        return self.model(X_hedge)

class ReverseAttack():
    def __init__(
        self,
        attack_steps=20,
        step_size=0.007,
        epsilon=0.031,
        temperature=1,
        attack_loss='ce',
        random=True,
        target=None,
    ):
        # assert attack_loss in ['ce', 'bce', 'dlr', 'cw']
        self.attack_steps = attack_steps
        self.step_size    = step_size
        self.epsilon      = epsilon
        self.temperature  = temperature
        self.attack_loss  = attack_loss
        self.random       = random
        self.target       = target

    def pgd_attack(self, model, X, y):
        epsilon = self.epsilon
        X_pgd = Variable(X.data, requires_grad=True)

        if self.random:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(X.device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(self.attack_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                logits = model(X_pgd)
                num_classes = list(logits.size())[-1]
                loss = F.kl_div(F.log_softmax(logits, dim=1),
                                torch.ones_like(logits) * (1./num_classes),
                                reduction='batchmean')

            loss.backward()
            eta = - self.step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        return X_pgd

    def __call__(self, model, data, target, epsilons=0.031):
        X_pgd = self.pgd_attack(model, data, target)
        pred = model(X_pgd).data.max(1)[1]
        is_adv = (pred != target.data).float()
        return data, X_pgd, is_adv