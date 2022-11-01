import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import random


class HedgeDefense():
    def __init__(
        self,
        attack_steps=20,
        step_size=0.007,
        epsilon=0.031,
        random=True
    ):
        self.attack_steps = attack_steps
        self.step_size    = step_size
        self.epsilon      = epsilon
        self.random       = random
        
    def hedge_defense(self, model, X):

        X_hedge = Variable(X.data, requires_grad=True)
        
        if self.random:
            random_noise = torch.FloatTensor(*X_hedge.shape).uniform_(-self.epsilon, self.epsilon).to(X.device)
            X_hedge = Variable(X_hedge.data + random_noise, requires_grad=True)
        
        sum_pred = None

        for k in range(self.attack_steps):
            opt = optim.SGD([X_hedge], lr=1e-3)
            opt.zero_grad()
            logits = model(X_hedge)
            # Pred on all searched hedge examples
            if k==0:
                sum_pred = torch.zeros_like(logits)
            else:
                sum_pred = sum_pred + logits

            num_classes = list(logits.size())[-1]
            loss = F.kl_div(F.log_softmax(logits, dim=1),
                            torch.ones_like(logits) * (1./num_classes),
                            reduction='batchmean')
            loss.backward()
            eta = self.step_size * X_hedge.grad.data.sign()
            X_hedge = Variable(X_hedge.data + eta, requires_grad=True)
            eta = torch.clamp(X_hedge.data - X.data, -self.epsilon, self.epsilon)
            X_hedge = Variable(X.data + eta, requires_grad=True)
            X_hedge = Variable(torch.clamp(X_hedge, 0, 1.0), requires_grad=True)
        
        return X_hedge.detach(), sum_pred
    
    def __call__(self, model, data, target):
        X_hedge, sum_pred = self.hedge_defense(model, data)
        pred = sum_pred.data.max(1)[1]
        is_adv = (pred != target.data).float()
        return X_hedge, is_adv
