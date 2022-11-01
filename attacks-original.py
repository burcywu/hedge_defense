import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from wideresnet import WideResNet

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
        random=True
    ):
        assert attack_loss in ['ce', 'bce', 'dlr', 'cw']
        self.attack_steps = attack_steps
        self.step_size    = step_size
        self.temperature  = temperature
        self.attack_loss  = attack_loss
        self.random       = random

    def pgd_attack(self, model, X, y, epsilon):
        X_pgd = Variable(X.data, requires_grad=True)

        if self.random:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(X.device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(self.attack_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                if self.attack_loss == 'ce':
                    loss = nn.CrossEntropyLoss()(model(X_pgd)/self.temperature, y)
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
        X_pgd = self.pgd_attack(model, data, target, epsilons)
        pred = model(X_pgd).data.max(1)[1]
        is_adv = (pred != target.data).float()
        return data, X_pgd, is_adv


class TransferAttack():
    def __init__(
        self,
        attack_steps=20,
        step_size=0.007,
        epsilon=0.031,
        temperature=1,
        attack_loss='ce',
        random=True
    ):
        assert attack_loss in ['ce', 'bce', 'dlr', 'cw']
        self.attack_steps = attack_steps
        self.step_size    = step_size
        self.temperature  = temperature
        self.attack_loss  = attack_loss
        self.random       = random

    def pgd_attack(self, model, X, y, epsilon):
        X_pgd = Variable(X.data, requires_grad=True)

        if self.random:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(X.device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(self.attack_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                if self.attack_loss == 'ce':
                    loss = nn.CrossEntropyLoss()(model(X_pgd)/self.temperature, y)
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
        source_path = '/apdcephfs/share_1367250/boxiwu/hedge-ckpt/transfer.pt'
        source = WideResNet(num_classes=10, widen_factor=10, depth=34).to(torch.device("cuda"))
        source.load_state_dict(torch.load(source_path)['model_state_dict'])
        X_pgd = self.pgd_attack(source, data, target, epsilons)
        pred = model(X_pgd).data.max(1)[1]
        is_adv = (pred != target.data).float()
        return data, X_pgd, is_adv
