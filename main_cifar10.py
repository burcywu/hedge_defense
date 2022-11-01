import numpy as np
import copy
import argparse
import os
import logging

import torch
from torch.utils.data import Sampler, Dataset, DataLoader
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

import foolbox as fb
from autoattack import AutoAttack
from robustbench import load_model

from hedge_defense import HedgeDefense
from attacks import PGDAttack, DAAWhite, DAABlack, ReverseAttack
from RayS import RayS
from wideresnet import WideResNet
from logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Evaluation on Hedge Attack')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--data-dir', default='data/',
                        help='directory of data')
    parser.add_argument('--model-dir', default='models/',
                        help='directory of data')
    parser.add_argument('--model-name', default='Zhang2019Theoretically',
                        choices=('Gowal2020Uncovering_70_16_extra', 'Gowal2020Uncovering_28_10_extra', 'Wu2020Adversarial_extra',
                                'Carmon2019Unlabeled', 'Sehwag2020Hydra', 'Gowal2020Uncovering_70_16', 'Gowal2020Uncovering_34_20',
                                'Wang2020Improving', 'Wu2020Adversarial', 'Hendrycks2019Using', 'Pang2020Boosting',
                                'Zhang2020Attacks', 'Rice2020Overfitting', 'Huang2020Self', 'Zhang2019Theoretically',
                                'Chen2020Adversarial', 'Engstrom2019Robustness', 'Zhang2019You', 'Wong2020Fast',
                                'Ding2020MMA', 'Standard', 'Madry2018Towards'),
                        help='model_name')
    parser.add_argument('--epsilon', type=float, default=0.031,
                        help='perturbation')
    parser.add_argument('--output-dir', default='output',
                        help='output directory of results')
    parser.add_argument('--repeat-times', type=int, default=1,
                        help='repeat time')

    return parser.parse_args()


def eval_random_noise(model, data, target, epsilon=0.031):
    random_noise = torch.FloatTensor(*data.shape).uniform_(-epsilon, epsilon).to(data.device)
    X_adv = Variable(data.data + random_noise, requires_grad=True)
    pred = model(X_adv).data.max(1)[1]
    is_adv = (pred != target.data).float()
    return X_adv, is_adv


def eval_attack(model, data, target):
    pred = model(data).data.max(1)[1]
    is_adv = (pred != target.data).float()
    return data, is_adv


def main():
    device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True}
    
    setup_logger(args.output_dir, args.model_name)
    logger = logging.getLogger()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    cifar10_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    cifar10_loader = torch.utils.data.DataLoader(cifar10_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    if args.model_name=='Madry2018Towards':
        model = WideResNet(num_classes=10, widen_factor=10, depth=34).to(device)
        model_path = 'PATH_TO_PGD_CKPT'
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
    else:
        model = load_model(model_name=args.model_name, model_dir=args.model_dir, dataset='cifar10', threat_model='Linf')
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    # fmodel = model.to(device)
    
    attack_pool_dict = {
        # 'MT': MultitargetedPGDAttack(),
        'PGD': PGDAttack(),
        # 'DAAWhite': DAAWhite(),
        # 'DAABlack': DAABlack(),
        # 'ReverseAttack': ReverseAttack(),
        'CW' : PGDAttack(attack_loss='cw'),
        'DeepFool': fb.attacks.LinfDeepFoolAttack(),
        'RayS': RayS(),
        'Transfer': TransferAttack()
    }
    adversary_pool_dict = {
        'apgd-ce': AutoAttack(model, norm='Linf', eps=args.epsilon, version='custom', attacks_to_run=['apgd-ce']),
        'apgd-t':  AutoAttack(model, norm='Linf', eps=args.epsilon, version='custom', attacks_to_run=['apgd-t']),
        'fab-t':   AutoAttack(model, norm='Linf', eps=args.epsilon, version='custom', attacks_to_run=['fab-t']),
        'square':  AutoAttack(model, norm='Linf', eps=args.epsilon, version='custom', attacks_to_run=['square']),
        'auto attack': AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard')
    }


    hedge_defense = HedgeDefense(epsilon=args.epsilon)
    
    robustness_adv_dict    = {}
    robustness_random_dict = {}
    robustness_hedge_dict  = {}
    for key in attack_pool_dict:
        robustness_adv_dict[key]    = []
        robustness_random_dict[key] = []
        robustness_hedge_dict[key]  = []
    for key in adversary_pool_dict:
        robustness_adv_dict[key]    = []
        robustness_random_dict[key] = []
        robustness_hedge_dict[key]  = []
    robustness_adv_dict['worst']    = []
    robustness_random_dict['worst'] = []
    robustness_hedge_dict['worst']  = []
        
    for repeat_step in range(args.repeat_times):
        logger.critical('==================== {} ===================='.format(repeat_step+1))
            
        worest_is_adv = torch.zeros(len(cifar10_set))
        worest_is_adv_random = torch.zeros(len(cifar10_set))
        worest_is_adv_hedge = torch.zeros(len(cifar10_set))

        for (attack_name, attack) in attack_pool_dict.items():
            robust = 0
            robust_random = 0
            robust_hedge = 0
            attack_is_adv = []
            attack_is_adv_random = []
            attack_is_adv_hedge = []
            for (data, target) in (cifar10_loader):
                data, target = data.to(device), target.to(device)

                raw, clipped, is_adv = attack(fmodel, data, target, epsilons=args.epsilon)
                _, is_adv_random = eval_random_noise(model, clipped, target, epsilon=args.epsilon)
                _, is_adv_hedge = hedge_defense(model, clipped, target)

                robust += is_adv.sum().cpu().item()
                robust_random += is_adv_random.sum().cpu().item()
                robust_hedge += is_adv_hedge.sum().cpu().item()

                attack_is_adv.append(is_adv.cpu())
                attack_is_adv_random.append(is_adv_random.cpu())
                attack_is_adv_hedge.append(is_adv_hedge.cpu())

            attack_is_adv        = torch.cat(attack_is_adv)
            attack_is_adv_random = torch.cat(attack_is_adv_random)
            attack_is_adv_hedge  = torch.cat(attack_is_adv_hedge)
            worest_is_adv        = torch.logical_or(worest_is_adv, attack_is_adv)
            worest_is_adv_random = torch.logical_or(worest_is_adv_random, attack_is_adv_random)
            worest_is_adv_hedge  = torch.logical_or(worest_is_adv_hedge, attack_is_adv_hedge)
            
            robustness_adv    = 100-100*robust/len(cifar10_set)
            robustness_random = 100-100*robust_random/len(cifar10_set)
            robustness_hedge  = 100-100*robust_hedge/len(cifar10_set)
            robustness_adv_dict[attack_name].append(robustness_adv)
            robustness_random_dict[attack_name].append(robustness_random)
            robustness_hedge_dict[attack_name].append(robustness_hedge)
            logger.critical(attack_name)
            logger.critical('robustness <adv>         : {}'.format(robustness_adv   ))
            logger.critical('robustness <random noise>: {}'.format(robustness_random))
            logger.critical('robustness <hedge attack>: {}'.format(robustness_hedge ))
            logger.critical('')

        x_test = torch.cat([x for (x, y) in cifar10_loader], 0)
        y_test = torch.cat([y for (x, y) in cifar10_loader], 0)

        for (adversary_name, adversary) in adversary_pool_dict.items():
            x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=200)
            robust = 0
            robust_random = 0
            robust_hedge = 0
            attack_is_adv = []
            attack_is_adv_random = []
            attack_is_adv_hedge = []
            for idx in range(0, 10000, 100):
                x_adv_batch = x_adv[idx:idx+100]
                y_batch = y_test[idx:idx+100]
                x_adv_batch, y_batch = x_adv_batch.to(device), y_batch.to(device)

                _, is_adv = eval_attack(model, x_adv_batch, y_batch)
                _, is_adv_random = eval_random_noise(model, x_adv_batch, y_batch, epsilon=args.epsilon)
                _, is_adv_hedge = hedge_defense(model, x_adv_batch, y_batch)

                robust += is_adv.sum().cpu().item()
                robust_random += is_adv_random.sum().cpu().item()
                robust_hedge += is_adv_hedge.sum().cpu().item()

                attack_is_adv.append(is_adv.cpu())
                attack_is_adv_random.append(is_adv_random.cpu())
                attack_is_adv_hedge.append(is_adv_hedge.cpu())

            attack_is_adv        = torch.cat(attack_is_adv)
            attack_is_adv_random = torch.cat(attack_is_adv_random)
            attack_is_adv_hedge  = torch.cat(attack_is_adv_hedge)
            worest_is_adv        = torch.logical_or(worest_is_adv, attack_is_adv)
            worest_is_adv_random = torch.logical_or(worest_is_adv_random, attack_is_adv_random)
            worest_is_adv_hedge  = torch.logical_or(worest_is_adv_hedge, attack_is_adv_hedge)
            
            robustness_adv    = 100-100*robust/len(cifar10_set)
            robustness_random = 100-100*robust_random/len(cifar10_set)
            robustness_hedge  = 100-100*robust_hedge/len(cifar10_set)
            robustness_adv_dict[adversary_name].append(robustness_adv)
            robustness_random_dict[adversary_name].append(robustness_random)
            robustness_hedge_dict[adversary_name].append(robustness_hedge)
            logger.critical(adversary_name)
            logger.critical('robustness <adv>         : {}'.format(robustness_adv   ))
            logger.critical('robustness <random noise>: {}'.format(robustness_random))
            logger.critical('robustness <hedge attack>: {}'.format(robustness_hedge ))
            logger.critical('')

        robustness_adv    = 100-100*torch.mean(worest_is_adv.float()).item()
        robustness_random = 100-100*torch.mean(worest_is_adv_random.float()).item()
        robustness_hedge  = 100-100*torch.mean(worest_is_adv_hedge.float()).item()
        robustness_adv_dict['worst'].append(robustness_adv)
        robustness_random_dict['worst'].append(robustness_random)
        robustness_hedge_dict['worst'].append(robustness_hedge)
        logger.critical('worst final output')
        logger.critical('robustness <adv>         : {}'.format(robustness_adv   ))
        logger.critical('robustness <random noise>: {}'.format(robustness_random))
        logger.critical('robustness <hedge attack>: {}'.format(robustness_hedge ))

    logger.critical('\n\n\n')
    for key in robustness_adv_dict:
        robustness_adv    = np.array(robustness_adv_dict[key])
        robustness_random = np.array(robustness_random_dict[key])
        robustness_hedge  = np.array(robustness_hedge_dict[key])
        logger.critical(key)
        logger.critical('robustness <adv>         : mean is {}, var is {}.'.format(robustness_adv.mean()   , robustness_adv.std()   ))
        logger.critical('robustness <random noise>: mean is {}, var is {}.'.format(robustness_random.mean(), robustness_random.std()))
        logger.critical('robustness <hedge attack>: mean is {}, var is {}.'.format(robustness_hedge.mean() , robustness_hedge.std() ))
        logger.critical('')

if __name__ == '__main__':
    args = parse_args()
    main()
