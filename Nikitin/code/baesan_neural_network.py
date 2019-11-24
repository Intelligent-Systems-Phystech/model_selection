import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import namedtuple, OrderedDict


VariationalParameter = namedtuple('VariationalParameter', ['mean', 'rho', 'eps'])


def evaluate_prior_params(variational_parameter):
    return variational_parameter.mean + \
        (1 + variational_parameter.rho.exp()).log() * variational_parameter.eps


def build_params(parameters_dict, module, epsilon_setting):
    for name, p in parameters_dict.items():
        if isinstance(p, VariationalParameter):
            if p.eps is None:
                parameters_dict[name] = p._replace(eps=Variable(p.mean.data.clone()))
            epsilon_setting(name, parameters_dict[name])
            setattr(module, name, evaluate_prior_params(parameters_dict[name]))
        elif p is None:
            setattr(module, name, None)
        else:
            build_params(p, getattr(module, name), epsilon_setting)


def get_prior_std(p):
    stdv = 1
#     if p.dim() > 1:
#         for i in range(p.dim() - 1):
#             stdv = stdv * p.size()[i + 1]
#         stdv = 1 / np.sqrt(stdv)
#     else:
#         stdv = 1e-2
    return stdv


def get_KL_divergence(parametrs_dict):
    loss = 0
    for p in parametrs_dict.values():
        if isinstance(p, VariationalParameter):
            mean = p.mean
            std = (1 + p.rho.exp()).log()
            std_prior = get_prior_std(mean)
            loss += (-(std / std_prior).log() +
                     (std.pow(2) + mean.pow(2)) /
                     (2 * std_prior ** 2) - 1 / 2).sum()
        else:
            loss += get_KL_divergence(p)
    return loss


class BayesanNeuralNetwork(nn.Module):
    def __init__(self, model, zero_mean=True, learn_mean=True, learn_rho=True):
        super(BayesanNeuralNetwork, self).__init__()

        self.model = model
        self.dico = OrderedDict()
        self.variationalize(self.dico, self.model, '', zero_mean,
                            learn_mean, learn_rho)
        self.prior_loss = lambda: get_KL_divergence(self.dico)

    def variationalize(self, dico, module, prefix, zero_mean,
                       learn_mean, learn_rho):
        to_erase = []
        paras = module._parameters.items()
        for name, p in paras:

            if p is None:
                dico[name] = None
            else:
                stdv = get_prior_std(p)
                init_rho = np.log(np.exp(stdv) - 1)

                init_mean = p.data.clone()
                if zero_mean:
                    init_mean.fill_(0)

                dico[name] = VariationalParameter(
                    Parameter(init_mean),
                    Parameter(p.data.clone().fill_(init_rho)),
                    None)

                if learn_mean:
                    self.register_parameter(prefix + '_' + name + '_mean',
                                            dico[name].mean)
                if learn_rho:
                    self.register_parameter(prefix + '_' + name + '_rho',
                                            dico[name].rho)

            to_erase.append(name)

        for name in to_erase:
            delattr(module, name)

        for mname, sub_module in module.named_children():
            sub_dico = OrderedDict()
            self.variationalize(sub_dico, sub_module,
                                prefix + ('_' if prefix else '') +
                                mname, zero_mean,
                                learn_mean, learn_rho)
            dico[mname] = sub_dico

    def forward(self, inputs):
        def get_epsilon_setting(name, p):
            if self.training:
                return p.eps.data.normal_()
            return p.eps.data.zero_()

        build_params(self.dico, self.model, get_epsilon_setting)
        return self.model(inputs)
