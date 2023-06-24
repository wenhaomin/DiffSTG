# -*- coding: utf-8 -*-
import easydict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .ugnet import UGnet
from utils.common_utils import gather


class DiffSTG(nn.Module):
    """
    Masked Diffusion Model
    """
    def __init__(self, config: easydict):
        super().__init__()
        self.config = config

        self.N = config.N #steps in the forward process
        self.sample_steps = config.sample_steps # steps in the sample process
        self.sample_strategy = self.config.sample_strategy # sampe strategy
        self.device = config.device
        self.beta_start = config.get('beta_start', 0.0001)
        self.beta_end = config.get('beta_end', 0.02)
        self.beta_schedule = config.beta_schedule

        if config.epsilon_theta == 'UGnet':
            self.eps_model = UGnet(config).to(self.device)


        # create $\beta_1, \dots, \beta_T$
        if self.beta_schedule ==  'uniform':
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.N).to(self.device)

        elif self.beta_schedule == 'quad':
            self.beta = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.N) ** 2
            self.beta = self.beta.to(self.device)

        else:
            raise NotImplementedError

        self.alpha = 1.0 - self.beta

        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor]=None):
        """
        Sample from  q(x_t|x_0) ~ N(x_t; \sqrt\bar\alpha_t * x_0, (1 - \bar\alpha_t)I)
        """
        if eps is None:
            eps = torch.randn_like(x0)

        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)

        return mean + eps * (var ** 0.5)

    def p_sample(self, xt: torch.Tensor, t:torch.Tensor, c):
        """
        Sample from p(x_{t-1}|x_t, c)
        """
        eps_theta = self.eps_model(xt, t, c) # c is the condition
        alpha_coef = 1. / (gather(self.alpha, t) ** 0.5)
        eps_coef =  gather(self.beta, t) / (1 - gather(self.alpha_bar, t)) ** 0.5
        mean = alpha_coef * (xt - eps_coef * eps_theta)

        # var = gather(self.sigma2, t)
        var = (1 - gather(self.alpha_bar, t-1)) / (1 - gather(self.alpha_bar, t)) * gather(self.beta, t)

        eps = torch.randn(xt.shape, device=xt.device)

        return mean + eps * (var ** 0.5)

    def p_sample_loop(self, c):
        """
        :param c: is the masked input tensor, (B, T, V, D), in the prediction task, T = T_h + T_p
        :return: x: the predicted output tensor, (B, T, V, D)
        """
        x_masked, _, _ = c
        # B, F, V, T = x_masked.shape
        B, _, V, T = x_masked.shape
        with torch.no_grad():
            x = torch.randn([B, self.config.F, V, T], device=self.device)#generate input noise
            # Remove noise for $T$ steps
            for t in range(self.N, 0, -1):  #in paper, t should start from T, and end at 1
                t = t - 1 # in code, t is index, so t should minus 1
                if t>0: x = self.p_sample(x, x.new_full((B, ),t, dtype=torch.long), c)
        return  x

    def p_sample_loop_ddim(self, c):
        x_masked, _, _ = c
        B, F, V, T = x_masked.shape

        N = self.N
        timesteps = self.sample_steps
        # skip_type = "uniform"
        skip_type = self.beta_schedule
        if skip_type == "uniform":
            skip = N // timesteps
            # seq = range(0, N, skip)
            seq = range(0, N, skip)
        elif skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(N * 0.8), timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        x = torch.randn([B, self.config.F, V, T], device=self.device) #generate input noise # generate input noise
        xs, x0_preds = generalized_steps(x, seq, self.eps_model, self.beta, c, eta=1)
        return xs, x0_preds

    def set_sample_strategy(self, sample_strategy):
        self.sample_strategy = sample_strategy

    def set_ddim_sample_steps(self, sample_steps):
        self.sample_steps = sample_steps

    def evaluate(self, input, n_samples=2):
        x_masked, _, _ = input
        B, F, V, T = x_masked.shape
        if self.sample_strategy == 'ddim_multi':
            x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)#.to(self.config.device)
            xs, x0_preds = self.p_sample_loop_ddim((x_masked, _, _))
            x = xs[-1]
            x = x.reshape(B, n_samples, F, V, T)
            return x # (B, n_samples, F, V, T)
        elif self.sample_strategy == 'ddim_one':
            xs, x0_preds = self.p_sample_loop_ddim((x_masked, _, _))
            x= xs[-n_samples:]
            x = torch.stack(x, dim=1)
            return x
        if self.sample_strategy == 'ddpm':
            x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)  # .to(self.config.device)
            x = self.p_sample_loop((x_masked, _, _))
            x = x.reshape(B, n_samples, F, V, T)
            return x  # (B, n_samples, F, V, T)
        else:
            raise  NotImplementedError

    def forward(self, input, n_samples = 1):

        return self.evaluate(input, n_samples)

    def loss(self, x0: torch.Tensor, c: Tuple):
        """
        Loss calculation
        x0: (B, ...)
        c: The condition, c is a tuple of torch tensor, here c = (feature, pos_w, pos_d)
        """
        #
        t = torch.randint(0, self.N, (x0.shape[0],), device=x0.device, dtype=torch.long)

        # Note that in the paper, t \in [1, T], but in the code, t \in [0, T-1]
        eps = torch.randn_like(x0)

        xt = self.q_xt_x0(x0, t, eps)
        eps_theta = self.eps_model(xt, t, c)
        return F.mse_loss(eps, eps_theta)


    def model_file_name(self):
        file_name = '+'.join([f'{k}-{self.config[k]}' for k in ['N','T_h','T_p','epsilon_theta']])
        file_name = f'{file_name}.dm4stg'
        return file_name

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

# simple strategy for DDIM
def generalized_steps(x, seq, model, b, c, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            t = t.long()
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = model(xt, t, c)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


# ---Log--
from utils.common_utils import save2file_meta, ws
def save2file(params):
    file_name = ws + f'/output/metrics/DiffSTG.csv'
    head = [
        # data setting
        'data.name',
        # mdoel parameters
        'model', 'model.N', 'model.epsilon_theta', 'model.d_h', 'model.T_h', 'model.T_p', 'model.sample_strategy', 'model.sample_steps', 'model.beta_end',
        # evalution setting
        'n_samples',
        # training set
        'epoch', 'best_epoch', 'batch_size', 'lr', 'wd', 'early_stop', 'is_test', 'log_time',
        # metric result
        'mae', 'rmse', 'mape', 'crps',  'mis', 'time', 'model_path', 'log_path', 'forecast_path',
    ]
    save2file_meta(params,file_name,head)

