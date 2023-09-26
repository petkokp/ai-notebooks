import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PNNColumn(nn.Module):
    def __init__(self, cid, nchannels, nactions):
        super(PNNColumn, self).__init__()
        nhidden = 256
        self.cid = cid
        self.nlayers = 6

        self.w = nn.ModuleList()
        self.u = nn.ModuleList()
        self.v = nn.ModuleList()
        self.alpha = nn.ModuleList()

        self.w.append(
            nn.Conv2d(nchannels, 32, kernel_size=3, stride=2, padding=1))
        self.w.extend([
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            for _ in range(self.nlayers - 3)
        ])
        conv_out_size = self._get_conv_out((nchannels, 84, 84))
        self.w.append(nn.Linear(conv_out_size, nhidden))
        self.w.append(
            nn.ModuleList(
                [nn.Linear(nhidden, 1),
                 nn.Linear(nhidden, nactions)]))

        for i in range(self.cid):
            self.v.append(nn.ModuleList())
            self.v[i].append(nn.Identity())
            self.v[i].extend([
                nn.Conv2d(32, 1, kernel_size=1)
                for _ in range(self.nlayers - 3)
            ])
            self.v[i].append(nn.Linear(conv_out_size, conv_out_size))
            self.v[i].append(
                nn.ModuleList(
                    [nn.Linear(nhidden, nhidden),
                     nn.Linear(nhidden, nhidden)]))

            self.alpha.append(nn.ParameterList())
            self.alpha[i].append(
                nn.Parameter(torch.Tensor(1), requires_grad=False))
            self.alpha[i].extend([
                nn.Parameter(
                    torch.Tensor(np.array(np.random.choice([1e0, 1e-1,
                                                            1e-2]))))
                for _ in range(self.nlayers)
            ])

            self.u.append(nn.ModuleList())
            self.u[i].append(nn.Identity())
            self.u[i].extend([
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
                for _ in range(self.nlayers - 3)
            ])
            self.u[i].append(nn.Linear(conv_out_size, nhidden))
            self.u[i].append(
                nn.ModuleList(
                    [nn.Linear(nhidden, 1),
                     nn.Linear(nhidden, nactions)]))

        self._reset_parameters()
        self.w[-1][0].weight.data = self._normalized(self.w[-1][0].weight.data)
        self.w[-1][1].weight.data = self._normalized(self.w[-1][1].weight.data,
                                                     1e-2)

        for i in range(self.cid):
            self.v[i][-1][0].weight.data = self._normalized(
                self.v[i][-1][0].weight.data)
            self.v[i][-1][1].weight.data = self._normalized(
                self.v[i][-1][1].weight.data, 1e-2)

            self.u[i][-1][0].weight.data = self._normalized(
                self.u[i][-1][0].weight.data)
            self.u[i][-1][1].weight.data = self._normalized(
                self.u[i][-1][1].weight.data, 1e-2)

    def forward(self, x, pre_out):
        next_out, w_out = [torch.zeros(x.shape)], x

        critic_out, actor_out = None, None
        for i in range(self.nlayers - 1):
            if i == self.nlayers - 2:
                w_out = w_out.view(w_out.size(0), -1)
                for k in range(self.cid):
                    pre_out[k][i] = pre_out[k][i].view(pre_out[k][i].size(0),
                                                       -1)
            w_out = self.w[i](w_out)
            u_out = [
                self.u[k][i](self._activate(self.v[k][i](self.alpha[k][i] *
                                                         (pre_out[k][i]))))
                if self.cid and i else torch.zeros(w_out.shape)
                for k in range(self.cid)
            ]
            w_out = self._activate(w_out + sum(u_out))
            next_out.append(w_out)

        critic_out = self.w[-1][0](w_out)
        pre_critic_out = [
            self.u[k][-1][0](self._activate(self.v[k][-1][0](
                self.alpha[k][-2] * pre_out[k][self.nlayers - 1])))
            if self.cid else torch.zeros(critic_out.shape)
            for k in range(self.cid)
        ]
        actor_out = self.w[-1][1](w_out)
        pre_actor_out = [
            self.u[k][-1][1](self._activate(self.v[k][-1][1](
                self.alpha[k][-1] * pre_out[k][self.nlayers - 1])))
            if self.cid else torch.zeros(actor_out.shape)
            for k in range(self.cid)
        ]

        return critic_out + sum(pre_critic_out), \
            actor_out + sum(pre_actor_out), \
            next_out

    def _activate(self, x):
        return F.elu(x)

    def _normalized(self, weights, std=1.0):
        output = torch.randn(weights.size())
        output *= std / torch.sqrt(output.pow(2).sum(1, keepdim=True))
        return output

    def _get_conv_out(self, shape):
        output = torch.zeros(1, *shape)
        for i in range(self.nlayers - 2):
            output = self.w[i](output)
        return int(np.prod(output.size()))

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class PNN(nn.Module):
    def __init__(self, allenvs, shared=None):
        super(PNN, self).__init__()
        self.shared = shared
        self.current = 0
        self.columns = nn.ModuleList()

        for i, env in enumerate(allenvs):
            nchannels = env.observation_space.shape[0]
            nactions = env.action_space.n
            self.columns.append(
                PNNColumn(len(self.columns), nchannels, nactions))

            if i != 0:
                for params in self.columns[i].parameters():
                    params.requires_grad = False

    def forward(self, X):
        h_actor, h_critic, next_out = None, None, []

        for i in range(self.current + 1):
            h_critic, h_actor, out = self.columns[i](X, next_out)
            next_out.append(out)

        return h_critic, h_actor

    def freeze(self):
        self.current += 1

        if self.shared is not None:
            self.shared.value = self.current

        if self.current >= len(self.columns):
            return

        for i in range(self.current + 1):
            for params in self.columns[i].parameters():
                params.requires_grad = False

        for params in self.columns[self.current].parameters():
            params.requires_grad = True
        for i in range(self.current):
            self.columns[self.current].alpha[i][0].requires_grad = False

    def parameters(self, cid=None):
        if cid is None:
            return super(PNN, self).parameters()
        return self.columns[cid].parameters()
