# -*- coding: utf-8 -*-
"""
ddpg_agent.py
=============
DDPG 标准实现:Actor (OPN/TPN) + Critic (OEN/TEN) + memory buffer + soft update.
公式:
    (37) a = μ(s|θ^μ) + N(t)
    (38) policy gradient ∇J
    (40) y = R + γ Q'(s', μ'(s'))
    (42) θ^{μ'} ← τθ^μ + (1-τ)θ^{μ'}
    (43) θ^{Q'} ← τθ^Q + (1-τ)θ^{Q'}
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=cfg.HIDDEN_SIZES):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s)


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=cfg.HIDDEN_SIZES):
        super().__init__()
        self.fc_s = nn.Linear(state_dim, hidden[0])
        self.fc_a = nn.Linear(action_dim, hidden[0])
        self.mid = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, s, a):
        h = F.relu(self.fc_s(s) + self.fc_a(a))
        return self.mid(h)


class ReplayBuffer(object):
    def __init__(self, cap, sd, ad):
        self.cap = cap
        self.size = 0
        self.ptr = 0
        self.s  = np.zeros((cap, sd), dtype=np.float32)
        self.a  = np.zeros((cap, ad), dtype=np.float32)
        self.r  = np.zeros((cap, 1),  dtype=np.float32)
        self.s2 = np.zeros((cap, sd), dtype=np.float32)
        self.d  = np.zeros((cap, 1),  dtype=np.float32)

    def store(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i]  = s
        self.a[i]  = a
        self.r[i]  = r
        self.s2[i] = s2
        self.d[i]  = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs):
        idx = np.random.randint(0, self.size, size=bs)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class DDPGAgent(object):
    def __init__(self, state_dim, action_dim,
                 lr_actor=cfg.LR_OPN, lr_critic=cfg.LR_OEN,
                 gamma=cfg.GAMMA_MDP, tau=cfg.TAU_SOFT,
                 buffer_cap=cfg.BUFFER_H, batch=cfg.BATCH_I, device=None):
        self.sd = state_dim
        self.ad = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch = batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.opn = ActorNet(state_dim, action_dim).to(self.device)
        self.tpn = ActorNet(state_dim, action_dim).to(self.device)
        self.oen = CriticNet(state_dim, action_dim).to(self.device)
        self.ten = CriticNet(state_dim, action_dim).to(self.device)
        self.tpn.load_state_dict(self.opn.state_dict())
        self.ten.load_state_dict(self.oen.state_dict())

        self.opt_a = torch.optim.Adam(self.opn.parameters(), lr=lr_actor)
        self.opt_c = torch.optim.Adam(self.oen.parameters(), lr=lr_critic)
        self.buffer = ReplayBuffer(buffer_cap, state_dim, action_dim)
        self.noise_sigma = cfg.NOISE_SIGMA_0

    def select_action(self, s, explore=True):
        with torch.no_grad():
            t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
            a = self.opn(t.unsqueeze(0)).cpu().numpy().flatten()
        if explore:
            a = a + np.random.randn(self.ad) * self.noise_sigma
            a = np.clip(a, -1.0, 1.0)
        return a

    def store(self, s, a, r, s2, done):
        self.buffer.store(s, a, r, s2, done)

    def decay_noise(self):
        self.noise_sigma = max(cfg.NOISE_MIN,
                               self.noise_sigma * cfg.NOISE_DECAY)

    def _soft(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def update(self):
        if self.buffer.size < self.batch:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch)
        s  = torch.as_tensor(s,  device=self.device)
        a  = torch.as_tensor(a,  device=self.device)
        r  = torch.as_tensor(r,  device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d  = torch.as_tensor(d,  device=self.device)

        with torch.no_grad():
            y = r + self.gamma * (1.0 - d) * self.ten(s2, self.tpn(s2))
        q = self.oen(s, a)
        c_loss = F.mse_loss(q, y)
        self.opt_c.zero_grad(); c_loss.backward(); self.opt_c.step()

        a_loss = -self.oen(s, self.opn(s)).mean()
        self.opt_a.zero_grad(); a_loss.backward(); self.opt_a.step()

        self._soft(self.tpn, self.opn)
        self._soft(self.ten, self.oen)
        return dict(c=float(c_loss.item()), a=float(a_loss.item()))
