import os
import time
import torch
import gym
from all.agents import Agent

class GreedyAgent(Agent):
    def __init__(
            self,
            action_space,
            feature=None,
            q=None,
            policy=None
    ):
        self.action_space = action_space
        self.feature = feature
        self.policy = None
        if policy:
            self.policy = policy
        else:
            self.policy = q
        if not self.policy:
            raise TypeError('GreedyAgent must have either policy or q function')

    def act(self, state, _):
        with torch.no_grad():
            if self.feature:
                state = self.feature(state)
            if isinstance(self.action_space, gym.spaces.Discrete):
                return self.choose_discrete(state)
            if isinstance(self.action_space, gym.spaces.Box):
                return self.choose_continuous(state)
            raise TypeError('Unknown action space')

    def eval(self, state, reward):
        return self.act(state, reward)

    def choose_discrete(self, state):
        ret = self.policy(state)
        if isinstance(ret, torch.Tensor):
            if len(ret.shape) == 2: # categorical dqn
                return torch.argmax((ret * self.policy.atoms).sum(dim=1)).item()
            return torch.argmax(self.policy(state)).item()
        if isinstance(ret, torch.distributions.distribution.Distribution):
            return ret.sample()
        return ret # unknown type, return it and pray!

    def choose_continuous(self, state):
        ret = self.policy(state)
        if isinstance(ret, torch.Tensor):
            return ret
        if isinstance(ret, tuple):
            return ret[0]
        if isinstance(ret, torch.distributions.distribution.Distribution):
            return ret.sample()
        return ret # unknown type, return it and pray!

    @staticmethod
    def load(dirname, env):
        feature = None
        policy = None
        q = None
        for filename in os.listdir(dirname):
            if filename == 'feature.pt':
                feature = torch.load(os.path.join(dirname, filename)).to(env.device)
            if filename == 'policy.pt':
                policy = torch.load(os.path.join(dirname, filename)).to(env.device)
            if filename in ('q.pt', 'q_dist.pt'):
                q = torch.load(os.path.join(dirname, filename)).to(env.device)

        agent = GreedyAgent(
            env.action_space,
            feature=feature,
            policy=policy,
            q=q,
        )

        return agent
