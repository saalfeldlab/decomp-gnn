
import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from tqdm import trange

class DeepNormal(nn.Module):

    def __init__(self, n_inputs, n_hidden):
        super().__init__()

        # Shared parameters
        self.shared_layer = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )

        # Mean parameters
        self.mean_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_hidden, 1),
        )

        # Standard deviation parameters
        self.std_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_hidden, 1),
            nn.Softplus(),  # enforces positivity
        )

    def forward(self, x):
        # Shared embedding
        shared = self.shared_layer(x)

        # Parametrization of the mean
        μ = self.mean_layer(shared)

        # Parametrization of the standard deviation
        σ = self.std_layer(shared)

        return torch.distributions.Normal(μ, σ)

def compute_loss(model, x, y):
    normal_dist = model(x)
    neg_log_likelihood = -normal_dist.log_prob(y)
    return torch.mean(neg_log_likelihood)


class my_distrib(torch.nn.Module):

    def __init__(self):
        
        super(my_distrib, self).__init__()

        self.learned_distribution = nn.Parameter(torch.ones(100, dtype=torch.float32, device='cuda:0', requires_grad=True))

    def forward(self):

        normalized_distribution = torch.relu(self.learned_distribution)
        normalized_distribution = normalized_distribution / torch.sum(normalized_distribution)

        m = Categorical(normalized_distribution)
        return m, normalized_distribution


if __name__ == '__main__':

    print('')
    print('version 0.2.0 240111')
    print('')


    gt_distrib1 = torch.distributions.Normal(25, 5)
    gt_distrib2 = torch.distributions.Normal(75, 5)

    prob1 = gt_distrib1.sample((200,))
    prob2 = gt_distrib2.sample((200,))
    probs = torch.cat((prob1, prob2), 0)

    # fig = plt.figure(figsize=(10, 10))
    # plt.hist(probs.detach().cpu().numpy(),150)
    # plt.show()

    t = plt.hist(probs.detach().cpu().numpy(), 100, range=[0, 100])
    t = torch.tensor(t[0],device='cuda:0')
    t = t / torch.sum(t)

    # matplotlib.use("Qt5Agg")
    # fig = plt.figure(figsize=(10, 10))
    # plt.scatter(range(100), t.detach().cpu().numpy(), c='r')

    gt_distrib = Categorical(t)

    md = my_distrib()    
    optimizer = torch.optim.Adam(md.parameters(), lr=0.01)

    # plot a normal distribution

    for n in trange(10000):

        m, normalized_distribution = md()

        draw = m.rsample((100, 1))

        loss = (draw.float()-40).norm(2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



    for n in trange(10000):
    
        m, normalized_distribution = md()
        draw = torch.randint(0, 100, (100,), device='cuda:0')
        loss = (m.log_prob(draw.detach()) - gt_distrib.log_prob(draw.detach())).norm(2)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(range(100), t.detach().cpu().numpy(), c='g',s=40)
    plt.scatter (range(100), normalized_distribution.detach().cpu().numpy(), c='k',s=2)
    plt.show()
    
    
    # plt.scatter(range(100), probs.detach().cpu().numpy())
    



    # def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
    #     total_policy_loss = 0
    #
    #
    # total_value_loss = 0
    # states = states.detach()
    # actions = actions.detach()
    # log_prob_actions = log_prob_actions.detach()
    # advantages = advantages.detach()
    # returns = returns.detach()
    #
    # for _ in range(ppo_steps):  # get new log prob of actions for all input states
    #
    #     action_pred, value_pred = policy(states)
    #     value_pred = value_pred.squeeze(-1)
    #
    #
    #     action_prob = F.softmax(action_pred, dim=-1)
    #
    #
    #     dist = distributions.Categorical(action_prob)  # new log prob using old actions
    #
    #
    #     new_log_prob_actions = dist.log_prob(actions)
    #
    #
    #     policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
    #     policy_loss_1 = policy_ratio * advantages
    #     policy_loss_2 = torch.clamp(policy_ratio, min=1.0 - ppo_clip, max=1.0 + ppo_clip) * advantages
    #
    #
    #     # here need to implement eligibility trace - really??! try
    #     policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
    #     # update previous eligibility trace
    #     # update eligibility trace        value_loss = F.smooth_l1_loss(returns, value_pred).mean()        optimizer.zero_grad()        policy_loss.backward()
    #     value_loss.backward()
    #     optimizer.step()
    #     total_policy_loss += policy_loss.item()
    #     total_value_loss += value_loss.item()
    #     return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


        
    

    

    
    