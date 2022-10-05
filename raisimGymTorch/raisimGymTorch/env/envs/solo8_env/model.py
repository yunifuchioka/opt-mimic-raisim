import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.distributions import Normal, Categorical
import math

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data.view(1, -1), gain=gain)
    return module

init_r_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    nn.init.orthogonal_,
    0.1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ActorCriticNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], num_contact=0):
        super(ActorCriticNet, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer
        self.p_fcs = nn.ModuleList()
        self.v_fcs = nn.ModuleList()
        self.hidden_layer_v = [512, 512]
        if (len(hidden_layer) > 0):
            p_fc = init_r_(nn.Linear(num_inputs, self.hidden_layer[0]))
            v_fc = init_r_(nn.Linear(num_inputs, self.hidden_layer_v[0]))
            self.p_fcs.append(p_fc)
            self.v_fcs.append(v_fc)
            for i in range(len(self.hidden_layer)-1):
                p_fc = init_r_(nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1]))
                v_fc = init_r_(nn.Linear(self.hidden_layer_v[i], self.hidden_layer_v[i+1]))
                self.p_fcs.append(p_fc)
                self.v_fcs.append(v_fc)
            self.mu = init_r_(nn.Linear(self.hidden_layer[-1], num_outputs))
        else:
            #p_fc = init_r_(nn.Linear(num_inputs, num_outputs))
            #self.p_fcs.append(p_fc)
            self.mu = init_r_(nn.Linear(num_inputs, num_outputs))
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        self.v = init_r_(nn.Linear(self.hidden_layer_v[-1],1))
        self.noise = 0
        #self.train()


    def forward(self, inputs):
        # actor
        if len(self.hidden_layer) > 0:
            x = F.relu(self.p_fcs[0](inputs))
            for i in range(len(self.hidden_layer)-1):
                x = F.relu(self.p_fcs[i+1](x))
            mu = torch.tanh(self.mu(x))
        else:
            mu = torch.tanh(self.mu(inputs))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v

    def get_log_stds(self, actions):
        return Variable(torch.Tensor(self.noise)*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(actions).to(device)
        #return self.log_std.unsqueeze(0).expand_as(actions)

    def sample_best_actions(self, inputs):
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = torch.tanh(self.mu(x))
        return mu

    def sample_actions(self, inputs):
        mu = self.sample_best_actions(inputs)
        log_std = self.get_log_stds(mu)
        #std = torch.exp(log_std)
        eps = torch.randn(mu.size(), device=device)
        actions = torch.clamp(mu + log_std.exp()*(eps), -1, 1)
        return actions, mu

    def set_noise(self, noise):
        self.noise = noise

    def get_action(self, inputs):
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = torch.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)
        return mu, log_std

    def get_value(self, inputs, device="cpu"):
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        return v
    def calculate_prob_gpu(self, inputs, actions):
        log_stds = self.get_log_stds(actions).to(device)
        #print(log_stds.shape)
        mean_actions = self.sample_best_actions(inputs)
        #print(mean_actions.shape)
        #w = self.get_w(inputs).to(device)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        #print(log_probs)
        #probs = (log_probs.exp() * w.t()).sum(dim=0).log()
        #print(probs)
        return log_probs

    def calculate_prob(self, inputs, actions, mean_actions):
        log_stds = self.get_log_stds(actions)
        #print(log_stds.shape)
        # mean_actions = self.sample_best_actions(inputs)
        #print(mean_actions.shape)
        #w = self.get_w(inputs).to(device)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        #print(log_probs)
        #probs = (log_probs.exp() * w.t()).sum(dim=0).log()
        #print(probs)
        return log_probs

class ActorCriticNetFF(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], num_contact=0):
        super(ActorCriticNetFF, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer

        self.policy_experts = [nn.ModuleList(), nn.ModuleList()]
        self.value_experts = [nn.ModuleList(), nn.ModuleList()]
        self.num_experts = 1
        self.num_gating_input = 8
        self.num_additional_expert_input = 8
        self.policy_layers = [
            (
                nn.Parameter(torch.empty(self.num_experts, num_inputs-self.num_gating_input+self.num_additional_expert_input, hidden_layer[0], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[0], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[0], hidden_layer[1], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[1], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[1], num_outputs, device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, num_outputs, device=device)),
                F.tanh,
            ),
        ]

        self.value_layers = [
            (
                nn.Parameter(torch.empty(self.num_experts, num_inputs-self.num_gating_input+self.num_additional_expert_input, hidden_layer[0], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[0], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[0], hidden_layer[1], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[1], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[1], 1, device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, 1, device=device)),
                lambda x: x,
            ),
        ]

        self.actor_params = []
        for index, (weight, bias, activation) in enumerate(self.policy_layers):
            if "relu" in activation.__name__:
                gain = 0.1
            elif "tanh" in activation.__name__:
                gain = 0.1
            else:
                gain = 1.0
            index = str(index)
            for w in weight:
                nn.init.orthogonal_(w, gain=gain)
            for b in bias:
                nn.init.orthogonal_(b, gain=gain)
            self.register_parameter("policy_w" + index, weight)
            self.register_parameter("policy_b" + index, bias)

            self.actor_params.append(weight)
            self.actor_params.append(bias)

        self.critic_params = []
        for index, (weight, bias, activation) in enumerate(self.value_layers):
            if activation is None:
                gain = 1.0
            elif "relu" in activation.__name__:
                gain = 0.1
            elif "tanh" in activation.__name__:
                gain = 0.1
            else:
                gain = 1.0
            index = str(index)
            for w in weight:
                nn.init.orthogonal_(w, gain=gain)
            for b in bias:
                nn.init.orthogonal_(b, gain=gain)
            self.register_parameter("value_w" + index, weight)
            self.register_parameter("value_b" + index, bias)

            self.critic_params.append(weight)
            self.critic_params.append(bias)
        
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        # self.v = nn.Linear(self.hidden_layer_v[-1],1)
        self.noise = 0
        #self.train()

    def forward(self, inputs):
        # actor
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = torch.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v

    def get_log_stds(self, actions):
        # print("\n\n\n\n", actions.shape, self.noise, self.num_outputs)
        return self.noise_tensor.unsqueeze(0).expand_as(actions)
        #return self.log_std.unsqueeze(0).expand_as(actions)

    def sample_best_actions(self, inputs):
        out = inputs[:, :].clone()
        for (weight, bias, activation) in self.policy_layers:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                .add(bias)  # (N, B, H)
                .sum(dim=0)
            )
        return out

    def sample_best_action_and_coefficients(self, inputs):
        # out = inputs[:, :-self.num_gating_input+self.num_additional_expert_input].clone()
        out = inputs[:, :].clone()

        for (weight, bias, activation) in self.policy_layers:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                .add(bias)  # (N, B, H)
                .sum(dim=0)
            )
        return out, gating_weights

    def sample_actions(self, inputs):
        mu = self.sample_best_actions(inputs)
        log_std = self.get_log_stds(mu)
        #std = torch.exp(log_std)
        eps = torch.randn(mu.size(), device=device)
        actions = torch.clamp(mu + log_std.exp()*(eps), -1, 1)
        return actions, mu

    def set_noise(self, noise):
        self.noise = noise
        self.noise_tensor = torch.Tensor(self.noise).to(device)

    def get_value(self, inputs, device="cpu"):
        out = inputs[:, :].clone()

        for (weight, bias, activation) in self.value_layers:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                .add(bias)  # (N, B, H)
                .sum(dim=0)
            )
            
        # out = out.matmul(weight).add(bias).mul(gating_weights).sum(dim=0)
        return out

    def calculate_prob_gpu(self, inputs, actions):
        log_stds = self.get_log_stds(actions)
        mean_actions = self.sample_best_actions(inputs)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        return log_probs

    # def calculate_prob(self, inputs, actions):
    #     log_stds = self.get_log_stds(actions)
    #     #print(log_stds.shape)
    #     mean_actions = self.sample_best_actions(inputs)
    #     #print(mean_actions.shape)
    #     #w = self.get_w(inputs).to(device)
    #     numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
    #     log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
    #     #print(log_probs)
    #     #probs = (log_probs.exp() * w.t()).sum(dim=0).log()
    #     #print(probs)
    #     return log_probs

    def calculate_prob(self, inputs, actions, mean_actions):
        log_stds = self.get_log_stds(actions)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        return log_probs

class ActorCriticNetMann(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], num_contact=0):
        super(ActorCriticNetMann, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer

        self.policy_experts = [nn.ModuleList(), nn.ModuleList()]
        self.value_experts = [nn.ModuleList(), nn.ModuleList()]
        self.num_experts = 8
        self.num_gating_input = num_inputs #8
        self.num_additional_expert_input = num_inputs
        self.policy_layers = [
            (
                nn.Parameter(torch.empty(self.num_experts, num_inputs-self.num_gating_input+self.num_additional_expert_input, hidden_layer[0], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[0], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[0], hidden_layer[1], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[1], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[1], num_outputs, device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, num_outputs, device=device)),
                F.tanh,
            ),
        ]

        self.value_layers = [
            (
                nn.Parameter(torch.empty(self.num_experts, num_inputs-self.num_gating_input+self.num_additional_expert_input, hidden_layer[0], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[0], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[0], hidden_layer[1], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[1], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[1], 1, device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, 1, device=device)),
                lambda x: x,
            ),
        ]

        self.actor_params = []
        for index, (weight, bias, activation) in enumerate(self.policy_layers):
            if "relu" in activation.__name__:
                gain = 0.1
            elif "tanh" in activation.__name__:
                gain = 0.1
            else:
                gain = 1.0
            index = str(index)
            for w in weight:
                nn.init.orthogonal_(w, gain=gain)
            for b in bias:
                nn.init.orthogonal_(b, gain=gain)
            self.register_parameter("policy_w" + index, weight)
            self.register_parameter("policy_b" + index, bias)

            self.actor_params.append(weight)
            self.actor_params.append(bias)

        self.critic_params = []
        for index, (weight, bias, activation) in enumerate(self.value_layers):
            if activation is None:
                gain = 1.0
            elif "relu" in activation.__name__:
                gain = 0.1
            elif "tanh" in activation.__name__:
                gain = 0.1
            else:
                gain = 1.0
            index = str(index)
            for w in weight:
                nn.init.orthogonal_(w, gain=gain)
            for b in bias:
                nn.init.orthogonal_(b, gain=gain)
            self.register_parameter("value_w" + index, weight)
            self.register_parameter("value_b" + index, bias)

            self.critic_params.append(weight)
            self.critic_params.append(bias)

        self.policy_gate = nn.Sequential(
            nn.Linear(self.num_gating_input, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_experts),
        )

        self.value_gate = nn.Sequential(
            nn.Linear(self.num_gating_input, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_experts),
        )
        
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        # self.v = nn.Linear(self.hidden_layer_v[-1],1)
        self.noise = 0
        #self.train()

    def forward(self, inputs):
        # actor
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = torch.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v

    def get_log_stds(self, actions):
        # print("\n\n\n\n", actions.shape, self.noise, self.num_outputs)
        return self.noise_tensor.unsqueeze(0).expand_as(actions)
        #return self.log_std.unsqueeze(0).expand_as(actions)

    def evaluate_policy_gate_l2(self, inputs):
        gating_weights = F.softmax(self.policy_gate(inputs[:, -self.num_gating_input:]), dim=1)
        return (gating_weights**2).mean()

    def evaluate_value_gate_l2(self, inputs):
        gating_weights = F.softmax(self.value_gate(inputs[:, -self.num_gating_input:]), dim=1)
        return (gating_weights**2).mean()

    def sample_best_actions(self, inputs):
        gating_weights = F.softmax(self.policy_gate(inputs[:, -self.num_gating_input:]), dim=1).t().unsqueeze(-1)
        # out = inputs[:, :-self.num_gating_input+self.num_additional_expert_input].clone()
        out = inputs[:, :].clone()

        for (weight, bias, activation) in self.policy_layers:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                .add(bias)  # (N, B, H)
                .mul(gating_weights)  # (B, H)
                .sum(dim=0)
            )
        return out

    def sample_best_action_and_coefficients(self, inputs):
        gating_weights = F.softmax(self.policy_gate(inputs[:, -self.num_gating_input:]), dim=1).t().unsqueeze(-1)
        # out = inputs[:, :-self.num_gating_input+self.num_additional_expert_input].clone()
        out = inputs[:, :].clone()

        for (weight, bias, activation) in self.policy_layers:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                .add(bias)  # (N, B, H)
                .mul(gating_weights)  # (B, H)
                .sum(dim=0)
            )
        return out, gating_weights

    def sample_actions(self, inputs):
        mu = self.sample_best_actions(inputs)
        log_std = self.get_log_stds(mu)
        #std = torch.exp(log_std)
        eps = torch.randn(mu.size(), device=device)
        actions = torch.clamp(mu + log_std.exp()*(eps), -1, 1)
        return actions, mu

    def set_noise(self, noise):
        self.noise = noise
        self.noise_tensor = torch.Tensor(self.noise).to(device)

    def get_value(self, inputs, device="cpu"):
        gating_weights = F.softmax(self.value_gate(inputs[:, -self.num_gating_input:]), dim=1).t().unsqueeze(-1)
        # out = inputs[:, :-self.num_gating_input+self.num_additional_expert_input].clone()
        out = inputs[:, :].clone()

        for (weight, bias, activation) in self.value_layers:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                .add(bias)  # (N, B, H)
                .mul(gating_weights)  # (B, H)
                .sum(dim=0)
            )
            
        # out = out.matmul(weight).add(bias).mul(gating_weights).sum(dim=0)
        return out

    def calculate_prob_gpu(self, inputs, actions):
        log_stds = self.get_log_stds(actions)
        mean_actions = self.sample_best_actions(inputs)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        return log_probs

    # def calculate_prob(self, inputs, actions):
    #     log_stds = self.get_log_stds(actions)
    #     #print(log_stds.shape)
    #     mean_actions = self.sample_best_actions(inputs)
    #     #print(mean_actions.shape)
    #     #w = self.get_w(inputs).to(device)
    #     numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
    #     log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
    #     #print(log_probs)
    #     #probs = (log_probs.exp() * w.t()).sum(dim=0).log()
    #     #print(probs)
    #     return log_probs

    def calculate_prob(self, inputs, actions, mean_actions):
        log_stds = self.get_log_stds(actions)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        return log_probs

class ActorCriticNetMixtureExpert(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], v_hidden_layer=[256, 256], w_hidden_layer=[256, 256], num_expert=1, num_contact=0):
        super(ActorCriticNetMixtureExpert, self).__init__()
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        self.num_expert = num_expert
        self.hidden_layer = hidden_layer
        self.v_hidden_layer = v_hidden_layer
        self.w_hidden_layer = w_hidden_layer
        self.experts = nn.ModuleList()
        for i in range(num_expert):
            expert_i = ActorCriticNet(num_inputs, num_outputs, hidden_layer)
            self.experts.append(expert_i)

        # self.v_fcs = nn.ModuleList()
        # v_fc = nn.Linear(num_inputs, self.v_hidden_layer[0])
        # self.v_fcs.append(v_fc)
        # for i in range(len(self.v_hidden_layer)-1):
        #     v_fc = nn.Linear(self.v_hidden_layer[i], self.v_hidden_layer[i+1])
        #     self.v_fcs.append(v_fc)
        # self.v = nn.Linear(self.v_hidden_layer[-1],1)

        self.w_fcs = nn.ModuleList()
        w_fc = nn.Linear(num_inputs, self.w_hidden_layer[0])
        self.w_fcs.append(w_fc)
        for i in range(len(self.w_hidden_layer)-1):
            w_fc = nn.Linear(self.w_hidden_layer[i], self.w_hidden_layer[i+1])
            self.w_fcs.append(w_fc)
        self.w = nn.Linear(self.w_hidden_layer[-1], num_expert)
        self.noise = -1
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)

    def get_value(self, inputs, device="cpu"):
        values = []
        for i in range(self.num_expert):
            value = self.experts[i].get_value(inputs)
            values.append(value)
        w = self.get_w(inputs)
        categorical = Categorical(w)
        pis = list(categorical.sample().data)
        return_v = torch.zeros(inputs.shape[0], 1)
        for i, j in enumerate(pis):
            return_v[i] = (values[j][i])
        return return_v.to(device)

    def forward(self, inputs):
        actions = self.get_mean_actions(inputs)
        
        # critic
        v = self.get_value(inputs)

        # w
        w = self.get_w(inputs)
        return actions, v, w

    def get_mean_actions(self, inputs):
        actions = []
        for i in range(self.num_expert):
            a = self.experts[i].get_action(inputs)[0]
            actions.append(a)
        #print(actions[0].shape)
        return actions

    def get_w(self, inputs):
        x = F.relu(self.w_fcs[0](inputs))
        for i in range(len(self.w_hidden_layer)-1):
            x = F.relu(self.w_fcs[i+1](x))
        w = self.w(x)
        #w = torch.ones(inputs.size()[0], self.num_expert)
        w = F.softmax(w, dim=-1)
        #print(w)
        return w

    def sample_actions(self, inputs):
        actions = self.get_mean_actions(inputs)
        #print("list", actions)
        w = self.get_w(inputs)
        categorical = Categorical(w)
        pis = list(categorical.sample().data)
        #print(pis)
        log_std = self.get_log_stds(actions[0])#Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(actions[0])
        #print(log_std)
        std = torch.exp(log_std)
        sample = Variable(std.data.new(std.size(0), std.size(1)).normal_())
        for i, j in enumerate(pis):
            sample[i] = sample[i].mul(std[i]).add(actions[j][i])
        #sample[:, 0:3] *= 0
        #sample[:, 11:17] *= 0
        #print("sample", sample)    
        return sample

    def sample_best_actions(self, inputs):
        actions = self.get_mean_actions(inputs)
        w = self.get_w(inputs)
        log_std = self.get_log_stds(actions[0])#log_stds = self.noise * torch.ones(self.num_outputs).expand_as(actions[0])
        #print(log_std)
        eps = torch.randn(actions[0].size())
        values, indices = torch.max(w, 1)
        #print(values ,indices)
        return actions[indices]# + eps * log_stds.exp()

    def calculate_prob(self, inputs, actions):
        log_stds = self.get_log_stds(actions)
        #print(log_stds.shape)
        mean_actions = torch.stack(self.get_mean_actions(inputs))
        #print(mean_actions.shape)
        w = self.get_w(inputs)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        #print(log_probs)
        probs = (log_probs.exp() * w.t()).sum(dim=0).log()
        #probs = log_probs.sum(dim=0)
        return probs

    def calculate_prob_gpu(self, inputs, actions):
        log_stds = self.get_log_stds(actions).to(device)
        #print(log_stds.shape)
        mean_actions = torch.stack(self.get_mean_actions(inputs))
        #print(mean_actions.shape)
        w = self.get_w(inputs).to(device)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        #print(log_probs)
        probs = (log_probs.exp() * w.t()).sum(dim=0).log()
        return probs

    def get_log_stds(self, actions):
        return Variable(torch.Tensor(self.noise)*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(actions)

        #return self.log_std.unsqueeze(0).expand_as(actions)

    def get_actions_difference(self, inputs):
        actions = self.get_mean_actions(inputs)
        difference_matrix = torch.zeros(self.num_expert, self.num_expert)
        for i in range(self.num_expert):
            for j in range(self.num_expert):
                difference_matrix[i, j] =  torch.max((actions[i]-actions[j])**2)
        return difference_matrix

    def set_noise(self, noise):
        self.noise = noise

class ActorCriticNetWithContact(ActorCriticNetMixtureExpert):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], v_hidden_layer=[256, 256], w_hidden_layer=[256, 256], num_contact=1):
        self.num_contact = num_contact
        num_expert = self.num_contact**2
        super().__init__(num_inputs, num_outputs, hidden_layer=hidden_layer, v_hidden_layer=v_hidden_layer, w_hidden_layer=w_hidden_layer, num_expert=num_expert)
    def get_w(self, inputs):
        # x = F.relu(self.w_fcs[0](inputs))
        # for i in range(len(self.w_hidden_layer)-1):
        #     x = F.relu(self.w_fcs[i+1](x))
        # w = self.w(x)
        # #w = torch.ones(inputs.size()[0], self.num_expert)
        # w = F.softmax(w, dim=-1)
        #print(w)
        w = torch.ones(inputs.size()[0], self.num_expert)
        #print(w.shape)
        if self.num_expert == 2:
            w[:, 0] = (inputs[:, -1] > 0.1)
            w[:, 1] = (inputs[:, -1] < 0.1)
        elif self.num_expert == 4:
            #print(inputs[:, -1].shape)
            w[:, 0] = ((inputs[:, -1] > 0.1) & (inputs[:, -2] > 0.1))
            w[:, 1] = ((inputs[:, -1] > 0.1) & (inputs[:, -2] < 0.1))
            w[:, 2] = ((inputs[:, -1] < 0.1) & (inputs[:, -2] > 0.1))
            w[:, 3] = ((inputs[:, -1] < 0.1) & (inputs[:, -2] < 0.1))
        w.float()
        return w


class ActorNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64]):
        super(ActorNet, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer
        self.p_fcs = nn.ModuleList()
        self.log_stds = nn.ModuleList()
        p_fc = nn.Linear(num_inputs, self.hidden_layer[0])
        log_std = nn.Linear(num_inputs, self.hidden_layer[0])
        self.p_fcs.append(p_fc)
        self.log_stds.append(log_std)
        for i in range(len(self.hidden_layer)-1):
            p_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            log_std = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            self.p_fcs.append(p_fc)
            self.log_stds.append(log_std)
        self.mu = nn.Linear(self.hidden_layer[-1], num_outputs)
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        self.noise = -2.0
        self.noises = torch.Tensor(num_outputs)
        self.log_std_linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, inputs):
        # actor
        x = F.relu(self.p_fcs[0](inputs))
        #log_std = F.relu(self.log_stds[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
            #log_std = F.relu(self.log_stds[i+1](log_std))
        #print(self.mu(x))
        mu = F.softsign(self.mu(x))
        #print(mu)
        log_std = Variable(self.noise * torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)
        #log_std.to(device)
        #log_std = torch.tanh((self.log_std_linear(inputs)))
        #log_std = torch.clamp(log_std, min=-2, max=2)
        return mu, log_std
    def sample_gpu(self, inputs):
        mean, log_std = self.forward(inputs)
        #mean.to(device)
        std = log_std.exp().to(device)
        eps = torch.randn(mean.shape).to(device)
        normal = Normal(mean, std)
        #action = normal.rsample()
        action = (mean + (std * eps).clamp(-0.5, 0.5)).clamp(-1.0, 1.0)#torch.clamp(normal.rsample(), -1.0, 1.0)
        return (action), 0, (mean), 0
    def sample(self, inputs):
        mean, log_std = self.forward(inputs)
        #mean.to(device)
        std = log_std.exp()
        eps = torch.randn(mean.shape)
        normal = Normal(mean, std)
        #action = normal.rsample()
        action = (mean + (std * eps).clamp(-0.5, 0.5)).clamp(-1.0, 1.0)#torch.clamp(normal.rsample(), -1.0, 1.0)
        return (action), 0, (mean), 0
    def set_noise(self, noise):
        self.noise = noise

class ValueNet(nn.Module):
    def __init__(self, num_inputs, hidden_layer=[64, 64]):
        super(ValueNet, self).__init__()
        self.hidden_layer = hidden_layer
        self.v_fcs = nn.ModuleList()
        v_fc = nn.Linear(num_inputs, self.hidden_layer[0])
        self.v_fcs.append(v_fc)
        for i in range(len(self.hidden_layer)-1):
            v_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            self.v_fcs.append(v_fc)
        self.v = nn.Linear(self.hidden_layer[-1],1)
    def forward(self, inputs):
        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return v

class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64]):
        super(QNet, self).__init__()
        self.hidden_layer = hidden_layer
        self.num_outputs = num_outputs
        self.q_fcs1 = nn.ModuleList()
        self.q_fcs2 = nn.ModuleList()
        q_fc1 = nn.Linear(num_inputs + num_outputs, self.hidden_layer[0])
        q_fc2 = nn.Linear(num_inputs + num_outputs, self.hidden_layer[0])
        self.q_fcs1.append(q_fc1)
        self.q_fcs2.append(q_fc2)
        for i in range(len(self.hidden_layer)-1):
            q_fc1 = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            q_fc2 = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            self.q_fcs1.append(q_fc1)
            self.q_fcs2.append(q_fc2)
        self.q_1 = nn.Linear(self.hidden_layer[-1],1)
        self.q_2 = nn.Linear(self.hidden_layer[-1],1)
    def forward(self, states, actions):
        inputs = torch.cat([states, actions], 1)
        q1 = F.relu(self.q_fcs1[0](inputs))
        q2 = F.relu(self.q_fcs2[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            q1 = F.relu(self.q_fcs1[i+1](q1))
            q2 = F.relu(self.q_fcs2[i+1](q2))
        q1 = (self.q_1(q1))
        q2 = (self.q_2(q2))
        return q1, q2


class Shared_grad_buffers():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] = torch.ones(p.size()).share_memory_()

    def add_gradient(self, model):
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] += p.grad.data

    def reset(self):
        for name,grad in self.grads.items():
            self.grads[name].fill_(0)

class Shared_obs_stats():
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs).share_memory_()
        self.mean = torch.zeros(num_inputs).share_memory_()
        self.mean_diff = torch.zeros(num_inputs).share_memory_()
        self.std = torch.ones(num_inputs).share_memory_()
        self.num_inputs = num_inputs
        self.sum = torch.zeros(num_inputs).share_memory_()
        self.sum_sqr = torch.zeros(num_inputs).share_memory_()

    def observes(self, obs):
        # observation mean var updates
        x = obs.data.squeeze()
        if True:
            self.n += 1.
            last_mean = self.mean.clone()
            self.sum = self.sum + x
            self.sum_sqr += x.pow(2)
            self.mean = self.sum / self.n
            self.std = (self.sum_sqr / self.n - self.mean.pow(2)).clamp(1e-2,1e9).sqrt()
            self.mean = self.mean.float()
            self.std = self.std.float()
        #self.mean = (self.mean * self.n + x) / self.
            #self.mean += (x-self.mean)/self.n
            #self.mean_diff += (x-last_mean)*(x-self.mean)
            #self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        #if (inputs.shape[1]) > self.num_inputs:
        #    inputs = inputs[:, 0:self.num_inputs]
        obs_mean = Variable(self.mean.unsqueeze(0).expand_as(inputs[:, 0:self.num_inputs]))
        obs_std = Variable(self.std.unsqueeze(0).expand_as(inputs[:, 0:self.num_inputs]))
        obs_mean = ((inputs[:, 0:self.num_inputs] - obs_mean) / obs_std)
        if (inputs.shape[1]) > self.num_inputs:
            #print(inputs.shape, obs_mean.shape)
            obs_mean = torch.cat([obs_mean, inputs[:, self.num_inputs:self.num_inputs+1]], dim=1)
        #print("outout", obs_mean)
        #obs_std = Variable(torch.sqrt(self.var).unsqueeze(0).expand_as(inputs))
        return torch.clamp(obs_mean, -10.0, 10.0)

    def reset(self):
        self.n = torch.zeros(self.num_inputs).share_memory_()
        self.mean = torch.zeros(self.num_inputs).share_memory_()
        self.mean_diff = torch.zeros(self.num_inputs).share_memory_()
        self.var = torch.zeros(self.num_inputs).share_memory_()