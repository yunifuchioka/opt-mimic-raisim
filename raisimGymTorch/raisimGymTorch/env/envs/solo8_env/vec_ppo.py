import argparse
import os
import sys
import random
import numpy as np
import scipy

from params import Params

import pickle
import time

import statistics
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from operator import add, sub
import pickle
import pandas as pd


class PPOStorage:
    def __init__(self, num_inputs, num_outputs, max_size=64000):
        self.states = torch.zeros(max_size, num_inputs).to(device)
        self.next_states = torch.zeros(max_size, num_inputs).to(device)
        self.actions = torch.zeros(max_size, num_outputs).to(device)
        self.dones = torch.zeros(max_size, 1, dtype=torch.int8).to(device)
        self.log_probs = torch.zeros(max_size).to(device)
        self.rewards = torch.zeros(max_size).to(device)
        self.q_values = torch.zeros(max_size, 1).to(device)
        self.mean_actions = torch.zeros(max_size, num_outputs).to(device)
        self.counter = 0
        self.sample_counter = 0
        self.max_samples = max_size

    def sample(self, batch_size):
        # idx = np.random.choice(self.counter, batch_size, replace=False)
        # idx = torch.from_numpy(idx).to(device)
        idx = torch.randint(self.counter, (batch_size,), device=device)
        return self.states[idx, :], self.actions[idx, :], self.next_states[idx,
                                                                           :], self.rewards[idx], self.q_values[idx, :], self.log_probs[idx]

    def clear(self):
        self.counter = 0

    def push(
            self,
            states,
            actions,
            next_states,
            rewards,
            q_values,
            log_probs,
            size):
        self.states[self.counter:self.counter +
                    size, :] = states.detach().clone()
        self.actions[self.counter:self.counter +
                     size, :] = actions.detach().clone()
        self.next_states[self.counter:self.counter +
                         size, :] = next_states.detach().clone()
        self.rewards[self.counter:self.counter +
                     size] = rewards.detach().clone()
        self.q_values[self.counter:self.counter +
                      size, :] = q_values.detach().clone()
        self.log_probs[self.counter:self.counter +
                       size] = log_probs.detach().clone()
        # self.mean_actions[self.counter:self.counter+size, :] = mean_actions.detach().clone()
        self.counter += size

    def critic_sample(self, batch_size):
        # idx = np.random.choice(self.counter, batch_size, replace=False)
        # idx = torch.from_numpy(idx).to(device)
        # return self.states[idx, :], self.q_values[idx, :]
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter += batch_size
        self.sample_counter %= self.max_samples
        return self.states[self.sample_counter - batch_size:self.sample_counter,
                           :], self.q_values[self.sample_counter - batch_size:self.sample_counter, :]

    def actor_sample(self, batch_size):
        # idx = np.random.choice(self.counter, batch_size, replace=False)
        # idx = torch.from_numpy(idx).to(device)
        # return self.states[idx, :], self.actions[idx, :], self.q_values[idx,
        # :], self.log_probs[idx]
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter += batch_size
        self.sample_counter %= self.max_samples
        return self.states[self.sample_counter -
                           batch_size:self.sample_counter, :], self.actions[self.sample_counter -
                                                                            batch_size:self.sample_counter, :], self.q_values[self.sample_counter -
                                                                                                                              batch_size:self.sample_counter, :], self.log_probs[self.sample_counter -
                                                                                                                                                                                 batch_size:self.sample_counter]

    def permute(self):
        permuted_index = torch.randperm(self.max_samples)
        self.states[:, :] = self.states[permuted_index, :]
        self.actions[:, :] = self.actions[permuted_index, :]
        self.q_values[:, :] = self.q_values[permuted_index, :]
        self.log_probs[:] = self.log_probs[permuted_index]


class RL(object):
    def __init__(self, env, hidden_layer=[64, 64]):
        self.env = env
        #self.env.env.disableViewer = False
        self.num_inputs = env.observation_space.shape[0]
        self.num_outputs = env.action_space.shape[0]
        self.hidden_layer = hidden_layer

        self.params = Params()
        self.Net = ActorCriticNet
        self.model = self.Net(
            self.num_inputs,
            self.num_outputs,
            self.hidden_layer)
        self.model.share_memory()
        self.shared_obs_stats = Shared_obs_stats(self.num_inputs)
        self.test_mean = []
        self.test_std = []

        self.noisy_test_time = []
        self.noisy_test_mean = []
        self.noisy_test_std = []
        self.noisy_test_min = []
        self.noisy_test_max = []
        self.lr = self.params.lr
        plt.show(block=False)

        self.test_list = []

        self.best_score_queue = mp.Queue()
        self.best_score = mp.Value("f", 0)
        self.max_reward = mp.Value("f", 5)

        self.best_validation = 1.0
        self.current_best_validation = 1.0

        self.return_obs_stats = Shared_obs_stats(1)

        self.gpu_model = self.Net(
            self.num_inputs,
            self.num_outputs,
            self.hidden_layer)
        self.gpu_model.to(device)
        self.model_old = self.Net(
            self.num_inputs,
            self.num_outputs,
            self.hidden_layer).to(device)

        self.base_controller = None
        self.base_policy = None

        self.total_rewards = []

    def normalize_data(self, num_iter=1000, file='shared_obs_stats.pkl'):
        state = self.env.reset()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        for i in range(num_iter):
            print(i)
            self.shared_obs_stats.observes(state)
            state = self.shared_obs_stats.normalize(state)  # .to(device)
            #mu = self.model.sample_actions(state)
            # action = mu#(mu + log_std.exp()*Variable(eps))
            #env_action = action.cpu().data.squeeze().numpy()
            env_action = np.random.randn(self.num_outputs)
            state, reward, done, _ = self.env.step(env_action * 0)

            if done:
                state = self.env.reset()

            state = Variable(torch.Tensor(state).unsqueeze(0))

        with open(file, 'wb') as output:
            pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)

    def run_test(self, num_test=1):
        state = self.env.reset()
        ave_test_reward = 0

        total_rewards = []
        if self.num_envs > 1:
            test_index = 1
        else:
            test_index = 0

        for i in range(num_test):
            total_reward = 0
            while True:
                state = self.shared_obs_stats.normalize(state)
                mu = self.gpu_model.sample_best_actions(state)
                state, reward, done, _ = self.env.step(mu)
                total_reward += reward[test_index].item()

                if done[test_index]:
                    state = self.env.reset()
                    # print(self.env.position)
                    # print(self.env.time)
                    ave_test_reward += total_reward / num_test
                    total_rewards.append(total_reward)
                    break
        #print("avg test reward is", ave_test_reward)
        reward_mean = statistics.mean(total_rewards)
        reward_std = statistics.stdev(total_rewards)
        self.test_mean.append(reward_mean)
        self.test_std.append(reward_std)
        self.test_list.append((reward_mean, reward_std))
        # print(self.model.state_dict())

    def run_test_with_noise(self):

        reward_mean = statistics.mean(self.total_rewards)
        reward_std = statistics.stdev(self.total_rewards)
        reward_max = max(self.total_rewards)
        reward_min = min(self.total_rewards)
        #print(reward_mean, reward_std, self.total_rewards)
        self.noisy_test_time.append((time.time() - training_start)/60.0)
        self.noisy_test_mean.append(reward_mean)
        self.noisy_test_std.append(reward_std)
        self.noisy_test_min.append(reward_min)
        self.noisy_test_max.append(reward_max)

        print("Reward Mean: {}".format(reward_mean))
        print("Reward Std: {}".format(reward_std))

        return reward_mean

    def save_reward_stats(self, stats_name):
        with open(stats_name, 'wb') as f:
            tosave = np.vstack((self.noisy_test_mean, self.noisy_test_std))
            np.save(f, np.array(tosave))

    def plot_statistics(self):

        # check new version...
        noisy_test_mean = np.array(self.noisy_test_mean)
        noisy_test_std = np.array(self.noisy_test_std)
        noisy_low = noisy_test_mean - noisy_test_std
        noisy_high = noisy_test_mean + noisy_test_std
        index = np.arange(len(noisy_test_mean))*100
        
        plt.clf()
        plt.plot(index, noisy_test_mean, color='b')
        plt.fill_between(index, noisy_low, noisy_high, color='b', alpha=0.2)
        plt.xlabel('Training Iterations')
        plt.ylabel('Rewards')
        plt.title(self.experiment_name)
        plt.grid('True')
        plt.savefig(self.save_dir + "{}_test.png".format(self.experiment_name))

    def save_csv(self):
        data = {'iteration': list(np.arange(len(self.noisy_test_mean))*100),
                'elapsed time': self.noisy_test_time,
                'mean reward': self.noisy_test_mean,
                'stddev reward': self.noisy_test_std,
                'min reward': self.noisy_test_min,
                'max reward': self.noisy_test_max}
        df = pd.DataFrame(data)
        df.to_csv(self.save_dir + "{}_stats.csv".format(self.experiment_name), index=False)
        



    def collect_samples_vec(
            self,
            num_samples,
            start_state=None,
            noise=-2.5,
            env_index=0,
            random_seed=1):

        # if start_state == None:
        #     start_state = self.env.reset()
        start_state = self.env.observe()
        samples = 0
        done = False
        states = []
        next_states = []
        actions = []
        # mean_actions = []
        rewards = []
        values = []
        q_values = []
        real_rewards = []
        log_probs = []
        dones = []
        noise = self.base_noise * self.explore_noise.value
        self.gpu_model.set_noise(noise)

        state = start_state
        total_reward1 = 0
        total_reward2 = 0
        calculate_done1 = False
        calculate_done2 = False
        self.total_rewards = []
        start = time.time()
        while samples < num_samples:
            with torch.no_grad():
                action, mean_action = self.gpu_model.sample_actions(state)
                log_prob = self.gpu_model.calculate_prob(
                    state, action, mean_action)

            states.append(state.clone())
            actions.append(action.clone())
            log_probs.append(log_prob.clone())
            state, reward, done, _ = self.env.step(action)

            rewards.append(reward.clone())

            dones.append(done.clone())

            next_states.append(state.clone())
            #next_state = self.shared_obs_stats.normalize(state)

            samples += 1

            self.env.reset_time_limit()
        #print("sim time", time.time() - start)
        #import ipdb; ipdb.set_trace()
        start = time.time()
        counter = num_samples - 1
        R = self.gpu_model.get_value(state)
        #import ipdb; ipdb.set_trace()
        while counter >= 0:
            #print(R, dones[counter])
            R = R * (1 - dones[counter].unsqueeze(-1))
            R = 0.995 * R + rewards[counter].unsqueeze(-1)
            q_values.insert(0, R)
            counter -= 1
            # print(len(q_values))
        for i in range(num_samples):
            self.storage.push(
                states[i],
                actions[i],
                next_states[i],
                rewards[i],
                q_values[i],
                log_probs[i],
                self.num_envs)
        self.total_rewards = self.env.total_rewards.cpu().numpy().tolist()
        #print("processing time", time.time() - start)

    def fresh_update(self):
        fresh_model = self.Net(
            self.num_inputs,
            self.num_outputs,
            self.hidden_layer).to(device)
        fresh_model.train()
        optimizer = optim.Adam(fresh_model.parameters(), lr=3e-4)
        storage = self.storage
        for k in range(1000):
            batch_states, batch_actions, batch_q_values, batch_log_probs = storage.actor_sample(
                20000 // 4)
            loss_value = (
                fresh_model.get_value(batch_states) -
                self.gpu_model.get_value(batch_states))**2
            loss_value = loss_value.mean()
            loss_action = (
                fresh_model.sample_best_actions(batch_states) -
                self.gpu_model.sample_best_actions(batch_states))**2
            loss_action = loss_action.mean()
            loss_total = loss_value + 100 * loss_action
            print("action loss", loss_action)
            print("value loss", loss_value)
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
        self.gpu_model.load_state_dict(fresh_model.state_dict())

    def update_critic(self, batch_size, num_epoch, fixed_gating):
        self.gpu_model.train()
        # if fixed_gating:
        #   optimizer = optim.Adam(self.gpu_model.critic_params, lr=10*self.lr)
        # else:
        #   optimizer = optim.Adam(self.gpu_model.value_gate.parameters(), lr=10*self.lr)
        optimizer = optim.Adam(self.gpu_model.parameters(), lr=10 * self.lr)

        storage = self.storage
        gpu_model = self.gpu_model

        for k in range(num_epoch):
            batch_states, batch_q_values = storage.critic_sample(batch_size)
            batch_q_values = batch_q_values  # / self.max_reward.value
            v_pred = gpu_model.get_value(batch_states)

            loss_value = (v_pred - batch_q_values)**2
            loss_value = 0.5 * loss_value.mean()

            # if not fixed_gating:
            #  loss_value -= 0.01 * gpu_model.evaluate_value_gate_l2(batch_states)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

    def update_actor(self, batch_size, num_epoch, fixed_gating):
        # model_old = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer).to(device)
        # self.model_old.load_state_dict(self.gpu_model.state_dict())
        # self.model_old.set_noise(self.model.noise)
        self.gpu_model.train()
        # if fixed_gating:
        #   optimizer = optim.Adam(self.gpu_model.actor_params, lr=self.lr)
        # else:
        #   optimizer = optim.Adam(self.gpu_model.policy_gate.parameters(), lr=self.lr)
        optimizer = optim.Adam(self.gpu_model.parameters(), lr=self.lr)

        storage = self.storage
        gpu_model = self.gpu_model
        model_old = self.model_old
        params_clip = self.params.clip

        for k in range(num_epoch):
            batch_states, batch_actions, batch_q_values, batch_log_probs = storage.actor_sample(
                batch_size)

            batch_q_values = batch_q_values  # / self.max_reward.value

            with torch.no_grad():
                v_pred_old = gpu_model.get_value(batch_states)

            batch_advantages = (batch_q_values - v_pred_old)

            probs = gpu_model.calculate_prob_gpu(batch_states, batch_actions)
            # model_old.calculate_prob_gpu(batch_states, batch_actions)
            probs_old = batch_log_probs
            ratio = (probs - (probs_old)).exp()
            ratio = ratio.unsqueeze(1)
            surr1 = ratio * batch_advantages
            surr2 = ratio.clamp(
                1 - params_clip,
                1 + params_clip) * batch_advantages
            loss_clip = -(torch.min(surr1, surr2)).mean()

            total_loss = loss_clip  # + mirror_loss
            # if not fixed_gating:
            #  total_loss -= 0.01 * gpu_model.evaluate_policy_gate_l2(batch_states)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        # print(self.shared_obs_stats.mean.data)
        if self.lr > 1e-4:
            self.lr *= 0.99
        else:
            self.lr = 1e-4

    def save_model(self, filename):
        torch.save(self.gpu_model.state_dict(), filename)

    def save_shared_obs_stas(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)

    def save_statistics(self, filename):
        statistics = [
            self.time_passed,
            self.num_samples,
            self.test_mean,
            self.test_std,
            self.noisy_test_mean,
            self.noisy_test_std]
        with open(filename, 'wb') as output:
            pickle.dump(statistics, output, pickle.HIGHEST_PROTOCOL)

    def collect_samples_multithread(self):
        #queue = Queue.Queue()
        import time
        self.num_envs = 200
        self.start = time.time()
        self.lr = 1e-4
        self.weight = 10
        num_threads = 1
        self.num_samples = 0
        self.time_passed = 0
        score_counter = 0
        total_thread = 0
        max_samples = self.num_envs * 100
        self.storage = PPOStorage(
            self.num_inputs,
            self.num_outputs,
            max_size=max_samples)
        seeds = [
            i * 100 for i in range(num_threads)
        ]
        self.explore_noise = mp.Value("f", -2.0)
        self.base_noise = np.ones(self.num_outputs)
        self.base_noise[0:14] = 1.25
        self.base_noise[14:28] = 1.25  # 1.15
        noise = self.base_noise * self.explore_noise.value
        self.model.set_noise(noise)
        self.gpu_model.set_noise(noise)
        self.env.reset()
        best_reward_mean = -np.inf
        for iterations in range(100000):
            iteration_start = time.time()
            #print(self.save_dir)
            while self.storage.counter < max_samples:
                self.collect_samples_vec(100, noise=noise)
                # print("memeory length", self.storage.counter)
            # print("memeory length", self.storage.counter)
            start = time.time()

            fixed_gating = False
            self.update_critic(max_samples // 4, 40, fixed_gating)
            #self.update_critic(max_samples//4, 40, True)
            self.update_actor(max_samples // 4, 40, fixed_gating)
            #self.update_actor(max_samples//4, 40, True)
            self.storage.clear()

            if (iterations) % 100 == 0:
                print("\nIteration: {}".format(iterations))
                current_reward_mean = self.run_test_with_noise()
                self.plot_statistics()
                self.save_csv()
                # plt.savefig("{}_test.png".format(self.experiment_name))
            # if self.explore_noise.value > -2.5:
            #     self.explore_noise.value *= 1.001
            #     self.gpu_model.set_noise(self.explore_noise.value * self.base_noise)
            #print("update policy time", time.time()-start)
            # if self.env.task.reward_termination and iterations > 200:
            #     self.env.task.reward_termination = False
            #print("iteration time", iterations, time.time()-iteration_start)

            if (iterations) % 500 == 0:
                self.save_model(self.save_dir + "iter%d.pt" % (iterations))
                self.save_model(self.save_dir + "latest.pt")
                self.save_reward_stats(self.save_dir + "{}_reward_stats.npy".format(self.experiment_name))
                print("Current mean: {}".format(current_reward_mean))
                print("Best mean: {}".format(best_reward_mean))
                if current_reward_mean > best_reward_mean:
                    self.save_model(self.save_dir + "best.pt")
                    best_reward_mean = current_reward_mean
                    print("Updating best.pt")
            
            # if (iterations+1) % 25 == 0:
            #     self.env.reset()
            # if (iterations+1) % 300 == 0:
            #     self.env.task.v_d += 0.1

        self.save_reward_stats(self.save_dir + "{}_reward_stats.npy".format(self.experiment_name))
        self.save_model(self.save_dir + "final.pt")
        current_reward_mean = self.run_test_with_noise()
        if current_reward_mean > best_reward_mean:
            self.save_model(self.save_dir + "best.pt")
            best_reward_mean = current_reward_mean
            print("Updating best.pt")
        self.plot_statistics()
        self.save_csv()

    def add_env(self, env):
        self.env_list.append(env)


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    import json
    from ruamel.yaml import YAML, dump, RoundTripDumper
    from raisimGymTorch.env.bin import solo8_env
    from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecTorchEnv as VecEnv
    from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher

    import torch
    import torch.optim as optim
    import torch.multiprocessing as mp
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    import torch.utils.data
    from model import ActorCriticNet, ActorCriticNetMann, Shared_obs_stats, ActorCriticNetFF
    import argparse
    from datetime import datetime
    # TODO: uncomment this... only done to avoid using up others' GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    # get experiment name
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="experiment name", default=None)
    args = parser.parse_args()
    if args.name is None:
        args.name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    seed = 1  # 8
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    # directories
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + "/../../../../.."

    # config
    cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

    # create environment from the configuration file
    env = VecEnv(
        solo8_env.RaisimGymEnv(
            home_path + "/rsc",
            dump(
                cfg['environment'],
                Dumper=RoundTripDumper)),
        cfg['environment'])
    ppo = RL(env, [128, 128])

    ppo.base_dim = ppo.num_inputs

    ppo.save_dir = "stats/{}/".format(args.name)
    ppo.experiment_name = args.name
    # create results dir if not there already
    if not os.path.exists(ppo.save_dir):
        os.makedirs(ppo.save_dir)
    # ppo.gpu_model.load_state_dict(torch.load("stats/soccer3D_around_the_world//iter6999.pt"))
    # ppo.model.load_state_dict(torch.load("stats/simple_model_walk_Jan05/iter1300.pt"))
    ppo.max_reward.value = 1  # 50

    # ppo.save_model(ppo.save_dir)
    training_start = time.time()
    ppo.collect_samples_multithread()
    print("Total training time: {}".format(time.time() - training_start))

    #ppo.start = t.time()
