if __name__ == '__main__':
    import json
    from ruamel.yaml import YAML, dump, RoundTripDumper
    from raisimGymTorch.env.bin import solo8_env
    from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecTorchEnv as VecEnv
    from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
    import glob

    import matplotlib.pyplot as plt
    # plt.style.use('seaborn')
    plt.rc("figure", titlesize=15)
    plt.rc("legend", fontsize=15)
    plt.rc("axes", labelsize=15)
    plt.rc("axes", titlesize=15)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)

    import torch
    import torch.optim as optim
    import torch.multiprocessing as mp
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    import torch.utils.data
    from model import ActorCriticNet, Shared_obs_stats
    import os
    import numpy as np
    import time
    import argparse
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="experiment/folder name", type=str, default=None)
    parser.add_argument("-s", "--steps", help="maximum steps to run episodes for", type=int, default=1000)
    args = parser.parse_args()

    seed = 4  # 8
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    # directories
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + "/../../../../.."

    # config
    cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

    # manually change config parameters
    # cfg['environment']['num_envs'] = 1 # no parallel simulation
    # cfg['environment']['disable_termination'] = True # disable early termination
    cfg['environment']['port'] = 8082

    # create environment from the configuration file
    env = VecEnv(
        solo8_env.RaisimGymEnv(
            home_path + "/rsc",
            dump(
                cfg['environment'],
                Dumper=RoundTripDumper)),
        cfg['environment'])
    print("env_created")

    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    num_envs = cfg['environment']['num_envs']

    if args.name is None:
        raise NotImplementedError() # todo

    network_fnames = glob.glob("stats/" + args.name + "/iter*")
    network_fnames.sort()

    mean_dict = {}
    min_dict = {}
    max_dict = {}
    std_dict = {}
    dist_dict = {}

    for network_fname in network_fnames:
        network_iter_number = int(network_fname.split("iter")[1].split(".pt")[0])
        print(network_iter_number)

        model = ActorCriticNet(num_inputs, num_outputs, [128, 128])
        model.load_state_dict(
            torch.load(
                network_fname,
                map_location=torch.device(device)))
        model.to(device)
        model.set_noise(-2.5 * np.ones(num_outputs))
        
        done_flag = torch.zeros(num_envs, dtype=torch.bool, device="cpu")
        total_reward = torch.zeros(num_envs, device="cpu")
        
        env.reset()
        obs = env.observe()
        time.sleep(1.0) # give time for GUI to start before motion does

        for i in range(args.steps):
            with torch.no_grad():
                act = model.sample_best_actions(obs)

            obs, rew, done, _ = env.step(act)

            done_flag = torch.logical_or(done_flag, done.cpu())
            # done_flag = torch.logical_or(done_flag, total_reward > env.total_rewards.cpu())
            
            # total_reward[done_flag.logical_not()] = env.total_rewards[done_flag.logical_not()].cpu()
            total_reward[done_flag.logical_not()] += rew[done_flag.logical_not()].cpu()

            if i == args.steps - 1 or all(done_flag):
                # print(network_fname)
                # print(network_iter_number)
                # print("min: ", total_reward.numpy().min())
                # print("mean: ", total_reward.numpy().mean())
                # print("max: ", total_reward.numpy().max())
                # print("std: ", total_reward.numpy().std())
                mean_dict[network_iter_number] = total_reward.numpy().mean()
                std_dict[network_iter_number] = total_reward.numpy().std()
                min_dict[network_iter_number] = total_reward.numpy().min()
                max_dict[network_iter_number] = total_reward.numpy().max()
                dist_dict[network_iter_number] = total_reward.numpy()
                break
            
            env.reset_time_limit()
    
    x_axis = []
    rew_mean = []
    rew_std = []
    rew_min = []
    rew_max = []
    rew_dist = []
    for iter_idx in range(0, 100000, 500):
        x_axis.append(iter_idx)
        rew_mean.append(mean_dict[iter_idx])
        rew_std.append(std_dict[iter_idx])
        rew_min.append(min_dict[iter_idx])
        rew_max.append(max_dict[iter_idx])
        rew_dist.append(dist_dict[iter_idx])
        if iter_idx == max(mean_dict.keys()):
            break

    # convert list into numpy array
    x_axis = np.array(x_axis)
    rew_mean = np.array(rew_mean)
    rew_std = np.array(rew_std)
    rew_min = np.array(rew_min)
    rew_max = np.array(rew_max)
    rew_dist = np.array(rew_dist)

    save_dir = "training_curves/{}/".format(args.name)

    # create save dir if not there already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(save_dir + "x_axis", x_axis)
    np.save(save_dir + "rew_mean", rew_mean)
    np.save(save_dir + "rew_std", rew_std)
    np.save(save_dir + "rew_min", rew_min)
    np.save(save_dir + "rew_max", rew_max)
    np.save(save_dir + "rew_dist", rew_dist)

    plt.plot(x_axis, rew_mean, color='b')
    plt.fill_between(x_axis, rew_mean - rew_std, rew_mean + rew_std, color='b', alpha=0.2)
    plt.fill_between(x_axis, rew_min, rew_max, color='b', alpha=0.1)
    plt.title(args.name)
    plt.grid("True")
    plt.savefig(save_dir + "training_curve.png")
