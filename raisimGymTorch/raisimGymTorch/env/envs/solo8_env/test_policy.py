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
    from model import ActorCriticNet, Shared_obs_stats
    import os
    import numpy as np
    import time
    import argparse
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs='?', help=".pt file name. no need to include \"stats/\"")
    args = parser.parse_args()

    seed = 1  # 8
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    # directories
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + "/../../../../.."

    # config
    cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

    # manually change config parameters
    cfg['environment']['num_envs'] = 1 # no parallel simulation
    # cfg['environment']['disable_termination'] = True # disable early termination

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

    if args.filename is not None:
        model = ActorCriticNet(num_inputs, num_outputs, [128, 128])
        model.load_state_dict(
            torch.load(
                "stats/{}".format(args.filename),
                map_location=torch.device(device)))
        model.to(device)
        model.set_noise(-2.5 * np.ones(num_outputs))

    sleep_time = cfg['environment']['control_dt']

    env.reset()
    obs = env.observe()
    time.sleep(1.0) # give time for GUI to start before motion does
    tic = time.time()
    for i in range(10000000):
        if args.filename is not None:
            with torch.no_grad():
                act = model.sample_best_actions(
                    obs + torch.randn_like(obs).mul(0.0))
        else:
            act = torch.zeros((num_envs, num_outputs))
        obs, rew, done, _ = env.step(act)
        env.reset_time_limit()

        toc = time.time() - tic
        time.sleep(max(sleep_time - toc, 0))
        # print(time.time() - tic)
        tic = time.time()
