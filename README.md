# opt-mimic-raisim

Reinforcement Learning (RL) code for the paper "OPT-Mimic: Imitation of Optimized Trajectories for Dynamic Quadruped Behaviors". This repository was modified from the [raisimLib](https://github.com/raisimTech/raisimlib) repository. Modifications include the addition of a URDF file for the ODRI Solo 8 robot to `rsc/solo8_v7/`, and the addition of RL environment and training code to `raisimGymTorch/raisimGymTorch/env/envs/solo8_env/`.

## Installation
The installation steps from [Raisim](https://raisim.com/) should be followed first. Then this repo can be cloned into your `$WORKSPACE` directory.

## Training RL Policies
- After any changes to C++ code, go to `raisimGymTorch/` and run `python setup.py develop`
- Go to `raisimGymTorch/raisimGymTorch/env/envs/solo8_env/` and run `python vec_ppo.py -n my-experiment-name` to start RL training
- Files indicating training progress will be saved to `raisimLibSolo/raisimGymTorch/raisimGymTorch/env/envs/solo8_env/stats/my-experiment-name/`
- The reference motion to track can be specified in the `ref_filename` argument of `raisimLibSolo/raisimGymTorch/raisimGymTorch/env/envs/solo8_env/cfg.yaml`. this should correspond to a filename in `raisimGymTorch/raisimGymTorch/env/envs/solo8_env/traj/`, which includes reference motion csv files produced from trajectory optimization
- Note that Raisim comes bundled with RL training code building off of OpenAI Stable Baselines, but this is unused and instead a custom implementation of RL training code is used here

## Testing a trained RL policy
- Go to `raisimGymTorch/raisimGymTorch/env/envs/solo8_env/` and run `python test_policy.py my-experiment-name/latest.pt` to run the latest policy trained using `python vec_ppo.py -n my-experiment-name`
- Early termination, which is important during RL training, can be turned off during testing by setting `cfg['environment']['disable_termination']` to true in `raisimGymTorch/raisimGymTorch/env/envs/solo8_env/test_policy.py` (line 40 as of writing).