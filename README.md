
# Baselines


## Saving, loading and visualizing models
The algorithms serialization API is not properly unified yet; however, there is a simple method to save / restore trained models.
`--save_path` and `--load_path` command-line option loads the tensorflow state from a given path before training, and saves it after the training, respectively.
Let's imagine you'd like to train ppo2 on Atari Pong,  save the model and then later visualize what has it learnt.
```bash
python -m baselines.run --alg=ppo2 --env=tcn-push-v0 --num_timesteps=2e7 --save_path=~/models/pong_20M_ppo2
```
This should get to the mean reward per episode about 20. To load and visualize the model, we'll do the following - load the model, train it for 0 steps, and then visualize:
```bash
python -m baselines.run --alg=ppo2 --env=tcn-push-v0--num_timesteps=0 --load_path=~/models/pong_20M_ppo2 --play
```

*NOTE:* At the moment Mujoco training uses VecNormalize wrapper for the environment which is not being saved correctly; so loading the models trained on Mujoco will not work well if the environment is recreated. If necessary, you can work around that by replacing RunningMeanStd by TfRunningMeanStd in [baselines/common/vec_env/vec_normalize.py](baselines/common/vec_env/vec_normalize.py#L12). This way, mean and std of environment normalizing wrapper will be saved in tensorflow variables and included in the model file; however, training is slower that way - hence not including it by default

## Loading and vizualizing learning curves and other training metrics
See [here](docs/viz/viz.ipynb) for instructions on how to load and display the training data.

## Subpackages

- [A2C](baselines/a2c)
- [ACER](baselines/acer)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [GAIL](baselines/gail)
- [HER](baselines/her)
- [PPO1](baselines/ppo1) (obsolete version, left here temporarily)
- [PPO2](baselines/ppo2)
- [TRPO](baselines/trpo_mpi)
