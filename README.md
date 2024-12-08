# RL³: Boosting Meta Reinforcement Learning via RL inside RL². 

Source code for the paper [RL³: Boosting Meta Reinforcement Learning via RL inside RL²]

Authors: Abhinav Bhatia, Samer B. Nashed, and Shlomo Zilberstein


## Installation

Install Julia version 1.11.x from https://julialang.org/downloads/

In the root folder of this project:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running Code

### Scripts

```bash
# Train RL^2 for Bandits H=512 for 5000 episodes
julia --project -t auto main.jl bandits rl2 5000 --suffix train --seed 0 --nactions 5 --horizon 512 --inference_device cpu --include_time_context concat --batch_size 32768 --minibatch_size 4096 --ent_bonus 0.1 --decay_ent_bonus --lr 0.0003 --no_multithreading --progressmeter

# Train RL^2 for MDPs H=512 for 10000 episodes
julia --project -t auto main.jl mdps rl2 10000 --suffix train --seed 0 --nstates=10 --nactions 5 --horizon 512 --inference_device cpu --include_time_context concat --batch_size 32768 --minibatch_size 4096 --ent_bonus 0.1 --decay_ent_bonus --lr 0.0003 --progressmeter

# Train RL^2 for Gridworld 11x11 H=250 for 20000 episodes
julia --project -t auto main.jl gridworlds rl2 20000 --suffix train --variation 11x11 --seed 0 --horizon 250 --include_time_context concat --batch_size 32000 --minibatch_size 4000 --ent_bonus 0.04 --lr 0.0002 --laplace_smoothing=0.01 --video --video_interval 100 --progressmeter

# Train RL^2 for Gridworld 13x13 H=350 for 20000 episodes
julia --project -t auto main.jl gridworlds rl2 20000 --suffix train --variation 13x13 --seed 0 --horizon 350 --include_time_context concat --batch_size 33600 --minibatch_size 4200 --ent_bonus 0.04 --lr 0.0002 --laplace_smoothing=0.01 --video --video_interval 100 --progressmeter
```

You can specify `rl3` instead of `rl2` after the problem name to train RL^3 in the above commands. Specify `rl3_coarse` for state abstractions. Specify `--enable_wandb` to log to Weights and Biases. You will need to install `PythonCall` (via `Pkg.add("PythonCall")`) to use this feature. Also, it is suggested to use inference_device as `gpu` for context lengths > 1024.

The above will save models in `models/` directory, plots in `plots/` directory and videos in `videos/` directory. Model and video save interval (in terms of ppo iterations) can be controlled using `--model_save_interval` and `--video_interval` respectively.

For testing trained models, do something like:

```bash
julia --project -t auto main.jl mdps rl2 0 --suffix test --nstates=10 --nactions 5 --horizon 512 --include_time_context concat --test_model <model_path> --test_episodes 1000 --parallel_testing
```


For ood testing for MDPs and Bandits, pass `--ood` flag to the above command. For ood testing Gridworlds 13x13, pass `13x13_dense`, `13x13_deterministic`, `13x13_watery`, `13x13_dangerous`, `13x13_corner` for the `--variation` argument.