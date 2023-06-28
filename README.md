# RL³: Boosting Meta Reinforcement Learning via RL inside RL². 

Source code for the paper RL³: Boosting Meta Reinforcement Learning via RL inside RL².

Authors: Abhinav Bhatia, Samer B. Nashed, and Shlomo Zilberstein

## Installation

Install Julia version 1.9.x from https://julialang.org/downloads/

In the root folder of this project:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running Code

### RL^2 and RL^3 for Bandits environments

```bash

# Train RL^2 for Bandits H=100 for 5000 episodes
julia --project=. main.jl bandits rl2 5000 --suffix trainH100 --seed 0 --nactions 5 --horizon 100 --ent_bonus 0.001 # Note: this will save the trained model in models/bandits-rl2-trainH100-0

# Test RL^2 for Bandits H=100 for 1000 episodes using the provided trained model
julia --project=. main.jl bandits rl2 1000 --suffix -testH100 --nactions 5 --horizon 100 --test_model models/bandits/H100/rl2.bson --act_greedy

# Train RL^3 for Bandits H=100 for 5000 episodes
julia --project=. main.jl bandits rl3 5000 --suffix train100 --seed 0 --nactions 5 --horizon 100 --ent_bonus 0.001
# Test RL^3 for Bandits H=100 for 1000 episodes
julia --project=. main.jl bandits rl3 1000 --suffix testH100 --nactions 5 --horizon 100 --test_model models/bandits/H100/rl3.bson --act_greedy

```

Similarly, you can run for horizon = 500. We used entropy bonus = 0.01 for horizon = 500.

Testing H500 model for OOD Bandits:

```bash
# Test RL^2 for Bandits H=500 for 1000 episodes using the provided trained model
julia --project=. main.jl bandits rl2 1000 --suffix testH500_OOD --nactions 5 --horizon 500 --test_model models/bandits/H500/rl2.bson --act_greedy --ood

# Test RL^3 for Bandits H=500 for 1000 episodes using the provided trained model
julia --project=. main.jl bandits rl3 1000 --suffix testH500_OOD --nactions 5 --horizon 500 --test_model models/bandits/H500/rl3.bson --act_greedy --ood

```

### RL^2 and RL^3 for MDPs environments

```bash
# Train RL^2 for MDPs H=100 for 10000 episodes
julia --project=. main.jl mdps rl2 10000 --suffix trainH100 --seed 0 --nstates=10 --nactions 5 --horizon 100 --ent_bonus 0.01 # Note: this will save the trained model in models/mdps-rl2-trainH100-0

# Test RL^2 for MDPs H=100 for 1000 episodes using the provided trained model
julia --project=. main.jl mdps rl2 1000 --suffix testH100 --nstates=10 --nactions 5 --horizon 100 --test_model models/mdps/H100/rl2.bson --act_greedy

# Train RL^3 for MDPs H=100 for 10000 episodes
julia --project=. main.jl mdps rl3 10000 --suffix train100 --seed 0 --nstates=10 --nactions 5 --horizon 100 --ent_bonus 0.01
# Test RL^3 for MDPs H=100 for 1000 episodes
julia --project=. main.jl mdps rl3 1000 --suffix testH100 --nstates=10 --nactions 5 --horizon 100 --test_model models/mdps/H100/rl3.bson --act_greedy

```

Similarly, you can run for horizon = 500. We used entropy bonus = 0.01 for horizon = 500.

Testing H500 model for OOD MDPs:

```bash
# Test RL^2 for MDPs H=500 for 1000 episodes using the provided trained model
julia --project=. main.jl mdps rl2 1000 --suffix testH500_OOD --nstates=10 --nactions 5 --horizon 500 --test_model models/mdps/H500/rl2.bson --act_greedy --ood

# Test RL^3 for MDPs H=500 for 1000 episodes using the provided trained model
julia --project=. main.jl mdps rl3 1000 --suffix testH500_OOD --nstates=10 --nactions 5 --horizon 500 --test_model models/mdps/H500/rl3.bson --act_greedy --ood

```

Testing H500 model extrapolated to H1000:

```bash
# Test RL^2 for MDPs H=1000 for 1000 episodes using the provided H500 trained model
julia --project=. main.jl mdps rl2 1000 --suffix testH1000_extrapolate --nstates=10 --nactions 5 --horizon 1000 --test_model models/mdps/H500/rl2.bson --act_greedy --extrapolate_mode

# Test RL^3 for MDPs H=1000 for 1000 episodes using the provided H500 trained model
julia --project=. main.jl mdps rl3 1000 --suffix testH1000_extrapolate --nstates=10 --nactions 5 --horizon 1000 --test_model models/mdps/H500/rl3.bson --act_greedy --extrapolate_mode

```

### RL^2 and RL^3 for Gridworld environments

```bash
# Train RL^2 for Gridworld 11x11 H=250 for 10000 episodes
julia --project=. main.jl gridworlds rl2 10000 --suffix train11x11 --grid_variation 11x11 --seed 0 --horizon 250 --ent_bonus 0.04 --lr 0.0002  # Note: this will save the trained model in models/gridworlds-rl2-train11x11-0

# Test RL^2 for Gridworld 11x11 H=250 for 1000 episodes using the provided trained model
julia --project=. main.jl gridworlds rl2 1000 --suffix test11x11 --grid_variation 11x11 --horizon 250 --test_model models/gridworlds/11x11/rl2.bson --act_greedy

# Train RL^3 for Gridworld 11x11 H=250 for 10000 episodes
julia --project=. main.jl gridworlds rl3 10000 --suffix train11x11 --grid_variation 11x11 --seed 0 --horizon 250 --ent_bonus 0.04 --lr 0.0002 # Note: this will save the trained model in models/gridworlds-rl3-train11x11-0

# Test RL^3 for Gridworld 11x11 H=250 for 1000 episodes using the provided trained model
julia --project=. main.jl gridworlds rl3 1000 --suffix test11x11 --grid_variation 11x11 --horizon 250 --test_model models/gridworlds/11x11/rl3.bson --act_greedy

# Train RL^3-Coarse for Gridworld 11x11 H=250 for 10000 episodes
julia --project=. main.jl gridworlds rl3_coarse 10000 --suffix train11x11 --grid_variation 11x11 --seed 0 --horizon 250 --ent_bonus 0.04 --lr 0.0002 # Note: this will save the trained model in models/gridworlds-rl3_coarse-train11x11-0

# Test RL^3-Coarse for Gridworld 11x11 H=250 for 1000 episodes using the provided trained model
julia --project=. main.jl gridworlds rl3_coarse 1000 --suffix test11x11 --grid_variation 11x11 --horizon 250 --test_model models/gridworlds/11x11/rl3_coarse.bson --act_greedy

```

Similarly, you can run for Gridworld 13x13 with H = 350.

To test OOD generalization, you can choose one of the following values for `grid_variation`: 13x13_dense, 13x13_deterministic, 13x13_watery, 13x13_dangerous, 13x13_corner.

Examples:
    
```bash
# Test RL^2 for Gridworld 13x13_deterministic H=350 for 1000 episodes using the provided trained model
julia --project=. main.jl gridworlds rl2 1000 --suffix test13x13_deterministic_OOD --grid_variation 13x13_deterministic --horizon 350 --test_model models/gridworlds/13x13/rl2.bson --act_greedy

# Test RL^3 for Gridworld 13x13_deterministic H=350 for 1000 episodes using the provided trained model
julia --project=. main.jl gridworlds rl3 1000 --suffix test13x13_deterministic_OOD --grid_variation 13x13_deterministic --horizon 350 --test_model models/gridworlds/13x13/rl3.bson --act_greedy

# Test RL^3-Coarse for Gridworld 13x13_deterministic H=350 for 1000 episodes using the provided trained model
julia --project=. main.jl gridworlds rl3_coarse 1000 --suffix test13x13_deterministic_OOD --grid_variation 13x13_deterministic --horizon 350 --test_model models/gridworlds/13x13/rl3_coarse.bson --act_greedy
```
