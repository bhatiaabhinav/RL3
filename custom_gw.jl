using MDPs
using GridWorlds
using Plots
using ValueIteration
using Random

enter_rewards = Dict('S' => -1.0, ' ' => -1.0, 'G' => 100.0, 'O' => 0.0, 'W'=>-2.0, '!' => -10.0, 'X' => -100.0);
absorbing_states = Set(['G', 'X'])
GridWorlds.drawcolors['W'] = "light blue"
GridWorlds.drawcolors['X'] = "red"
GridWorlds.drawcolors['!'] = "yellow"
vmax = enter_rewards['G']

function allowed_goal_positions(grid_side::Int = 11, min_manhattan_distance::Int = 8)
    Iterators.filter(pos::Tuple{Int, Int} -> sum(abs.(pos .- ceil(Int, grid_side / 2))) >= min_manhattan_distance, Iterators.product(1:grid_side, 1:grid_side))
end

function create_grid(rng::Random.AbstractRNG, grid_side=11, num_obstacle_sets=11, obstacle_set_len=3, num_water_sets=5, water_set_length=2, num_dangers=2, min_goal_manhat=8)
    grid::Matrix{Char} = fill(' ', grid_side, grid_side)
    grid_center::Int = ceil(Int, grid_side / 2)
    for i in 1:num_obstacle_sets
        isvertical::Bool = rand(rng, Bool)
        if isvertical
            r1::Int = rand(rng, 1:grid_side) # row 1
            direction::Int = r1 == grid_center ? rand(rng, (-1, 1)) : (r1 > grid_center ? - 1 : 1)
            r2::Int = r1 + direction * (obstacle_set_len - 1) # row 2
            r1, r2 = min(r1, r2), max(r1, r2) # swap if r1 > r2
            grid[r1:r2, rand(rng, 1:(grid_side))] .= 'O' # set the obstacle from row 1 to row 2 on a random column
        else
            c1::Int = rand(rng, 1:grid_side) # column 1
            direction = c1 == grid_center ? rand(rng, (-1, 1)) : (c1 > grid_center ? - 1 : 1)
            c2::Int = c1 + direction * (obstacle_set_len - 1) # column 2
            c1, c2 = min(c1, c2), max(c1, c2) # swap if c1 > c2
            grid[rand(rng, 1:grid_side), c1:c2] .= 'O' # set the obstacle from column 1 to column 2 on a random row
        end
    end
    for i in 1:num_water_sets
        isvertical::Bool = rand(rng, Bool)
        if isvertical
            r1::Int = rand(rng, 1:grid_side) # row 1
            direction = r1 == grid_center ? rand(rng, (-1, 1)) : (r1 > grid_center ? - 1 : 1)
            r2::Int = r1 + direction * (water_set_length - 1) # row 2
            r1, r2 = min(r1, r2), max(r1, r2) # swap if r1 > r2
            grid[r1:r2, rand(rng, 1:(grid_side))] .= 'W' # set the water from row 1 to row 2 on a random column
        else
            c1::Int = rand(rng, 1:grid_side) # column 1
            direction = c1 == grid_center ? rand(rng, (-1, 1)) : (c1 > grid_center ? - 1 : 1)
            c2::Int = c1 + direction * (water_set_length - 1) # column 2
            c1, c2 = min(c1, c2), max(c1, c2) # swap if c1 > c2
            grid[rand(rng, 1:grid_side), c1:c2] .= 'W' # set the water from column 1 to column 2 on a random row
        end
    end
    grid[grid_center, grid_center] = 'S' # set the start position
    goal_position = rand(rng, collect(allowed_goal_positions(grid_side, min_goal_manhat)))
    grid[goal_position...] = 'G' # set the goal position to a random position that is at least min_goal_manhat away from the center
    _d::Int = 0 # number of dangers placed so far
    while _d < num_dangers
        r::Int, c::Int = rand(rng, 2:(grid_side-1)), rand(rng, 2:(grid_side-1)) # random position in the grid that is not on the border
        if grid[r, c] == ' ' # if the position is empty
            up::Tuple{Int, Int} = (r-1, c) # up position 
            right::Tuple{Int, Int} = (r, c+1) # right position
            down::Tuple{Int, Int} = (r+1, c) # down position
            left::Tuple{Int, Int} = (r, c-1) # left position
            dirs::NTuple{4, Tuple{Int, Int}} = (up, down, left, right) # all neighbor positions
            if all(map(dir -> grid[dir...] ∈ (' ', 'O'), dirs)) # if all directions are empty or obstacles
                grid[r, c] = 'X' # set the danger
                for dir in dirs # set the danger's neighbors to be warnings
                    grid[dir...] = '!'
                end
                _d += 1
            end 
        end
    end
    return grid
end


CUSTOMGW_11x11_PARAMS = (11, 11, 3, 5, 2, 2, 8, 0.2) # grid_side, num_obstacle_sets, obstacle_set_len, num_water_sets, water_set_length, num_dangers, min_goal_manhat, default_slip_probabality
CUSTOMGW_11x11_PARAMS_DETERMINISTIC = (11, 11, 3, 5, 2, 2, 8, 0.0)

CUSTOMGW_13x13_PARAMS = (13, 11, 3, 5, 2, 2, 8, 0.2)
CUSTOMGW_13x13_PARAMS_DETERMINISTIC = (13, 11, 3, 5, 2, 2, 8, 0.0)
CUSTOMGW_13x13_PARAMS_WATERY = (13, 11, 3, 8, 2, 2, 8, 0.2)
CUSTOMGW_13x13_PARAMS_DANGEROUS = (13, 11, 3, 5, 2, 4, 8, 0.2)
CUSTOMGW_13x13_PARAMS_DENSE = (13, 11, 4, 5, 2, 2, 8, 0.2)
CUSTOMGW_13x13_PARAMS_CORNER = (13, 11, 3, 5, 2, 2, 12, 0.2)

CUSTOMGW_15x15_PARAMS = (15, 15, 4, 6, 3, 3, 8, 0.2)


CUSTOMGW_DEFAULT_PARAMS = CUSTOMGW_13x13_PARAMS


function CustomGridWorld(rng::Random.AbstractRNG, args::NTuple{8, Real}=CUSTOMGW_DEFAULT_PARAMS)
    return_in_100 = -1
    gw = nothing
    default_slip_prob = args[end]
    slip_probabality = Dict('S' => default_slip_prob, ' ' => default_slip_prob, 'G' => 0.0,  'O' => 0.0, 'W' => 1.0, '!' => default_slip_prob, 'X' => 0.0)
    while !(50 <= return_in_100 <= 100)
        gw = GridWorld(create_grid(rng, args[1:7]...), enter_rewards, failuremode_slip_probability=slip_probabality, absorbing_states=absorbing_states)
        return_in_100 = value_iteration(gw, 1, 100)[1]
    end
    return gw
end

CustomGridWorld(args::NTuple{8, Real}=CUSTOMGW_DEFAULT_PARAMS) = CustomGridWorld(Random.GLOBAL_RNG, args)

import ProgressMeter

function get_avg_obstacle_density(args::NTuple{8, Real}, samples::Int=1000)
    sum(ProgressMeter.@showprogress map(1:samples) do i
        grid = CustomGridWorld(args).grid
        return sum(grid .== 'O') / length(grid)
    end) / samples
end