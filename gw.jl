using MDPs
using GridWorlds
using ValueIteration
using Random
using MDPVideoRecorder

enter_rewards = Dict('S' => -1.0, ' ' => -1.0, 'G' => 100.0, 'O' => 0.0, 'W'=>-2.0, '!' => -10.0, 'X' => -100.0);
absorbing_states = Set(['G', 'X'])
slip_probabality_stochastic = Dict('S' => 0.2, ' ' => 0.2, 'G' => 0.0,  'O' => 0.0, 'W' => 1.0, '!' => 0.2, 'X' => 0.0) 
# for deterministic, slip_probabality:
slip_probabality_determinisitc = Dict('S' => 0.0, ' ' => 0.0, 'G' => 0.0,  'O' => 0.0, 'W' => 1.0, '!' => 0.0, 'X' => 0.0)
slip_probabality = slip_probabality_stochastic
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

#  grid_variation: either of 11x11, 13x13, 13x13_dense, 13x13_deterministic, 13x13_watery, 13x13_dangerous, 13x13_corner
grid_variation_dict = Dict(
    "11x11" => (11, 11, 3, 5, 2, 2, 8),
    "11x11_deterministic" => (11, 11, 3, 5, 2, 2, 8),
    "13x13" => (13, 11, 3, 5, 2, 2, 8),
    "13x13_dense" => (13, 11, 4, 5, 2, 2, 8),
    "13x13_deterministic" => (13, 11, 3, 5, 2, 2, 8),
    "13x13_watery" => (13, 11, 3, 8, 2, 2, 8),
    "13x13_dangerous" => (13, 11, 3, 5, 2, 4, 8),
    "13x13_corner" => (13, 11, 3, 5, 2, 2, 12)
)


function make_gw(rng::Random.AbstractRNG, args::NTuple{7, Int})
    return_in_100 = -1
    gw = nothing
    while !(50 <= return_in_100 <= 100)
        gw = GridWorld(create_grid(rng, args...), enter_rewards, failuremode_slip_probability=slip_probabality, absorbing_states=absorbing_states)
        return_in_100 = value_iteration(gw, 1, 100)[1]
    end
    # println("return_in_100: $return_in_100")
    return gw
end
make_grid(rng::Random.AbstractRNG, args::NTuple{7, Int}) = make_gw(rng, args).grid
make_grid(args::NTuple{7, Int}) = make_grid(Random.GLOBAL_RNG, args)

function make_grid(rng::Random.AbstractRNG, grid_variation::String)
    make_grid(rng, grid_variation_dict[grid_variation])
end
make_grid(grid_variation::String) = make_grid(Random.GLOBAL_RNG, grid_variation)