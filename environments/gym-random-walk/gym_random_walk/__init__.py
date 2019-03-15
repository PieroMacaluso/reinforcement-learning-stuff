from gym.envs.registration import register

register(
    id='RandomWalkSeven-v0',
    entry_point='gym_random_walk.envs:RandomWalkEnv',
    kwargs={'n_states': 7},
    timestep_limit=100,
    reward_threshold=1.0,
    nondeterministic=True,
)
register(
    id='RandomWalkTwentyOne-v0',
    entry_point='gym_random_walk.envs:RandomWalkEnv',
    kwargs={'n_states': 21},
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic=True,
)
