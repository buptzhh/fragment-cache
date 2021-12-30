from gym.envs.registration import register
register(
    id="CacheSimulator_bi-v0",
    entry_point="a3c.cache.envs.CacheSimulatorEnv_bi:CacheSimulator",
    max_episode_steps=2000,
    reward_threshold=100.0,
)
register(
    id="CacheFragmentSimulator-v0",
    entry_point="a3c.cache.envs.CacheFragmentSimulatorEnv_bi:CacheFragmentSimulator",
    max_episode_steps=2000,
    reward_threshold=100.0,
)
register(
    id="CacheSimulator_s1-v1",
    entry_point="a3c.cache.envs.CacheSimulatorEnv_s1:CacheSimulator",
    max_episode_steps=2000,
    reward_threshold=100.0,
)
register(
    id="CacheFragmentSimulator-v1",
    entry_point="a3c.cache.envs.CacheFragmentSimulatorEnv_s1:CacheFragmentSimulator",
    max_episode_steps=2000,
    reward_threshold=100.0,
)