from gym.envs.registration import register

register(
    id='montage-v0',
    entry_point='gym_workflow.envs:MontageEnv',
)
