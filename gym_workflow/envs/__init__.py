from gym.envs.registration import register

register(
	id='Montage-v1',
	entry_point="gym_workflow.envs.scheme.version_1:Version1",
)
register(
	id='Montage-v2',
	entry_point="gym_workflow.envs.scheme.version_2:Version2",
)
register(
	id='Montage-v3',
	entry_point="gym_workflow.envs.scheme.version_3:Version3",
)
register(
	id='Montage-v4',
	entry_point="gym_workflow.envs.scheme.version_4:Version4",
)
register(
	id='Montage-v5',
	entry_point="gym_workflow.envs.scheme.version_5:Version5",
)
