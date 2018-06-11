from gym.envs.registration import register

register(
	id='Montage-v0',
	entry_point='gym_workflow.envs.workflow_env:WorkflowEnv',
)
