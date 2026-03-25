from src.all_policy.policy_manager import PolicyManager
from src.task_assign.task_manager import TaskManager

class Policy():
    def __init__(self, args):
        self.path_planner = PolicyManager(args)
        self.task_manager = TaskManager(args.task_assigner, args)

    def policy(self, n_obs, env):
        agents_action = self.path_planner.policy(n_obs, env)
        task_assign = self.task_manager.assign_task(env)

        joint_action = {"pass": agents_action, "task": task_assign}
        return joint_action
