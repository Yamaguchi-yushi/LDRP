from src.task_assign.task_policy.fifo import FIFO
from src.task_assign.task_policy.ppo import PPOAgent
from src.task_assign.task_policy.tp import TP

class TaskAgent():
    def __init__(self, name, args=None):
        if name == "fifo":
            print("call fifo")
            self.task_assigner = FIFO()
        elif name == "tp":
            print("call TP")
            self.task_assigner = TP()
        elif name == "ppo":
            print("call ppo")
            self.task_assigner = PPOAgent(args)
        else:
            raise ValueError(f"Unknown task assignment method: {name}")

    def assign_task(self, env, task_info):
        current_tasklist = task_info["Tasks"]
        assigned_tasklist = task_info["Assigned"]
        return self.task_assigner.assign_task(env, current_tasklist, assigned_tasklist)
