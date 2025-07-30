class TaskManager():
    def __init__(self):
        pass

    def assign_task(self, env, current_tasklist, assigned_tasklist):
        task_assign = []
        current_tasklist = current_tasklist.copy()
        assigned_tasklist = assigned_tasklist.copy()

        for i in range(env.agent_num):
            task = -1
            if len(assigned_tasklist[i]) == 0 and len(current_tasklist)>0 :
                task = current_tasklist.pop(0)
            task_assign.append(task)

        return task_assign