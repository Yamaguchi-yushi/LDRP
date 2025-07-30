class FIFO():
    def __init__(self):
        pass

    def assign_task(self, env, current_tasklist, assigned_tasklist):
        task_assign = []
        current_tasklist = current_tasklist.copy()
        assigned_tasklist = assigned_tasklist.copy()

        for i in range(env.agent_num):
            task = 0
            if len(assigned_tasklist[i]) == 0 and len(current_tasklist)>0 :
                task_assign.append(task)
                current_tasklist.pop(0)
                task += 1
            else:
                task_assign.append(-1)


        return task_assign