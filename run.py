import subprocess
import time
import os
import sys

map_name = [
    #"map_5x4",
    #"map_8x5",
    "map_aoba00",
    #"map_aoba01",
]

agent_num = [
    # 3,
    # 4,
    # 5,
    # 6,
    # 7,
    # 9,
    # 8,
     10,
    # 11,
    # 12,
    # 13,
    # 14,
    # 15,
]

path_planner = [
    #"iql",
    #"qmix",
    #"vdn",
    "mappo",
    #"qplex",
    #"happo",
    #"mat",
    #"mat_dec",
    #"pbs",
]

task_assigner = [
    "fifo",
    "tp",
]

method_tag = [
    #"",
    "safe",
    #"ours",
]

reassign_before_pickup = [
    "base",
    #"reassign",
]

mat_model_agent_num = [     # mat_decのモデルを学習した際のエージェント数を指定する．
    "",
    #"4",
    #7,
    #"8",
    # 10,
]

#"""
command = [
    [sys.executable, "-u", "test.py", str(i), str(j), str(k), str(l), str(m), str(n), str(o)]
    for i in map_name
    for j in agent_num
    for k in path_planner
    for l in task_assigner
    for m in method_tag
    for n in reassign_before_pickup
    for o in mat_model_agent_num
]
"""
command = [
    ["python3", "test.py", "map_aoba00", "4", "pbs", "tp"]
]
"""
"""
for cmd in command:
    with open("logs/" + str(cmd[2]) + "_" + str(cmd[3]) + "_" + str(cmd[4]) + "_" + str(cmd[5]) + ".txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

"""
maxpurocesses = 5
running_processes = []

#logファイルのパス変更ver
for cmd in command:
    log_dir = "logs/" + str(cmd[3]) + "/safe" + "/" + str(cmd[5]) + "/" + str(cmd[4]) + "agent"
    os.makedirs(log_dir, exist_ok=True)
    method_suffix = f"_{cmd[7]}" if len(cmd) > 7 and cmd[7] else ""
    reassign_suffix = f"_{cmd[8]}" if len(cmd) > 8 and cmd[8] else ""
    train_n = cmd[9] if len(cmd) > 9 and cmd[9] else cmd[4]
    log_name = f"{cmd[3]}_{cmd[4]}_{cmd[5]}_{cmd[6]}{method_suffix}{reassign_suffix}_{train_n}.txt"
    with open(os.path.join(log_dir, log_name), "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    running_processes.append((proc ,cmd))
    print("Started:", cmd, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    while len(running_processes) >= maxpurocesses:
        for p,c in running_processes[:]:
            if p.poll() is not None:
                print("Finished:", c)
                running_processes.remove((p,c))
        time.sleep(0.1)

#ver2
"""
for cmd in command:
    with open("logs/" + str(cmd[2]) + "_" + str(cmd[3]) + "_" + str(cmd[4]) + "_" + str(cmd[5]) + ".txt", "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    running_processes.append((proc ,cmd))

    while len(running_processes) >= maxpurocesses:
        for p,c in running_processes[:]:
            if p.poll() is not None:
                print("Finished:", c)
                running_processes.remove((p,c))
        time.sleep(0.1)
"""

for p, c in running_processes:
    p.wait()
    print("Finished:", c)
#"""