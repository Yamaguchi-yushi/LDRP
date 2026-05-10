"""LaRe-Task encoder: 10-factor evaluation function for task assignment quality.

Each factor is in roughly [0, 1] and aims to reflect *how good a single
assignment decision is at the moment it is taken*. A higher value should
correspond to a better decision.

Designed for LDRP's task model where each task is `[pickup_node, dropoff_node, deadline]`
and each agent holds at most one task at a time.

Inputs:
    `assignment_state` (dict) describes one decision:
        agent_id          : int             - agent receiving the task
        task              : [pickup, dropoff, deadline]
        agent_pos_node    : int             - node id of the agent (or nearest node)
        agent_prev_goal   : int | None      - the goal node the agent had before this assignment
        agent_was_idle    : bool            - True if the agent had no task before
        wait_steps        : int             - steps the task waited in the queue before assignment
        unassigned_after  : int             - count of still-unassigned tasks after this decision
        agent_loads_after : list[int]       - per-agent #pending tasks AFTER this assignment
        agent_dists       : list[float]     - per-agent shortest path to the pickup (only free agents
                                              get a finite value; busy agents get inf or > diameter)
        nearest_dist      : float           - the smallest dist among free agents
        n_assignments_step: int             - total assignments produced in the same env step
    `env_info` (dict): {
        n_agents          : int
        task_num          : int             - max queue capacity
        time_limit        : int
        graph_diameter    : float
        delivery_distance : float           - shortest_path(pickup, dropoff)
        pickup_distance   : float           - shortest_path(agent_pos_node, pickup) for THIS agent
        redirect_distance : float           - shortest_path(agent_prev_goal, pickup) (or pickup_distance if was idle)
    }

Returns: list of 10 numpy scalars, each shape (1,).
"""

import numpy as np


FACTOR_NUMBER = 10


def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def evaluation_func_task(assignment_state, env_info):
    diameter = float(env_info.get("graph_diameter", 1.0))
    n_agents = int(env_info.get("n_agents", 1))
    task_num = int(env_info.get("task_num", max(1, n_agents * 2)))
    time_limit = int(env_info.get("time_limit", 1000))

    pickup_distance = float(env_info.get("pickup_distance", diameter))
    delivery_distance = float(env_info.get("delivery_distance", diameter))
    redirect_distance = float(env_info.get("redirect_distance", pickup_distance))

    wait_steps = float(assignment_state.get("wait_steps", 0))
    agent_was_idle = bool(assignment_state.get("agent_was_idle", True))
    unassigned_after = int(assignment_state.get("unassigned_after", 0))
    agent_loads_after = list(assignment_state.get("agent_loads_after", [0] * n_agents))
    agent_dists = list(assignment_state.get("agent_dists", [diameter] * n_agents))
    nearest_dist = float(assignment_state.get("nearest_dist", pickup_distance))
    n_assignments_step = int(assignment_state.get("n_assignments_step", 1))

    # 1) pickup_proximity: closer pickup is better
    f_pickup_proximity = _clip01(1.0 - _safe_div(pickup_distance, diameter))

    # 2) delivery_efficiency: shorter pickup→dropoff is better (faster completion)
    f_delivery_efficiency = _clip01(1.0 - _safe_div(delivery_distance, diameter))

    # 3) wait_time_norm: longer-waiting tasks should be served first
    f_wait_time = _clip01(_safe_div(wait_steps, time_limit))

    # 4) load_balance_after: smaller std of loads is better
    if len(agent_loads_after) >= 1:
        std_load = float(np.std(agent_loads_after))
        max_imbalance = float(max(np.max(agent_loads_after), 1))
        f_load_balance = _clip01(1.0 - _safe_div(std_load, max_imbalance))
    else:
        f_load_balance = 1.0

    # 5) idle_assignment: assigning to a previously-idle agent is preferred
    f_idle_assignment = 1.0 if agent_was_idle else 0.0

    # 6) closest_agent_match: did we pick the agent closest to the pickup?
    finite_dists = [d for d in agent_dists if np.isfinite(d) and d < diameter * 10]
    if len(finite_dists) > 0 and pickup_distance <= max(finite_dists) + 1e-6:
        rank_score = 1.0 - _safe_div(pickup_distance - min(finite_dists), max(finite_dists) - min(finite_dists) + 1e-6)
        f_closest_match = _clip01(rank_score)
    else:
        f_closest_match = _clip01(1.0 - _safe_div(pickup_distance, diameter))

    # 7) queue_drain: how much we drained the unassigned queue (more drained = better)
    f_queue_drain = _clip01(1.0 - _safe_div(unassigned_after, task_num))

    # 8) low_redirect_cost: minimal rerouting from agent's previous goal
    f_low_redirect = _clip01(1.0 - _safe_div(redirect_distance, diameter))

    # 9) urgency_response: higher when handling old tasks (non-linear)
    urgency = _safe_div(wait_steps, time_limit * 0.5)
    f_urgency_response = _clip01(urgency)

    # 10) batch_assignment_density: more assignments in one step = better orchestration
    f_batch_density = _clip01(_safe_div(n_assignments_step, n_agents))

    return [
        np.array([f_pickup_proximity], dtype=np.float32),
        np.array([f_delivery_efficiency], dtype=np.float32),
        np.array([f_wait_time], dtype=np.float32),
        np.array([f_load_balance], dtype=np.float32),
        np.array([f_idle_assignment], dtype=np.float32),
        np.array([f_closest_match], dtype=np.float32),
        np.array([f_queue_drain], dtype=np.float32),
        np.array([f_low_redirect], dtype=np.float32),
        np.array([f_urgency_response], dtype=np.float32),
        np.array([f_batch_density], dtype=np.float32),
    ]


def factors_as_array(assignment_state, env_info):
    """Concatenate evaluation_func_task output into a 1D array of shape (FACTOR_NUMBER,)."""
    parts = evaluation_func_task(assignment_state, env_info)
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers for building env_info / assignment_state from a DrpEnv instance.
# These are kept independent of the env class so they can be unit-tested in
# isolation. They use only public attributes (G, pos, agent_num, etc.).
# ---------------------------------------------------------------------------

def _agent_node(env, agent_id):
    """Return the node id where agent currently sits (current_start), falling back to obs[3]."""
    try:
        return int(env.current_start[agent_id])
    except Exception:
        try:
            return int(env.obs[agent_id][2])
        except Exception:
            return 0


def _shortest_path_safe(env, src, dst, fallback):
    """Robust shortest-path lookup (returns fallback on error)."""
    try:
        return float(env.get_path_length(int(src), int(dst)))
    except Exception:
        return float(fallback)


def build_env_info(env, graph_diameter, agent_id, pickup_node, dropoff_node, agent_prev_goal):
    """Build the per-decision env_info dict consumed by evaluation_func_task."""
    pickup_distance = _shortest_path_safe(env, _agent_node(env, agent_id), pickup_node, graph_diameter)
    delivery_distance = _shortest_path_safe(env, pickup_node, dropoff_node, graph_diameter)
    if agent_prev_goal is None:
        redirect_distance = pickup_distance
    else:
        redirect_distance = _shortest_path_safe(env, agent_prev_goal, pickup_node, graph_diameter)
    return {
        "n_agents": int(env.agent_num),
        "task_num": int(env.task_num),
        "time_limit": int(env.time_limit),
        "graph_diameter": float(graph_diameter),
        "pickup_distance": float(pickup_distance),
        "delivery_distance": float(delivery_distance),
        "redirect_distance": float(redirect_distance),
    }


def build_assignment_state(env, graph_diameter, agent_id, pickup_node,
                           agent_loads_after, wait_steps, agent_was_idle,
                           unassigned_after, n_assignments_step):
    """Build the assignment_state dict consumed by evaluation_func_task.

    Computes per-agent distances to the pickup so factor-6 (closest-agent match)
    can compare this decision against the alternatives that were available.
    """
    agent_dists = []
    for j in range(env.agent_num):
        try:
            had_task = len(env.assigned_tasks[j]) > 0
        except Exception:
            had_task = False
        if had_task and j != agent_id:
            agent_dists.append(graph_diameter * 10)  # busy agents cannot be picked
        else:
            agent_dists.append(_shortest_path_safe(env, _agent_node(env, j), pickup_node, graph_diameter))
    nearest_dist = float(min(agent_dists)) if len(agent_dists) > 0 else graph_diameter
    return {
        "agent_id": int(agent_id),
        "agent_pos_node": _agent_node(env, agent_id),
        "agent_was_idle": bool(agent_was_idle),
        "wait_steps": int(wait_steps),
        "unassigned_after": int(unassigned_after),
        "agent_loads_after": list(agent_loads_after),
        "agent_dists": agent_dists,
        "nearest_dist": nearest_dist,
        "n_assignments_step": int(n_assignments_step),
    }
