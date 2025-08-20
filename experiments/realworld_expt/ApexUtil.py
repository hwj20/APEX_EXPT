import math

def average_pos(buffer):
    x = sum(p[0] for p in buffer if p) / len(buffer)
    y = sum(p[1] for p in buffer if p) / len(buffer)
    return (x, y)

import numpy as np

def build_snapshot(pos_buffer_red, pos_buffer_green,
                   vel_red, vel_green,
                   window_size=5):
    """
    Build a snapshot dict containing averaged positions for 'red' and 'green',
    and directly passed velocities for each.

    Args:
        pos_buffer_red (deque or list of array-like): recent positions of red object.
        pos_buffer_green (deque or list of array-like): recent positions of green object.
        vel_red (array-like): velocity of red object (will use first 2 components).
        vel_green (array-like): velocity of green object (will use first 2 components).
        window_size (int): number of last frames to average over (default=5).

    Returns:
        dict with key 'objects', a list of dicts each having:
            'name': object name ('red' or 'green')
            'position': averaged [x, y] position
            'velocity': passed [vx, vy]
    """
    def average(buf):
        arr = np.asarray(list(buf)[-window_size:])
        return np.mean(arr, axis=0)

    # Compute averaged positions
    avg_pos_red   = average(pos_buffer_red)[:2].tolist()
    avg_pos_green = average(pos_buffer_green)[:2].tolist()

    # Prepare velocities (use first two components)
    vel_red2   = np.asarray(vel_red)[:2].tolist()
    vel_green2 = np.asarray(vel_green)[:2].tolist()

    # Build snapshot structure
    return {
        'objects': [
            {'name': 'red',   'position': avg_pos_red,   'velocity': vel_red2},
            {'name': 'green', 'position': avg_pos_green, 'velocity': vel_green2}
        ]
    }



def predict_collision(pos_buffer_red, pos_buffer_blue, N, dt=0.1, horizon_sec=5.0, threshold=170/18):
    if len(pos_buffer_red) < 2 or len(pos_buffer_blue) < 2:
        return False
    p0_red = average_pos(pos_buffer_red[:N])
    p1_red = average_pos(pos_buffer_red[N:])
    p0_blue = average_pos(pos_buffer_blue[:N])
    p1_blue = average_pos(pos_buffer_blue[N:])

    vx_red = (p1_red[0] - p0_red[0]) / dt
    vy_red = (p1_red[1] - p0_red[1]) / dt
    vx_blue = (p1_blue[0] - p0_blue[0]) / dt
    vy_blue = (p1_blue[1] - p0_blue[1]) / dt

    T = horizon_sec
    pred_red = (p1_red[0] + vx_red * T, p1_red[1] + vy_red * T)
    pred_blue = (p1_blue[0] + vx_blue * T, p1_blue[1] + vy_blue * T)

    dx = pred_red[0] - pred_blue[0]
    dy = pred_red[1] - pred_blue[1]
    distance = math.hypot(dx, dy)
    
    print(distance)

    return distance < threshold



def euler_rollout(snapshot, target_point, dt=0.1, horizon=5.0, collision_threshold=170/18, moving_object='green', target_object='red'):
    """
    Simulates a moving object (e.g., 'blue') heading toward a target object (e.g., 'red').
    If it enters a square region of size +-10 around target_point, it stops.
    Returns whether a collision occurs and the final distance.
    """
    objects = {obj['name']: {
        'pos': obj['position'][:2],
        'vel': obj.get('velocity', [0.0, 0.0])[:2]
    } for obj in snapshot['objects']}

    if moving_object not in objects or target_object not in objects:
        return {'valid': False, 'reason': 'Missing required objects'}

    mover = objects[moving_object]
    target = objects[target_object]

    mover_pos = list(mover['pos'])
    mover_vel = list(mover['vel'])
    target_pos = list(target['pos'])

    t = 0
    collided = False
    stopped = False
    while t < 1.0: # let it run 1 sec
        mover_pos[0] += mover_vel[0] * dt
        mover_pos[1] += mover_vel[1] * dt
        t += dt

    while t < horizon:
        if not stopped:
            # If entered the stop region, stop
            if (abs(mover_pos[0] - target_point[0]) <= 1 and abs(mover_pos[1] - target_point[1]) <= 1):
                mover_vel = [0.0, 0.0]
                stopped = True
            else:
                # Move the object
                mover_pos[0] += mover_vel[0] * dt
                mover_pos[1] += mover_vel[1] * dt

        # Target stays static
        dist = math.hypot(mover_pos[0] - target_pos[0], mover_pos[1] - target_pos[1])
        if dist < collision_threshold:
            collided = True
            break

        t += dt

    final_dist = math.hypot(mover_pos[0] - target_pos[0], mover_pos[1] - target_pos[1])
    return {
        'valid': True,
        'final_distance': final_dist,
        'collision': collided
    }


import json
import time

def get_points(filename="reachable_points.txt"):

    with open(filename, "r") as f:
        target_points = f.read().split('\n')
    return target_points

def select_safe_targets(snapshot, filename="reachable_points.txt", top_k=5):
    start_time = time.perf_counter()

    with open(filename, "r") as f:
        target_points = f.read().split('\n')
        
    results = []
    rolled_count = 0

    for point in target_points:
    
        if point == '':
            continue
        rolled_count += 1
        point = eval(point)
        result = euler_rollout(snapshot, target_point=tuple(point))
     #   print(result)
        if result["valid"] and not result["collision"]:
            results.append({
                "point": point,
                "final_distance": result["final_distance"],
                "collision":False
            })
    results = sorted(results, key=lambda x: -x["final_distance"])
    top_results = results[:top_k]
    

    print("Top safe points:")
    for i, item in enumerate(top_results):
        print(f"{i+1}. Point: {item['point']}, Final Distance: {item['final_distance']:.2f}")

    elapsed = time.perf_counter() - start_time
    print(f"Rolled {rolled_count} target points in {elapsed:.3f} seconds.")
    print("Top safe points:", top_results)
    return top_results
