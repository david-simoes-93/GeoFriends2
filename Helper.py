import numpy as np
import scipy.signal
import tensorflow as tf
import math
import random


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        # print(from_var, to_var)
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


# just some empty arrays
def get_empty_loss_arrays(size):
    v_l = np.empty(size)
    p_l = np.empty(size)
    e_l = np.empty(size)
    g_n = np.empty(size)
    v_n = np.empty(size)
    return v_l, p_l, e_l, g_n, v_n


# format input
def get_htg_rect(_screen_rectangle, simulator):
    action_vals = [-1, -1, -1, 0]

    reward_above_me = _screen_rectangle[0][1] - \
                      (simulator.rect_max + simulator.rect_min - _screen_rectangle[0][7]) / 2 < 0
    reward_left = _screen_rectangle[0][0] < 0
    hole_left = _screen_rectangle[0][3] > 0
    hole_right = _screen_rectangle[0][5] > 0
    reward_closer_than_hole = (reward_left and _screen_rectangle[0][0] > _screen_rectangle[0][2]) or \
                              (not reward_left and _screen_rectangle[0][0] < _screen_rectangle[0][4])
    rectangle_larger_than_hole = (reward_left and _screen_rectangle[0][3] == 0) or \
                                 (not reward_left and _screen_rectangle[0][5] == 0)
    rectangle_growing_side = _screen_rectangle[0][6]

    if reward_above_me:
        if reward_closer_than_hole or rectangle_larger_than_hole:
            if reward_left:
                action_vals[0] = 1
            else:
                action_vals[1] = 1
            if reward_closer_than_hole and rectangle_growing_side:  # grow high
                action_vals[2] = 1
        else:  # if hole closer than reward and we fall through it, move away from reward
            if reward_left:
                action_vals[1] = 1
            else:
                action_vals[0] = 1
            if not rectangle_growing_side:
                action_vals[2] = 1
    else:  # reward below, hole before reward
        if hole_left:
            action_vals[0] = 1
        elif hole_right:
            action_vals[1] = 1
        else:
            action_vals[3] = 1

        if rectangle_growing_side:
            action_vals[2] = 1

    return action_vals


def get_trainer_action_rectangle(_screen_rectangle, simulator):
    possibles = [(x, y) for x, y in enumerate(get_htg_rect(_screen_rectangle, simulator))]
    return random.choice([a[0] for a in possibles if a[1] == 1])


def get_trainer_action_circle(_screen_circle, simulator):
    possibles = [(x, y) for x, y in enumerate(get_htg_circ(_screen_circle, simulator))]
    return random.choice([a[0] for a in possibles if a[1] == 1])


def get_htg_circ(_screen_circle, simulator):
    action_vals = [-1, -1, 0, -1]

    reward_above_me = _screen_circle[0][1] - simulator.circle_radius < 0
    reward_left = _screen_circle[0][0] < 0

    reward_closer_than_hole = (reward_left and _screen_circle[0][0] > _screen_circle[0][2]) or \
                              (not reward_left and _screen_circle[0][0] < _screen_circle[0][4])
    circle_jumping = bool(_screen_circle[0][6])
    if circle_jumping:
        action_vals[2] = -1

    if reward_above_me:
        if reward_closer_than_hole:
            if reward_left:
                action_vals[0] = 1
            else:
                action_vals[1] = 1
            if not circle_jumping and abs(_screen_circle[0][0]) < simulator.circle_radius:
                action_vals[2] = 1
        else:  # if hole closer than reward, jump over it
            if reward_left:
                action_vals[0] = 1
                if not circle_jumping and _screen_circle[0][2] > -simulator.circle_radius * 2:
                    action_vals[2] = 1
            else:
                action_vals[1] = 1
                if not circle_jumping and _screen_circle[0][4] < simulator.circle_radius * 2:
                    action_vals[2] = 1

    else:  # reward below
        # if reward_closer_than_hole:
        if reward_left:
            action_vals[0] = 1
            if not circle_jumping and _screen_circle[0][2] > -simulator.circle_radius * 2:
                action_vals[2] = 1
        else:
            action_vals[1] = 1
            if not circle_jumping and _screen_circle[0][4] < simulator.circle_radius * 2:
                action_vals[2] = 1

    return action_vals


def rectangle_output(data, obstacles_rect, ground, s_size_rect):
    # [  closest_reward_x,       closest_reward_y,
    #   left_hole_x,            left_hole_width,
    #   right_hole_x,           right_hole_width,
    #   growing_wide_boolean,   width,
    #   reward_counter                              ]
    _screen_rectangle = np.zeros([1, s_size_rect])

    # TODO: maybe for from high to low instead of closest
    my_x = data[0]
    my_y = data[1]

    # closest reward
    # rwrd1, rwrd2, rwrd3 = (1000, 1000), (1000, 1000), (1000, 1000)
    if data[4] == 0 and data[5] == 0:
        data[4] = 10000
        data[5] = 10000
    if data[6] == 0 and data[7] == 0:
        data[6] = 10000
        data[7] = 10000
    if data[8] == 0 and data[9] == 0:
        data[8] = 10000
        data[9] = 10000
    rwrd1 = (data[4] - my_x, data[5] - my_y)
    rwrd2 = (data[6] - my_x, data[7] - my_y)
    rwrd3 = (data[8] - my_x, data[9] - my_y)
    d1 = math.sqrt(rwrd1[0] ** 2 + rwrd1[1] ** 2)
    d2 = math.sqrt(rwrd2[0] ** 2 + rwrd2[1] ** 2)
    d3 = math.sqrt(rwrd3[0] ** 2 + rwrd3[1] ** 2)
    if d1 < d2 and d1 < d3:
        _screen_rectangle[0][0], _screen_rectangle[0][1] = rwrd1[0], rwrd1[1]
    elif d2 < d1 and d2 < d3:
        _screen_rectangle[0][0], _screen_rectangle[0][1] = rwrd2[0], rwrd2[1]
    elif d3 < d1 and d3 < d2:
        _screen_rectangle[0][0], _screen_rectangle[0][1] = rwrd3[0], rwrd3[1]

    # ground holes in both directions
    hole_left, hole_right = get_rect_points(obstacles_rect, ground)
    _screen_rectangle[0][2] = hole_left[0] - my_x
    _screen_rectangle[0][3] = 0 if hole_left[1] < data[3] + 20 else 1  # 1 if needs to grow wide
    _screen_rectangle[0][4] = hole_right[0] - my_x
    _screen_rectangle[0][5] = 0 if hole_right[1] < data[3] + 20 else 1  # 1 if needs to grow wide

    # wall holes in proper direction
    # _screen[0][4] = 0  # right_down[0] - my_x
    # _screen[0][5] = 0  # if True else 1  # 1 if needs to grow wide

    # shape
    _screen_rectangle[0][6] = data[2]  # growing_wide
    _screen_rectangle[0][7] = data[3]  # width

    # reward counter
    _screen_rectangle[0][8] = (len(data) - 4) / 2

    return _screen_rectangle


def get_rect_points(obstacles_rect, ground):
    # if left:
    # find closest x-wise obstacle with top<=ground_top and right<=ground_left
    closest_ground_left = None
    for obs in obstacles_rect:
        # obstacle below and to the left of current ground
        # Highest obstacle, if multiple, then the right-most one
        if ground.top_y <= obs.top_y and obs.right_x < ground.left_x:
            if closest_ground_left is None or \
                            obs.top_y < closest_ground_left.top_y or \
                    (closest_ground_left.right_x < obs.right_x and obs.top_y == closest_ground_left.top_y):
                closest_ground_left = obs

    if closest_ground_left is not None:
        # calculate distance between obs_right and ground_left
        hole_width = ground.left_x - closest_ground_left.right_x
    else:
        hole_width = 0

    # return [ground_left, obs_right-ground_left], aka [hole start position, hole width]
    hole_left = [ground.left_x, hole_width]
    # else:
    # find closest x-wise obstacle with top<=ground_top and ground_right < left
    closest_ground_right = None
    for obs in obstacles_rect:
        # obstacle below and to the right of current ground
        # Highest obstacle, if multiple, then the right-most one
        if ground.top_y <= obs.top_y and ground.right_x < obs.left_x:
            if closest_ground_right is None or \
                            obs.top_y < closest_ground_right.top_y or \
                    (obs.left_x < closest_ground_right.left_x and obs.top_y == closest_ground_right.top_y):
                closest_ground_right = obs

    if closest_ground_right is not None:
        # calculate distance between obs_right and ground_left
        hole_width = closest_ground_right.left_x - ground.right_x
    else:
        hole_width = 0

    # return [ground_left, obs_right-ground_left], aka [hole start position, hole width]
    hole_right = [ground.right_x, hole_width]

    return hole_left, hole_right


def circle_output(data, obstacles_circ, ground, s_size_circ):
    # [ closest_reward_x,       closest_reward_y,
    #   left_hole_x,            left_hole_width,
    #   right_hole_x,           right_hole_width,
    #   jumping boolean,        ceiling distance,
    #   speed_x,                speed_y,
    #   reward_counter                              ]
    _screen_circle = np.zeros([1, s_size_circ])

    my_x = data[0]
    my_y = data[1]

    # closest reward
    # rwrd1, rwrd2, rwrd3 = (10000, 10000), (10000, 10000), (10000, 10000)
    _screen_circle[0][10] = 3
    if data[4] == 0 and data[5] == 0:
        data[4] = 10000
        data[5] = 10000
        _screen_circle[0][10] -= 1
    if data[6] == 0 and data[7] == 0:
        data[6] = 10000
        data[7] = 10000
        _screen_circle[0][10] -= 1
    if data[8] == 0 and data[9] == 0:
        data[8] = 10000
        data[9] = 10000
        _screen_circle[0][10] -= 1
    rwrd1 = (data[4] - my_x, data[5] - my_y)
    rwrd2 = (data[6] - my_x, data[7] - my_y)
    rwrd3 = (data[8] - my_x, data[9] - my_y)
    d1 = math.sqrt(rwrd1[0] ** 2 + rwrd1[1] ** 2)
    d2 = math.sqrt(rwrd2[0] ** 2 + rwrd2[1] ** 2)
    d3 = math.sqrt(rwrd3[0] ** 2 + rwrd3[1] ** 2)
    if d1 < d2 and d1 < d3:
        _screen_circle[0][0], _screen_circle[0][1] = rwrd1[0], rwrd1[1]
    elif d2 < d1 and d2 < d3:
        _screen_circle[0][0], _screen_circle[0][1] = rwrd2[0], rwrd2[1]
    elif d3 < d1 and d3 < d2:
        _screen_circle[0][0], _screen_circle[0][1] = rwrd3[0], rwrd3[1]

    # platforms in both directions
    ground, ceil, left_down, left_up, right_down, right_up = get_circ_points(my_x, my_y, ground, obstacles_circ)

    # if target below, show holes, else show platforms!
    if _screen_circle[0][1] < 0:
        _screen_circle[0][2] = left_up[0] - my_x
        _screen_circle[0][3] = left_up[1] - my_y
        _screen_circle[0][4] = right_up[0] - my_x
        _screen_circle[0][5] = right_up[1] - my_y
    else:
        _screen_circle[0][2] = left_down[0] - my_x
        _screen_circle[0][3] = left_down[1] - my_y
        _screen_circle[0][4] = right_down[0] - my_x
        _screen_circle[0][5] = right_down[1] - my_y

    # jump
    _screen_circle[0][6] = 1 if abs(data[3]) > 0.1 else 0  # jumping
    _screen_circle[0][7] = ceil.bot_y - my_y  # ceiling distance

    # SPEEDS
    _screen_circle[0][8] = data[2]  # vel x
    _screen_circle[0][9] = data[3]  # vel y

    # reward counter
    #_screen_circle[0][10] = (len(data) - 4) / 2

    return _screen_circle


def get_circ_points(my_x, my_y, ground, obstacles_circ):
    # todo: get index of ground from server, cheatsy doodles
    # ground = self.simulator.circle_ground
    ceil = get_obstacle_above_point(my_x, my_y, obstacles_circ)

    # move left until fall or wall
    index = -1
    for i, obs in enumerate(obstacles_circ):
        if ground.left_x <= obs.right_x < my_x and obs.top_y < my_y < ground.top_y <= obs.bot_y:
            if index == -1 or obstacles_circ[index].right_x < obs.right_x:
                index = i
    if index == -1:  # no walls found, there must be something below
        fall_zone_left = get_obstacle_below_point(ground.left_x, ground.top_y, obstacles_circ)
        left_down = (ground.left_x, fall_zone_left.top_y)
    else:
        left_down = (obstacles_circ[index].right_x, ground.top_y)

    index = -1
    # find any wall to the left
    for i, obs in enumerate(obstacles_circ):
        if obs.right_x < my_x and obs.top_y < my_y:  # wall to the left
            if index == -1 or obstacles_circ[index].right_x < obs.right_x:  # closest wall
                being_blocked = False
                for ceiling in obstacles_circ:  # nothing blocking the top right corner
                    if ceiling.left_x < obs.right_x < ceiling.right_x and obs.top_y < ceiling.bot_y < my_y:
                        being_blocked = True
                        break
                if not being_blocked:
                    index = i
    if index == -1:
        index = 2
    left_up = (obstacles_circ[index].right_x, obstacles_circ[index].top_y)

    # move right until fall or wall
    index = -1
    for i, obs in enumerate(obstacles_circ):
        if my_x < obs.left_x <= ground.right_x and obs.top_y < my_y < ground.top_y <= obs.bot_y:
            if index == -1 or obs.left_x < obstacles_circ[index].left_x:
                index = i
    if index == -1:  # no walls found, there must be something below
        fall_zone_right = get_obstacle_below_point(ground.right_x, ground.top_y, obstacles_circ)
        right_down = (ground.right_x, fall_zone_right.top_y)
    else:
        right_down = (obstacles_circ[index].left_x, ground.top_y)

    index = -1
    for i, obs in enumerate(obstacles_circ):
        if my_x < obs.left_x and obs.top_y < my_y:  # wall to the right
            if index == -1 or obs.left_x < obstacles_circ[index].left_x:  # closest wall
                being_blocked = False
                for ceiling in obstacles_circ:  # nothing blocking the top left corner
                    if ceiling.left_x < obs.left_x < ceiling.right_x and obs.top_y < ceiling.bot_y < my_y:
                        being_blocked = True
                        break
                if not being_blocked:
                    index = i
    if index == -1:
        index = 3
    right_up = (obstacles_circ[index].left_x, obstacles_circ[index].top_y)

    return ground, ceil, left_down, left_up, right_down, right_up


def get_obstacle_below_point(my_x, my_y, obstacles):
    index = 0  # default ground
    for i, obs in enumerate(obstacles):
        if obs.left_x < my_x < obs.right_x and my_y < obs.top_y < obstacles[index].top_y:
            index = i
    return obstacles[index]


def get_obstacle_above_point(my_x, my_y, obstacles):
    index = 1  # default ground
    for i, obs in enumerate(obstacles):
        if obs.left_x < my_x < obs.right_x and my_y > obs.bot_y > obstacles[index].bot_y:
            index = i
    return obstacles[index]
