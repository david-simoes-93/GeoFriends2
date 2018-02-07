#!/usr/bin/python3

import gym
from gym.spaces import *
import numpy as np
import random
import sys
import pygame


class Obstacle(object):
    def __init__(self, center_x, center_y, half_width, half_height):
        self.update_obs(center_x, center_y, half_width, half_height)

    def __str__(self):
        return "*(" + str(self.left_x) + "," + str(self.top_y) + "):(" + \
               str(self.right_x) + "," + str(self.bot_y) + ")*"

    def update_obs(self, center_x, center_y, half_width, half_height):
        self.top_y = center_y - half_height
        self.bot_y = center_y + half_height
        self.left_x = center_x - half_width
        self.right_x = center_x + half_width

        self.center_x = center_x
        self.center_y = center_y
        self.half_width = half_width
        self.half_height = half_height


class Map(object):
    def __init__(self, obstacles, circle, rectangle, rewards):
        self.obstacles = obstacles
        self.circle_pos = circle
        self.rectangle_pos = rectangle
        self.rewards = rewards

    def is_terminal(self, rectangle_pos, circle_pos):
        return False

class GymEnvGF(gym.Env):
    def __init__(self, rectangle=True, circle=False):
        # super?
        self.frameskip = 5
        self.air_movement = False
        self.square_interrupt_growth = False

        # Global info
        self.rewards = []
        self.obstacles = []
        self.obstacles_circle = []
        self.obstacles_rectangle = []
        self.terminal = False
        self.fps = 100

        # Circle info
        self.circle = circle
        self.circle_pos = None
        self.circle_vel = []
        self.circle_spin = 0
        self.circle_on_ground = False
        self.circle_ground = None
        self.circle_radius = 40

        # Rectangle info
        self.rectangle = rectangle
        self.rectangle_pos = None
        self.growing_side = False  # upwards
        self.rect_min, self.rect_max = 40, 200
        self.rect_w, self.rect_h = self.rect_max, self.rect_min
        self.rectangle_ground = None
        self.rectangle_ceiling = None

        # GUI
        self.metadata = {'render.modes': ['human']}
        self.screen = None

        # Public GYM variables
        # The Space object corresponding to valid actions
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(4), gym.spaces.Discrete(4)))
        # The Space object corresponding to valid observations
        self.observation_space = gym.spaces.Tuple((gym.spaces.Box(np.array([0, 0, -200, -500,
                                                                            0, 0, 0, 0, 0, 0]),
                                                                  np.array([1280, 800, 200, 500,
                                                                            1280, 800, 1280, 800, 1280, 800])),
                                                   gym.spaces.Box(np.array([0, 0, 0, 0,
                                                                            0, 0, 0, 0, 0, 0]),
                                                                  np.array([1280, 800, 1, self.rect_max,
                                                                            1280, 800, 1280, 800, 1280, 800]))))
        # A tuple corresponding to the min and max possible rewards
        self.reward_range = [0, 100]

        # Maps
        self.circle_maps = []
        self.rectangle_maps = []
        self.mixed_maps = []
        self.fill_maps()

    def fill_maps(self):
        # Circle map - high platform on left
        self.circle_maps.append(Map([Obstacle(480, 328, 480, 16)],
                                    [136, 264], [],
                                    [[136, 100], [1100, 272]]))
        self.circle_maps[-1].is_terminal = lambda rectangle_pos, circle_pos: circle_pos[1] > 700
        # Circle map - two big corners
        self.circle_maps.append(Map([Obstacle(136, 552, 96, 224), Obstacle(1072, 672, 168, 88)],
                                    [random.randint(80, 180), 264], [],
                                    [[random.randint(300, 550), 700], [random.randint(650, 850), 700],
                                     [random.randint(930, 1230), 400]]))
        # Circle map - wall in middle
        self.circle_maps.append(Map([Obstacle(640, 660, 200, 100)],
                                    [150, 700], [],
                                    [[random.randint(100, 300), random.randint(500, 700)],
                                     [random.randint(440, 840), random.randint(300, 500)],
                                     [random.randint(980, 1180), random.randint(500, 700)]]))
        # Circle map - two big corners
        self.circle_maps.append(Map([Obstacle(1144, 552, 96, 224), Obstacle(208, 672, 168, 88)],
                                    [random.randint(1000, 1200), 264], [],
                                    [[random.randint(750, 950), random.randint(500, 700)],
                                     [random.randint(400, 650), random.randint(500, 700)],
                                     [random.randint(50, 350), random.randint(300, 500)]]))
        # Circle map - high platform on right
        self.circle_maps.append(Map([Obstacle(800, 328, 480, 16)],
                                    [1150, 264], [],
                                    [[1150, 100], [180, 272]]))
        self.circle_maps[-1].is_terminal = lambda rectangle_pos, circle_pos: circle_pos[1] > 700
        #
        self.circle_maps.append(Map([Obstacle(400, 620, 40, 140), Obstacle(800, 620, 40, 140)],
                                    [200, 700], [],
                                    [[random.randint(500, 700), random.randint(350, 700)] if random.random() > 0.5 else
                                     [random.randint(900, 1100), random.randint(350, 700)],
                                     [400, random.randint(150, 400)], [800, random.randint(150, 400)]]))
        #
        self.circle_maps.append(Map([Obstacle(400, 620, 40, 140), Obstacle(800, 620, 40, 140)],
                                    [600, 700], [],
                                    [[random.randint(100, 300), random.randint(350, 700)] if random.random() > 0.5 else
                                     [random.randint(900, 1100), random.randint(350, 700)],
                                     [400, random.randint(150, 400)], [800, random.randint(150, 400)]]))
        #
        self.circle_maps.append(Map([Obstacle(400, 620, 40, 140), Obstacle(800, 620, 40, 140)],
                                    [1080, 700], [],
                                    [[random.randint(500, 700), random.randint(350, 700)] if random.random() > 0.5 else
                                     [random.randint(100, 300), random.randint(350, 700)],
                                     [400, random.randint(150, 400)], [800, random.randint(150, 400)]]))
        # Simple map, agents on left, reward on right
        self.circle_maps.append(Map([],
                                    [random.randint(700, 1150), 700], [],
                                    [[random.randint(100, 600), random.randint(400, 700)]]))
        # Simple map, agents on right, reward on left
        self.circle_maps.append(Map([],
                                    [random.randint(100, 600), 700], [],
                                    [[random.randint(700, 1150), random.randint(400, 700)]]))

        # Rectangle map - high platform on left
        self.rectangle_maps.append(Map([Obstacle(320, 500, 280, 20), Obstacle(960, 500, 280, 20)],
                                       [], [200, 450],
                                       [[random.randint(100, 600), random.randint(300, 450)],
                                        [random.randint(700, 1100), random.randint(600, 750)]]))
        # Rectangle map - high platform on left
        self.rectangle_maps.append(Map([Obstacle(320, 500, 280, 20), Obstacle(960, 500, 280, 20)],
                                       [], [200, 450],
                                       [[random.randint(100, 600), random.randint(300, 450)],
                                        [random.randint(700, 1100), random.randint(300, 450)]]))
        # Rectangle map - high platform on left
        self.rectangle_maps.append(Map([Obstacle(320, 500, 280, 20), Obstacle(960, 500, 280, 20)],
                                       [], [1080, 450],
                                       [[random.randint(700, 1100), random.randint(300, 450)],
                                        [random.randint(100, 600), random.randint(600, 750)]]))
        # Rectangle map - high platform on left
        self.rectangle_maps.append(Map([Obstacle(320, 500, 280, 20), Obstacle(960, 500, 280, 20)],
                                       [], [1080, 450],
                                       [[random.randint(700, 1100), random.randint(300, 450)],
                                        [random.randint(100, 600), random.randint(300, 450)]]))
        # Rectangle map - high platform on left
        self.rectangle_maps.append(Map([Obstacle(320, 500, 280, 20), Obstacle(960, 500, 280, 20)],
                                       [], [1080, 450],
                                       [[random.randint(700, 1100), random.randint(600, 750)]]))
        # Rectangle map - high platform on left
        self.rectangle_maps.append(Map([Obstacle(320, 500, 280, 20), Obstacle(960, 500, 280, 20)],
                                       [], [200, 450],
                                       [[random.randint(100, 600), random.randint(600, 750)]]))
        # Rectangle map - two platforms
        self.rectangle_maps.append(Map([Obstacle(200, 500, 160, 20), Obstacle(840, 500, 400, 20),
                                        Obstacle(440, 260, 400, 20), Obstacle(1080, 260, 160, 20)],
                                       [], [200, 150],
                                       [[1000, 600], [200, 600], [1000, 380], [200, 380], [1000, 100], [200, 100]]))

        # Simple map, agents on left, reward on right
        self.rectangle_maps.append(Map([],
                                       [], [random.randint(700, 1000), 700],
                                       [[random.randint(100, 600), random.randint(550, 700)]]))
        # Simple map, agents on right, reward on left
        self.rectangle_maps.append(Map([],
                                       [], [random.randint(280, 600), 700],
                                       [[random.randint(700, 1150), random.randint(550, 700)]]))

        # Simple map, agents on left, reward on right
        self.mixed_maps.append(Map([],
                                   [random.randint(700, 1150), 700], [random.randint(700, 1000), 700],
                                   [[random.randint(100, 600), random.randint(300, 700)]]))
        # Simple map, agents on right, reward on left
        self.mixed_maps.append(Map([],
                                   [random.randint(100, 600), 700], [random.randint(280, 600), 700],
                                   [[random.randint(700, 1150), random.randint(300, 700)]]))
        # mixed map with towers
        self.mixed_maps.append(Map([Obstacle(400, 570, 40, 190), Obstacle(800, 570, 40, 190)],
                                   [random.randint(100, 200), 700], [random.randint(200, 300), 700],
                                    [[random.randint(500, 700), random.randint(350, 700)],
                                     [400, random.randint(150, 300)], [800, random.randint(150, 300)]]))
        #

    def _render(self, mode='human', close=False):
        if close:
            pygame.quit()
            self.screen = None
            return

        if mode is not 'human':
            super(GymEnvGF, self).render(mode=mode)
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode([1280, 800])
            pygame.display.set_caption("GymGF2")

        self.screen.fill((0, 0, 255))

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (0, 0, 0),
                             [obs.left_x, obs.top_y, obs.right_x - obs.left_x, obs.bot_y - obs.top_y])

        # Draw circle
        if self.circle:
            pygame.draw.circle(self.screen, (255, 255, 0),
                               [int(self.circle_pos[0]), int(self.circle_pos[1])],
                               self.circle_radius)

        # Draw square
        if self.rectangle:
            pygame.draw.rect(self.screen, (0, 255, 0),
                             [int(self.rectangle_pos[0] - self.rect_w / 2),
                              int(self.rectangle_pos[1] - self.rect_h / 2),
                              self.rect_w, self.rect_h])

        # Draw rewards
        for reward in self.rewards:
            pygame.draw.circle(self.screen, (255, 0, 255), [int(reward[0]), int(reward[1])], 25)

        pygame.display.flip()

    def _reset(self):
        self.obstacles = [Obstacle(640, 780, 640, 20), Obstacle(640, 20, 640, 20),
                          Obstacle(20, 400, 20, 400), Obstacle(1260, 400, 20, 400)]
        self.obstacles_circle = []
        self.obstacles_rectangle = []
        self.terminal = False
        self.circle_vel = [0, 0]
        self.circle_spin = 0
        self.rect_w, self.rect_h = self.rect_max, self.rect_min
        self.growing_side = True  # growing sideways

        if self.rectangle and not self.circle:
            self.map = random.choice(self.rectangle_maps)
        elif not self.rectangle and self.circle:
            self.map = random.choice(self.circle_maps)
        else:
            self.map = random.choice(self.mixed_maps)
        self.obstacles += self.map.obstacles
        self.circle_pos = list(self.map.circle_pos)
        self.rectangle_pos = list(self.map.rectangle_pos)
        self.rewards = list(self.map.rewards)

        self.obstacles_circle += self.obstacles
        if self.rectangle:
            self.obstacles_circle += [Obstacle(self.rectangle_pos[0], self.rectangle_pos[1], self.rect_w / 2, self.rect_h / 2)]
        self.obstacles_rectangle += self.obstacles

        if self.circle:
            self.set_on_ground_circle()
        if self.rectangle:
            self.set_on_ground_rectangle()

        return self.get_rect_state(), self.get_circ_state()

    def _step(self, action):
        state, reward, terminal, extra_info = self._step_single_action(action[0], action[1])
        repeat_action_rect = action[0] if action[0]!=2 else 3
        repeat_action_circ = action[0] if action[0] != 2 else 3;
        for i in range(1, self.frameskip):
            state, reward_new, terminal_new, extra_info = self._step_single_action(
                repeat_action_rect, repeat_action_circ)
            reward += reward_new
            terminal |= terminal_new
        return state, reward, terminal, extra_info

    def _step_single_action(self, action_rectangle, action_circle):
        reward = 0

        if self.circle:
            self.circle_spin *= 0.99

            # Circle movement
            if action_circle == 0:  # LEFT
                self.circle_spin = self.circle_spin - 2
            elif action_circle == 1:  # RIGHT
                self.circle_spin = self.circle_spin + 2
            elif action_circle == 2 and self.circle_on_ground:  # JUMP
                self.circle_vel[1] = -440
            elif action_circle == 3:  # NOTHING
                pass

            # move on air
            if self.air_movement:
                self.circle_vel[0] = self.circle_spin
            # or keep velocity from being affected while on_air
            elif self.circle_on_ground:
                self.circle_vel[0] = self.circle_spin

            # gravity
            self.circle_vel[1] += 3

            self.circle_pos[0] += self.circle_vel[0] / self.fps
            self.circle_pos[1] += self.circle_vel[1] / self.fps

            # move circle out of obstacles
            for obs in self.obstacles_circle:
                if obs.left_x < self.circle_pos[0] < obs.right_x:
                    if obs.top_y < self.circle_pos[1] + self.circle_radius < obs.bot_y:  # ground
                        self.circle_pos[1] = obs.top_y - self.circle_radius
                        self.circle_vel[1] = bounce_speed(self.circle_vel[1])
                    elif obs.top_y < self.circle_pos[1] - self.circle_radius < obs.bot_y:  # ceiling
                        self.circle_pos[1] = obs.bot_y + self.circle_radius
                        self.circle_vel[1] = bounce_speed(self.circle_vel[1])

                if obs.top_y < self.circle_pos[1] < obs.bot_y:
                    if obs.left_x < self.circle_pos[0] - self.circle_radius < obs.right_x:  # left wall
                        self.circle_pos[0] = obs.right_x + self.circle_radius
                        self.circle_vel[0] = bounce_speed(self.circle_vel[0])
                        self.circle_spin = bounce_speed(self.circle_spin)
                    elif obs.left_x < self.circle_pos[0] + self.circle_radius < obs.right_x:  # right wall
                        self.circle_pos[0] = obs.left_x - self.circle_radius
                        self.circle_vel[0] = bounce_speed(self.circle_vel[0])
                        self.circle_spin = bounce_speed(self.circle_spin)

                        # TODO corner cases

            self.set_on_ground_circle()

            # rewards
            i = 0
            while i < len(self.rewards):
                if distance(self.rewards[i], self.circle_pos) < self.circle_radius + 25:
                    self.rewards.remove(self.rewards[i])
                    i -= 1
                    reward += 1
                i += 1

        if self.rectangle:
            can_grow_side = self.rect_w - 200 / self.fps < self.rect_max
            can_grow_up = self.rect_w - 200 / self.fps > self.rect_min

            # Rectangle movement
            if action_rectangle == 0:  # LEFT
                self.rectangle_pos[0] -= 500 / self.fps
            elif action_rectangle == 1:  # RIGHT
                self.rectangle_pos[0] += 500 / self.fps
            elif action_rectangle == 2:  # RESIZE
                if self.square_interrupt_growth:
                    self.growing_side = not self.growing_side
                else:
                    if not can_grow_up:
                        self.growing_side = True
                    elif not can_grow_side:
                        self.growing_side = False
            elif action_rectangle == 3:  # NOTHING
                pass

            self.rectangle_pos[1] += 300 / self.fps

            if not self.growing_side and can_grow_up:
                # if can grow upwards
                self.rectangle_pos[1] -= 100 / self.fps
                self.rect_w = self.rect_w - 200 / self.fps
                self.rect_h = self.rect_h + 200 / self.fps
            elif self.growing_side and can_grow_side:
                # if can grow sideways
                self.rectangle_pos[1] += 100 / self.fps
                self.rect_w = self.rect_w + 200 / self.fps
                self.rect_h = self.rect_h - 200 / self.fps

            # move rectangle out of obstacles
            for obs in self.obstacles_rectangle:
                if obs.left_x < self.rectangle_pos[0] - self.rect_w / 2 < obs.right_x:
                    if obs.top_y < self.rectangle_pos[1] + self.rect_h / 2 < obs.top_y + 301 / self.fps:  # ground
                        self.rectangle_pos[1] = obs.top_y - self.rect_h / 2

                if obs.left_x < self.rectangle_pos[0] + self.rect_w / 2 < obs.right_x:
                    if obs.top_y < self.rectangle_pos[1] + self.rect_h / 2 < obs.top_y + 301 / self.fps:  # ground
                        self.rectangle_pos[1] = obs.top_y - self.rect_h / 2

                if self.rectangle_pos[1] - self.rect_h / 2 < obs.center_y < self.rectangle_pos[1] + self.rect_h / 2:
                    if obs.left_x < self.rectangle_pos[0] - self.rect_w / 2 < obs.right_x:  # left wall
                        self.rectangle_pos[0] = obs.right_x + self.rect_w / 2
                    elif obs.left_x < self.rectangle_pos[0] + self.rect_w / 2 < obs.right_x:  # right wall
                        self.rectangle_pos[0] = obs.left_x - self.rect_w / 2

                if obs.top_y < self.rectangle_pos[1] + self.rect_h / 2 < obs.bot_y:
                    if obs.left_x < self.rectangle_pos[0] - self.rect_w / 2 < obs.right_x:  # left wall
                        self.rectangle_pos[0] = obs.right_x + self.rect_w / 2
                    elif obs.left_x < self.rectangle_pos[0] + self.rect_w / 2 < obs.right_x:  # right wall
                        self.rectangle_pos[0] = obs.left_x - self.rect_w / 2

                if obs.top_y < self.rectangle_pos[1] - self.rect_h / 2 < obs.bot_y:
                    if obs.left_x < self.rectangle_pos[0] - self.rect_w / 2 < obs.right_x:  # left wall
                        self.rectangle_pos[0] = obs.right_x + self.rect_w / 2
                    elif obs.left_x < self.rectangle_pos[0] + self.rect_w / 2 < obs.right_x:  # right wall
                        self.rectangle_pos[0] = obs.left_x - self.rect_w / 2

            if self.circle:
                self.obstacles_circle[-1].update_obs(self.rectangle_pos[0], self.rectangle_pos[1],
                                                     self.rect_w / 2, self.rect_h / 2)

            self.set_on_ground_rectangle()

            # rewards
            i = 0
            while i < len(self.rewards):
                if intersects(self.rectangle_pos, [self.rect_w, self.rect_h], self.rewards[i], 25):
                    self.rewards.remove(self.rewards[i])
                    i -= 1
                    reward += 1
                i += 1

        self.terminal = len(self.rewards) == 0 or self.map.is_terminal(self.rectangle_pos,self.circle_pos)

        return (self.get_rect_state(), self.get_circ_state()), reward * 100, self.terminal, {}

    # computes the circle's observations
    def get_circ_state(self):
        if not self.circle:
            return np.zeros(10)
        state = [self.circle_pos[0], self.circle_pos[1], self.circle_vel[0], self.circle_vel[1]]
        for [x, y] in self.rewards:
            state += [x, y]
        while len(state) < 10:
            state += [0, 0]
        return state

    # computes the rectangle's observations
    def get_rect_state(self):
        if not self.rectangle:
            return np.zeros(10)
        state = [self.rectangle_pos[0], self.rectangle_pos[1], int(self.growing_side), self.rect_w]
        for [x, y] in self.rewards:
            state += [x, y]
        while len(state) < 10:
            state += [0, 0]
        return state

    # returns the closest platform below the circle
    def get_ground_circle(self, pos):
        index = 0  # default ground

        for i, obs in enumerate(self.obstacles_circle):
            if obs.left_x - self.circle_radius < pos[0] < obs.right_x + self.circle_radius and \
                                    pos[1] < obs.top_y < self.obstacles_circle[index].top_y:
                index = i

        return self.obstacles_circle[index]

    # checks and sets if circle is touching the ground
    def set_on_ground_circle(self):
        self.circle_ground = self.get_ground_circle(self.circle_pos)

        # if completely on top and y=40, on ground
        if self.circle_ground.left_x < self.circle_pos[0] < self.circle_ground.right_x:
            self.circle_on_ground = self.circle_ground.top_y - self.circle_pos[1] < self.circle_radius + 1
        # if to the side and d=40, on ground
        elif self.circle_pos[0] < self.circle_ground.left_x:
            self.circle_on_ground = distance([self.circle_ground.left_x, self.circle_ground.top_y],
                                             self.circle_pos) < self.circle_radius + 1
        # if to the side and d=40, on ground
        elif self.circle_ground.right_x < self.circle_pos[0]:
            self.circle_on_ground = distance([self.circle_ground.right_x, self.circle_ground.top_y],
                                             self.circle_pos) < self.circle_radius + 1

    # checks and sets if rectangle is touching the ground
    def set_on_ground_rectangle(self):
        index_g = 0  # default ground
        index_c = 1  # default ceiling

        for i, obs in enumerate(self.obstacles_rectangle):
            if (obs.left_x < self.rectangle_pos[0] - self.rect_w / 2 < obs.right_x or
                            obs.left_x < self.rectangle_pos[0] + self.rect_w / 2 < obs.right_x):
                if self.rectangle_pos[1] < obs.top_y < self.obstacles_rectangle[index_g].top_y:
                    index_g = i
                elif self.rectangle_pos[1] < obs.top_y == self.obstacles_rectangle[index_g].top_y:
                    # multiple grounds, pick the one with most area
                    if obs.center_x < self.obstacles_rectangle[index_g].center_x and \
                                    abs(obs.right_x - self.rectangle_pos[0]) < abs(
                                        self.obstacles_rectangle[index_g].right_x - self.rectangle_pos[0]):
                        index_g = i
                if self.obstacles_rectangle[index_g].bot_y < obs.bot_y < self.rectangle_pos[1]:
                    index_c = i

        self.rectangle_ground = self.obstacles_rectangle[index_g]
        self.rectangle_ceiling = self.obstacles_rectangle[index_c]

    def _close(self):
        self.render(close=True)
        return

    def _seed(self, seed=None):
        if seed is None:
            seed = random.randrange(sys.maxsize)
        random.seed(seed)
        return [seed]


# get distance between points A and B
def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# calculate new speed value when ball hits something
def bounce_speed(speed):
    bounce_spd = 0.0001 * speed ** 2 - 0.3785 * speed - 0.5477
    if abs(bounce_spd) < 1.3:
        bounce_spd = 0
    return bounce_spd


# returns whether a rectangle with center at rectangle_pos and rectangle_size=[width,height]
# intersects a circle at reward_pos with radius reward_radius
def intersects(rectangle_pos, rectangle_size, reward_pos, reward_radius):
    circle_distance = [abs(reward_pos[0] - rectangle_pos[0]), abs(reward_pos[1] - rectangle_pos[1])]

    if circle_distance[0] > rectangle_size[0] / 2 + reward_radius:
        return False
    if circle_distance[1] > rectangle_size[1] / 2 + reward_radius:
        return False

    if circle_distance[0] <= rectangle_size[0] / 2:
        return True
    if circle_distance[1] <= rectangle_size[1] / 2:
        return True

    corner_distance_sq = (circle_distance[0] - rectangle_size[0] / 2) ** 2 + \
                         (circle_distance[1] - rectangle_size[1] / 2) ** 2

    return corner_distance_sq <= reward_radius ** 2
