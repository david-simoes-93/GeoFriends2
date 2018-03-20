#!/usr/bin/python3

import gym
from gym.spaces import *
from pygame.locals import *
import numpy as np
import sys
import pygame
from simulator.GFMaps import *


class GymEnvGF(gym.Env):
    # rectangle = whether to enable rectangle agent
    # circle = whether to enable circle agent
    # frameskip = amount of actions to repeat per step
    # air_movement = whether to allow agents to move while on air
    # square_interrupt_growth = whether to allow rectangle to invert its growth sequence
    def __init__(self, rectangle=True, circle=False, frameskip=1, air_movement=False, square_interrupt_growth=True,
                 screen_res=[640, 400]):
        # super?
        self.frameskip = frameskip
        self.air_movement = air_movement
        self.square_interrupt_growth = square_interrupt_growth
        self.screen_res = screen_res

        # Global info
        self.rewards = []
        self.obstacles = []
        self.obstacles_circle = []
        self.obstacles_rectangle = []
        self.terminal = False

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
        self.reward_range = [0, 1]

        # Maps
        self.circle_maps = []
        self.rectangle_maps = []
        self.mixed_maps = []
        fill_maps(self)

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
            self.screen = pygame.surface.Surface((1280, 800))  # original GF size
            self.gui_window = pygame.display.set_mode(self.screen_res, HWSURFACE | DOUBLEBUF | RESIZABLE)
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

        self.gui_window.blit(pygame.transform.scale(self.screen, self.screen_res), (0, 0))
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
            rect_maps(self)
            self.map = random.choice(self.rectangle_maps)
        elif not self.rectangle and self.circle:
            circle_maps(self)
            self.map = random.choice(self.circle_maps)
        else:
            both_maps(self)
            self.map = random.choice(self.mixed_maps)
        self.obstacles += self.map.obstacles
        self.circle_pos = list(self.map.circle_pos)
        self.rectangle_pos = list(self.map.rectangle_pos)
        self.rewards = list(self.map.rewards)

        self.obstacles_circle += self.obstacles
        self.obstacles_rectangle += self.obstacles
        """if self.rectangle and self.circle:
            self.obstacles_circle += [Obstacle(self.rectangle_pos[0], self.rectangle_pos[1],
                                               self.rect_w / 2, self.rect_h / 2)]
            self.obstacles_rectangle += [Obstacle(self.circle_pos[0], self.circle_pos[1], 40, 40)]"""

        if self.circle:
            self.set_on_ground_circle()
        if self.rectangle:
            self.set_on_ground_rectangle()

        return (self.get_rect_state(), self.get_circ_state())

    def _step(self, action):
        state, reward, terminal, extra_info = self._step_single_action(action[0], action[1])
        repeat_action_rect = action[0] if action[0] != 2 else 3
        repeat_action_circ = action[1] if action[1] != 2 else 3
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
                self.circle_spin = self.circle_spin - 0.02
            elif action_circle == 1:  # RIGHT
                self.circle_spin = self.circle_spin + 0.02
            elif action_circle == 2 and self.circle_on_ground:  # JUMP
                self.circle_vel[1] = -4.40
            elif action_circle == 3:  # NOTHING
                pass

            # move on air if allowed, or only move while on ground
            if self.air_movement or self.circle_on_ground:
                self.circle_vel[0] = self.circle_spin
            # gravity
            self.circle_vel[1] += 0.03

            self.circle_pos[0] += self.circle_vel[0]
            self.circle_pos[1] += self.circle_vel[1]

            # move circle out of obstacles
            circle_wall = None
            current_moved_x, current_moved_y = 0, 0
            for obs in self.obstacles_circle:
                if intersects([obs.center_x, obs.center_y], [obs.half_width * 2, obs.half_height * 2],
                              self.circle_pos, self.circle_radius - 0.01):
                    # circle crossed horizontal lines of obstacle
                    if obs.left_x < self.circle_pos[0] < obs.right_x:
                        # fell inside obstacle below
                        if obs.top_y < self.circle_pos[1] + self.circle_radius < obs.bot_y:
                            current_moved_y += (obs.top_y - self.circle_radius) - self.circle_pos[1]
                            self.circle_pos[1] = obs.top_y - self.circle_radius
                            self.circle_vel[1] = bounce_speed(self.circle_vel[1])
                        # jumped into obstacle above
                        else:
                            current_moved_y += (obs.bot_y + self.circle_radius) - self.circle_pos[1]
                            self.circle_pos[1] = obs.bot_y + self.circle_radius
                            self.circle_vel[1] = bounce_speed(self.circle_vel[1])
                    # circle crossed vertical lines of obstacle
                    elif obs.top_y < self.circle_pos[1] < obs.bot_y:
                        circle_wall = obs
                        # inside a wall on circle's left
                        if obs.left_x < self.circle_pos[0] - self.circle_radius < obs.right_x:
                            current_moved_x += (obs.right_x + self.circle_radius) - self.circle_pos[0]
                            self.circle_pos[0] = obs.right_x + self.circle_radius
                            self.circle_vel[0] = bounce_speed(self.circle_vel[0])
                            self.circle_spin = bounce_speed(self.circle_spin)
                        # inside a wall on circle's right
                        else:
                            current_moved_x += (obs.left_x - self.circle_radius) - self.circle_pos[0]
                            self.circle_pos[0] = obs.left_x - self.circle_radius
                            self.circle_vel[0] = bounce_speed(self.circle_vel[0])
                            self.circle_spin = bounce_speed(self.circle_spin)
                    # some corner interception
                    else:
                        circle_wall = obs
                        # the mod_x and mod_y allow us to treat all 4 corners in the same way and then just invert x or y as necessary
                        mod_x = 1 if obs.center_x - self.circle_pos[0] > 0 else -1
                        mod_y = 1 if obs.center_y - self.circle_pos[1] > 0 else -1
                        dist_x = np.abs(obs.center_x - self.circle_pos[0]) - obs.half_width
                        dist_y = np.abs(obs.center_y - self.circle_pos[1]) - obs.half_height

                        # if distance in Y is larger (so circle is more up or down), we change Y (we affect height)
                        if dist_x < dist_y:
                            new_dist_x = 0
                            new_dist_y = np.sqrt(self.circle_radius ** 2 - dist_x ** 2) - dist_y
                        # else if distance in X is larger (so circle is more to the side), we change X (we affect width)
                        else:
                            new_dist_x = np.sqrt(self.circle_radius ** 2 - dist_y ** 2) - dist_x
                            new_dist_y = 0

                        # if movement to dodge corner contradicts movement to avoid another obstacle, we change x and y
                        if new_dist_x*current_moved_x < 0:
                            new_dist_y = new_dist_x
                        if new_dist_y*current_moved_y < 0:
                            new_dist_x = new_dist_y

                        current_moved_x += (new_dist_x * -mod_x)
                        self.circle_pos[0] += new_dist_x * -mod_x
                        current_moved_y += (new_dist_y * -mod_y)
                        self.circle_pos[1] += new_dist_y * -mod_y

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
            can_grow_side = self.rect_w - 2 < self.rect_max
            can_grow_up = self.rect_w - 2 > self.rect_min

            # Rectangle movement
            if action_rectangle == 0:  # LEFT
                self.rectangle_pos[0] -= 5
            elif action_rectangle == 1:  # RIGHT
                self.rectangle_pos[0] += 5
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

            self.rectangle_pos[1] += 3

            if not self.growing_side and can_grow_up:
                # if can grow upwards
                self.rectangle_pos[1] -= 1
                self.rect_w = self.rect_w - 2
                self.rect_h = self.rect_h + 2
            elif self.growing_side and can_grow_side:
                # if can grow sideways
                self.rectangle_pos[1] += 1
                self.rect_w = self.rect_w + 2
                self.rect_h = self.rect_h - 2

            # move rectangle out of obstacles
            for obs in self.obstacles_rectangle:
                if obs.left_x < self.rectangle_pos[0] - self.rect_w / 2 < obs.right_x:
                    if obs.top_y < self.rectangle_pos[1] + self.rect_h / 2 < obs.top_y + 3.01:  # ground
                        self.rectangle_pos[1] = obs.top_y - self.rect_h / 2

                if obs.left_x < self.rectangle_pos[0] + self.rect_w / 2 < obs.right_x:
                    if obs.top_y < self.rectangle_pos[1] + self.rect_h / 2 < obs.top_y + 3.01:  # ground
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

            self.set_on_ground_rectangle()

            # rewards
            i = 0
            while i < len(self.rewards):
                if intersects(self.rectangle_pos, [self.rect_w, self.rect_h], self.rewards[i], 25):
                    self.rewards.remove(self.rewards[i])
                    i -= 1
                    reward += 1
                i += 1

        # keep circle from hitting rectangle
        if self.circle and self.rectangle:
            if intersects(self.rectangle_pos, [self.rect_w, self.rect_h], self.circle_pos, self.circle_radius):
                # TODO when we move rectangle out of circle, we should ensure its not going inside other obstacles
                rect_as_an_obstacle = Obstacle(self.rectangle_pos[0], self.rectangle_pos[1],
                                               self.rect_w / 2, self.rect_h / 2)
                # circle crossed horizontal lines of rectangle
                if rect_as_an_obstacle.left_x <= self.circle_pos[0] <= rect_as_an_obstacle.right_x:
                    # fell inside rectangle below, move circle up
                    if rect_as_an_obstacle.top_y <= self.circle_pos[1] + self.circle_radius <= rect_as_an_obstacle.bot_y:
                        self.circle_pos[1] = rect_as_an_obstacle.top_y - self.circle_radius
                        self.circle_vel[1] = bounce_speed(self.circle_vel[1])
                    # jumped into rectangle above or rectangle fell on circle
                    elif rect_as_an_obstacle.top_y <= self.circle_pos[1] - self.circle_radius <= rect_as_an_obstacle.bot_y:
                        if self.circle_on_ground:  # circle on ground, rectangle moves up
                            self.rectangle_pos[1] = self.circle_pos[1] - self.circle_radius - self.rect_h / 2
                        else:  # circle not on ground, circle moves down
                            self.circle_pos[1] = rect_as_an_obstacle.bot_y + self.circle_radius
                            self.circle_vel[1] = bounce_speed(self.circle_vel[1])
                # circle crossed vertical lines of rectangle
                elif rect_as_an_obstacle.top_y <= self.circle_pos[1] <= rect_as_an_obstacle.bot_y:
                    # rectangle on circle's left
                    if rect_as_an_obstacle.left_x <= self.circle_pos[0] - self.circle_radius <= rect_as_an_obstacle.right_x:
                        # if circle next to wall, move rectangle
                        if circle_wall is not None:
                            self.rectangle_pos[0] = self.circle_pos[0] - self.circle_radius - self.rect_w / 2
                        # else, move circle
                        else:
                            self.circle_pos[0] = rect_as_an_obstacle.right_x + self.circle_radius
                            self.circle_vel[0] = bounce_speed(self.circle_vel[0])
                            self.circle_spin = bounce_speed(self.circle_spin)
                    # rectangle on circle's right
                    else:
                        # if circle next to wall, move rectangle
                        if circle_wall is not None:
                            self.rectangle_pos[0] = self.circle_pos[0] + self.circle_radius + self.rect_w / 2
                        # else, move circle
                        else:
                            self.circle_pos[0] = rect_as_an_obstacle.left_x - self.circle_radius
                            self.circle_vel[0] = bounce_speed(self.circle_vel[0])
                            self.circle_spin = bounce_speed(self.circle_spin)
                            # self.circle_pos[1] -= 5
                            # pass
                else:
                    # the mod_x and mod_y allow us to treat all 4 corners in the same way and then just invert x or y as necessary
                    mod_x = 1 if self.rectangle_pos[0] - self.circle_pos[0] > 0 else -1
                    mod_y = 1 if self.rectangle_pos[1] - self.circle_pos[1] > 0 else -1
                    dist_x = np.abs(self.rectangle_pos[0] - self.circle_pos[0]) - self.rect_w / 2
                    dist_y = np.abs(self.rectangle_pos[1] - self.circle_pos[1]) - self.rect_h / 2

                    # if distance in Y is larger (so circle is more up or down), we change Y (we affect height)
                    if dist_x < dist_y:
                        new_dist_x = 0
                        new_dist_y = np.sqrt(self.circle_radius ** 2 - dist_x ** 2) - dist_y
                    # else if distance in X is larger (so circle is more to the side), we change X (we affect width)
                    else:
                        new_dist_x = np.sqrt(self.circle_radius**2 - dist_y**2) - dist_x
                        new_dist_y = 0

                    # if circle next to wall or on ground, move rectangle
                    if circle_wall is not None:
                        self.rectangle_pos[0] += new_dist_x * mod_x
                    # else, move circle
                    else:
                        # if movement to dodge corner contradicts movement to avoid another obstacle, we change x and y
                        if new_dist_y * current_moved_y < 0:
                            new_dist_x = new_dist_y

                        self.circle_pos[0] += new_dist_x * -mod_x

                    # if circle's lower part is below rectangle's lower part, move rectangle
                    if self.circle_pos[1]+self.circle_radius > self.rectangle_pos[1]+self.rect_h/2:
                        self.rectangle_pos[1] += new_dist_y * mod_y
                    # else, move circle
                    else:
                        # if movement to dodge corner contradicts movement to avoid another obstacle, we change x and y
                        if new_dist_x * current_moved_x < 0:
                            new_dist_y = new_dist_x

                        self.circle_pos[1] += new_dist_y * -mod_y

        self.terminal = len(self.rewards) == 0 or self.map.is_terminal(self.rectangle_pos, self.circle_pos,
                                                                       self.rewards)

        return (self.get_rect_state(), self.get_circ_state()), reward, self.terminal, {}

    # computes the circle's observations
    # [ POS X, POS Y, SPEED X, SPEED Y, REWARD_1 X, REWARD_1 Y, REWARD_2 X, REWARD_2 Y, REWARD_3 X, REWARD_3 Y]
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
    # [ POS X, POS Y, GROWING_SIDEWAYS, WIDTH, REWARD_1 X, REWARD_1 Y, REWARD_2 X, REWARD_2 Y, REWARD_3 X, REWARD_3 Y]
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

        # check if rectangle is ground
        if self.rectangle:
            rect_as_an_obstacle = Obstacle(self.rectangle_pos[0], self.rectangle_pos[1], self.rect_w / 2,
                                           self.rect_h / 2)
            if rect_as_an_obstacle.left_x - self.circle_radius < pos[
                0] < rect_as_an_obstacle.right_x + self.circle_radius and \
                                    pos[1] < rect_as_an_obstacle.top_y < self.obstacles_circle[index].top_y:
                return rect_as_an_obstacle
        return self.obstacles_circle[index]

    # checks and sets if circle is touching the ground
    def set_on_ground_circle(self):
        self.circle_ground = self.get_ground_circle(self.circle_pos)

        # if completely on top and y=40, on ground
        if self.circle_ground.left_x < self.circle_pos[0] < self.circle_ground.right_x:
            self.circle_on_ground = self.circle_ground.top_y - self.circle_pos[1] < self.circle_radius + 0.01
        # if to the side and d=40, on ground
        elif self.circle_pos[0] < self.circle_ground.left_x:
            self.circle_on_ground = distance([self.circle_ground.left_x, self.circle_ground.top_y],
                                             self.circle_pos) < self.circle_radius + 0.01
        # if to the side and d=40, on ground
        elif self.circle_ground.right_x < self.circle_pos[0]:
            self.circle_on_ground = distance([self.circle_ground.right_x, self.circle_ground.top_y],
                                             self.circle_pos) < self.circle_radius + 0.01

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
    speed *= 100
    bounce_spd = 0.0001 * speed ** 2 - 0.3785 * speed - 0.5477
    if abs(bounce_spd) < 1.3:
        bounce_spd = 0
    else:
        bounce_spd /= 100
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
