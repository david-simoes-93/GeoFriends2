import random


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
        self.rewards = rewards  # each map should haxe 3 rewards maximum, for consistency

    def is_terminal(self, rectangle_pos, circle_pos, rewards):
        # this function can be overriden for maps that have points of no return
        return False


def circle_maps(env):
    env.circle_maps = []
    """""
    # High platform on left
    env.circle_maps.append(Map([Obstacle(480, 328, 480, 16)],
                               [136, 264], [],
                               [[136, 100], [1100, 272]]))
    env.circle_maps[-1].is_terminal = lambda rectangle_pos, circle_pos, rewards: circle_pos[1] > 700

    # high corner on left, low corner on right
    env.circle_maps.append(Map([Obstacle(136, 552, 96, 224), Obstacle(1072, 672, 168, 88)],
                               [random.randint(80, 180), 264], [],
                               [[random.randint(300, 550), 700], [random.randint(650, 850), 700],
                                [random.randint(930, 1230), 400]]))

    # obstacle in middle
    env.circle_maps.append(Map([Obstacle(640, 660, 200, 100)],
                               [150, 700], [],
                               [[random.randint(100, 300), random.randint(500, 700)],
                                [random.randint(440, 840), random.randint(300, 500)],
                                [random.randint(980, 1180), random.randint(500, 700)]]))

    # high corner on right, low corner on left
    env.circle_maps.append(Map([Obstacle(1144, 552, 96, 224), Obstacle(208, 672, 168, 88)],
                               [random.randint(1000, 1200), 264], [],
                               [[random.randint(750, 950), random.randint(500, 700)],
                                [random.randint(400, 650), random.randint(500, 700)],
                                [random.randint(50, 350), random.randint(300, 500)]]))

    # High platform on right
    env.circle_maps.append(Map([Obstacle(800, 328, 480, 16)],
                               [1150, 264], [],
                               [[1150, 100], [180, 272]]))
    env.circle_maps[-1].is_terminal = lambda rectangle_pos, circle_pos, rewards: circle_pos[1] > 700
    """
    # two narrow obstacles, circle on left
    env.circle_maps.append(Map([Obstacle(400, 620, 40, 140), Obstacle(800, 620, 40, 140)],
                               [200, 700], [],
                               [[random.randint(500, 700), random.randint(350, 700)] if random.random() > 0.5 else
                                [random.randint(900, 1100), random.randint(350, 700)],
                                [400, random.randint(150, 400)], [800, random.randint(150, 400)]]))

    # two narrow obstacles, circle on center
    env.circle_maps.append(Map([Obstacle(400, 620, 40, 140), Obstacle(800, 620, 40, 140)],
                               [600, 700], [],
                               [[random.randint(100, 300), random.randint(350, 700)] if random.random() > 0.5 else
                                [random.randint(900, 1100), random.randint(350, 700)],
                                [400, random.randint(150, 400)], [800, random.randint(150, 400)]]))

    # two narrow obstacles, circle on right
    env.circle_maps.append(Map([Obstacle(400, 620, 40, 140), Obstacle(800, 620, 40, 140)],
                               [1080, 700], [],
                               [[random.randint(500, 700), random.randint(350, 700)] if random.random() > 0.5 else
                                [random.randint(100, 300), random.randint(350, 700)],
                                [400, random.randint(150, 400)], [800, random.randint(150, 400)]]))

    # Simple map, agent on right, reward on left
    env.circle_maps.append(Map([],
                               [random.randint(700, 1150), 700], [],
                               [[random.randint(100, 600), random.randint(400, 700)]]))

    # Simple map, agent on left, reward on right
    env.circle_maps.append(Map([],
                               [random.randint(100, 600), 700], [],
                               [[random.randint(700, 1150), random.randint(400, 700)]]))


def rect_maps(env):
    env.rectangle_maps = []

    # horizontal floor, narrow gap in middle, rect on upper, 1 reward per floor
    env.rectangle_maps.append(Map([Obstacle(320, 500, 280, 20), Obstacle(960, 500, 280, 20)],
                                  [], [200, 450] if random.random() > 0.5 else [1080, 450],
                                  [[random.randint(100, 600) if random.random() > 0.5 else random.randint(700, 1100),
                                    random.randint(300, 450)],
                                   [random.randint(700, 1100) if random.random() > 0.5 else random.randint(100, 600),
                                    random.randint(600, 750)]]))
    env.rectangle_maps[-1].is_terminal = lambda rectangle_pos, circle_pos, rewards: \
        rectangle_pos[1] > 660 and len(rewards) > 0 and rewards[-1][1] < 500
    
    # horizontal floor, narrow gap in middle, rect on upper, 2 rewards in upper
    env.rectangle_maps.append(Map([Obstacle(320, 500, 280, 20), Obstacle(960, 500, 280, 20)],
                              [], [200, 450] if random.random() > 0.5 else [1080, 450],
                              [[random.randint(100, 600), random.randint(300, 450)],
                               [random.randint(700, 1100), random.randint(300, 450)]]))
    env.rectangle_maps[-1].is_terminal = lambda rectangle_pos, circle_pos, rewards: rectangle_pos[1] > 660
    
    # horizontal floor, narrow gap in middle, rect on upper, 1 reward in lower
    env.rectangle_maps.append(Map([Obstacle(320, 500, 280, 20), Obstacle(960, 500, 280, 20)],
                              [], [200, 450] if random.random() > 0.5 else [1080, 450],
                              [[random.randint(700, 1100), random.randint(600, 750)]]))

    # 2 horizontal floors, narrow gap in right, then in left
    env.rectangle_maps.append(Map([Obstacle(200, 500, 160, 20), Obstacle(840, 500, 400, 20),
                               Obstacle(440, 260, 400, 20), Obstacle(1080, 260, 160, 20)],
                              [], [200, 160] if random.random() > 0.5 else [1080, 160],
                              [[1000, 100] if random.random() > 0.5 else [200, 100],
                               [1000, 380] if random.random() > 0.5 else [200, 380],
                               [1000, 600] if random.random() > 0.5 else [200, 600]]))
    env.rectangle_maps[-1].is_terminal = lambda rectangle_pos, circle_pos, rewards: \
        rectangle_pos[1] > 660 and len(rewards) > 0 and rewards[-1][1] < 500

    # 2 horizontal floors, narrow gap in left, then in right
    env.rectangle_maps.append(Map([Obstacle(200, 260, 160, 20), Obstacle(840, 260, 400, 20),
                                   Obstacle(440, 500, 400, 20), Obstacle(1080, 500, 160, 20)],
                                  [], [200, 160] if random.random() > 0.5 else [1080, 160],
                                  [[1000, 100] if random.random() > 0.5 else [200, 100],
                                   [1000, 380] if random.random() > 0.5 else [200, 380],
                                   [1000, 600] if random.random() > 0.5 else [200, 600]]))
    env.rectangle_maps[-1].is_terminal = lambda rectangle_pos, circle_pos, rewards: \
        rectangle_pos[1] > 660 and len(rewards) > 0 and rewards[-1][1] < 500

    # Simple map, agent on right, reward on left 
    env.rectangle_maps.append(Map([],
                              [], [random.randint(700, 1000), 700],
                              [[random.randint(100, 600), random.randint(550, 700)]]))

    # Simple map, agent on left, reward on right
    env.rectangle_maps.append(Map([],
                              [], [random.randint(280, 600), 700],
                              [[random.randint(700, 1150), random.randint(550, 700)]]))


def both_maps(env):
    env.mixed_maps = []

    # Simple map, agents on left, reward on right
    env.mixed_maps.append(Map([],
                              [random.randint(700, 1150), 700], [random.randint(700, 1000), 700],
                              [[random.randint(100, 600), random.randint(300, 700)]]))

    # Simple map, agents on right, reward on left
    env.mixed_maps.append(Map([],
                              [random.randint(100, 600), 700], [random.randint(280, 600), 700],
                              [[random.randint(700, 1150), random.randint(300, 700)]]))
    
    # The Two Towers
    env.mixed_maps.append(Map([Obstacle(400, 570, 40, 190), Obstacle(800, 650, 40, 150)],
                              [random.randint(100, 200), 700], [random.randint(200, 300), 700],
                              [[random.randint(500, 700), random.randint(350, 700)],
                               [random.randint(1000, 1100), random.randint(350, 700)],
                               [800, random.randint(150, 300)]]))

    # split map, circle can only access top corners, rectangle only bottom corners, 1 reward each
    env.mixed_maps.append(Map([Obstacle(400, 600, 40, 110), Obstacle(880, 600, 40, 110),
                               Obstacle(200, 510, 200, 20), Obstacle(1080, 510, 200, 20)],
                              [random.randint(500, 700), 700], [random.randint(500, 700), 700],
                              [[random.randint(100, 300), random.randint(600, 700)] if random.random() > 0.5 else
                               [random.randint(980, 1180), random.randint(600, 700)],
                               [random.randint(100, 300), random.randint(200, 400)] if random.random() > 0.5 else
                               [random.randint(980, 1180), random.randint(200, 400)]]))

    # The Dome, circle can only get his reward with rectangle's help, 1 reward each
    env.mixed_maps.append(Map([Obstacle(400, 520, 40, 180), Obstacle(880, 520, 40, 180),
                               Obstacle(640, 360, 200, 20)],
                              [random.randint(100, 300), 700], [random.randint(900, 1100), 700],
                              [[random.randint(400, 880), random.randint(600, 700)],
                               [random.randint(500, 780), random.randint(200, 400)]]))


def fill_maps(env):
    circle_maps(env)
    rect_maps(env)
    both_maps(env)
