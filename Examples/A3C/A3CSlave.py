from time import time
from A3C.A3CNetwork import AC_Network
from Helper import *
import numpy as np


# Worker class
class WorkerGeoFriends:
    def __init__(self, game, name, size_rect, size_circ, a_size, a_size_circle,
                 trainer_rectangle, trainer_circle,
                 model_path, global_episodes, use_lstm=False, use_conv_layers=False, display=False,
                 rectangle_learning=False, circle_learning=True):
        self.name = "worker_" + str(name)
        self.is_chief = self.name == 'worker_0'
        print(self.name)
        self.number = name
        self.model_path = model_path
        self.trainer_rectangle = trainer_rectangle
        self.trainer_circle = trainer_circle
        self.global_episodes = global_episodes

        self.episode_rewards_rectangle = []
        self.episode_rewards_circle = []
        self.episode_lengths = []
        self.episode_mean_values_rectangle = []
        self.episode_mean_values_circle = []

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC_rectangle = AC_Network(size_rect, a_size, self.name + "_square", trainer_rectangle,
                                             use_conv_layers, use_lstm)
        self.local_AC_circle = AC_Network(size_circ, a_size_circle, self.name + "_circle", trainer_circle,
                                          use_conv_layers, use_lstm)
        self.update_local_ops_rectangle = update_target_graph('global_square', self.name + "_square")
        self.update_local_ops_circle = update_target_graph('global_circle', self.name + "_circle")

        # Env Pursuit set-up
        self.env = game

        self.s_size_rect = size_rect
        self.s_size_circ = size_circ

        self.display = display
        self.use_lstm = use_lstm
        self.use_conv_layers = use_conv_layers

        self.rectangle_learning = rectangle_learning
        self.circle_learning = circle_learning

        self.trainer = False
        self.htg = True
        self.trainer_episodes_limit = 15000  # trainer slowly turns off after some games

    # Take an action using probabilities from policy network output.
    def take_action_from_network(self, sess, network, previous_screen, action_indexes,
                                 action_distribution, value, rnn_state):
        if self.use_lstm:
            action_distribution, value, rnn_state[0] = sess.run(
                [network.policy, network.value, network.state_out],
                feed_dict={network.inputs: previous_screen,
                           network.state_in[0]: rnn_state[0][0],
                           network.state_in[1]: rnn_state[0][1]})
        else:
            action_distribution, value = sess.run([network.policy, network.value],
                                                  feed_dict={network.inputs: previous_screen})

        action = np.random.choice(action_indexes, p=action_distribution[0])

        return action, action_distribution, value, rnn_state

    def work(self, max_episode_length, gamma, sess, coord=None, saver=None):
        print("Starting worker " + str(self.number))

        if coord is None:
            coord = sess

        action_indexes_rectangle = list(range(self.env.action_space.spaces[0].n))
        action_indexes_circle = list(range(self.env.action_space.spaces[1].n))

        # print("Copying global networks to local networks")
        if self.rectangle_learning:
            sess.run(self.update_local_ops_rectangle)
        if self.circle_learning:
            sess.run(self.update_local_ops_circle)

        while not coord.should_stop():
            # start new epi
            (obs_rect, obs_circ) = self.env.reset()

            random_actions = self.env.action_space.sample()
            actions = [random_actions[0], random_actions[1]]

            # Initial state; initialize values for the LSTM network
            if self.rectangle_learning:
                previous_screen_rectangle = rectangle_output(obs_rect, self.env.obstacles, self.env.rectangle_ground,
                                                             self.s_size_rect)

                action_rectangle, action_distribution_rectangle, value_rectangle, rnn_state_rectangle = \
                    self.take_action_from_network(sess, self.local_AC_rectangle, previous_screen_rectangle,
                                                  action_indexes_rectangle,
                                                  [None], [None], [self.local_AC_rectangle.state_init])

            if self.circle_learning:
                previous_screen_circle = circle_output(obs_circ, self.env.obstacles, self.env.circle_ground,
                                                       self.s_size_circ)

                action_circle, action_distribution_circle, value_circle, rnn_state_circle = \
                    self.take_action_from_network(sess, self.local_AC_circle, previous_screen_circle,
                                                  action_indexes_circle,
                                                  [None], [None], [self.local_AC_circle.state_init])

            for episode_step_count in range(max_episode_length):
                if self.display:
                    self.env.render()

                # Watch environment
                (obs_rect, obs_circ), reward, terminal, info = self.env.step(actions)

                # Move
                random_actions = self.env.action_space.sample()
                actions = [random_actions[0], random_actions[1]]

                if self.rectangle_learning:
                    current_screen_rectangle = rectangle_output(obs_rect, self.env.obstacles, self.env.rectangle_ground,
                                                                self.s_size_rect)

                    actions[0], action_distribution_rectangle, value_rectangle, rnn_state_rectangle = \
                        self.take_action_from_network(sess, self.local_AC_rectangle,
                                                      current_screen_rectangle, action_indexes_rectangle,
                                                      action_distribution_rectangle, value_rectangle,
                                                      rnn_state_rectangle)

                if self.circle_learning:
                    current_screen_circle = circle_output(obs_circ, self.env.obstacles, self.env.circle_ground,
                                                          self.s_size_circ)

                    actions[1], action_distribution_circle, value_circle, rnn_state_circle = \
                        self.take_action_from_network(sess, self.local_AC_circle,
                                                      current_screen_circle, action_indexes_circle,
                                                      action_distribution_circle, value_circle, rnn_state_circle)

                if terminal:
                    break

        self.env.close()
