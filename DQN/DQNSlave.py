from time import time
import tensorflow as tf
from DQN.DQNetwork import QNetwork1Step
from Helper import *
import numpy as np


# Worker class
class WorkerGF2:
    def __init__(self, game, name, s_size, s_size_circle, a_size, a_size_circle,
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

        with tf.variable_scope(self.name):
            self.increment = self.global_episodes.assign_add(1)
            self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC_rectangle = QNetwork1Step(s_size, a_size, self.name + "_square", trainer_rectangle,
                                                use_conv_layers, use_lstm)
        self.local_AC_circle = QNetwork1Step(s_size_circle, a_size_circle, self.name + "_circle", trainer_circle,
                                             use_conv_layers, use_lstm)
        self.update_local_ops_rectangle = update_target_graph('global_square', self.name + "_square")
        self.update_local_ops_circle = update_target_graph('global_circle', self.name + "_circle")

        self.env = game

        self.s_size_rect = s_size
        self.s_size_circ = s_size_circle

        self.display = display
        self.use_lstm = use_lstm
        self.use_conv_layers = use_conv_layers

        self.circle_learning = circle_learning
        self.rectangle_learning = rectangle_learning

        self.exploration_rate_rectangle = 1.0
        self.exploration_rate_circle = 1.0

        self.htg = True
        self.trainer = False
        self.trainer_episodes_limit = 15000

        self.n_step_qlearning = True

    def train(self, rollout, sess, gamma, ac_network, bootstrap_value=0, terminal=False):
        # print(self.number, "Train")

        rollout = np.array(rollout)
        observations = rollout[:, 0]  # state t
        actions = rollout[:, 1]  # action taken at timestep t
        rewards = rollout[:, 2]  # reward t
        # next_observations = np.vstack(rollout[:, 3])   # state t+1
        terminals = rollout[:, 4]  # whether timestep t was terminal
        next_max_q = rollout[:, 5]  # the best Q value of state t+1 as calculated by target network

        # enable n-step q-learning instead of 1-step q-learning
        if self.n_step_qlearning:
            rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
            discounted_rewards = discount(rewards_plus, gamma)[:-1]
        else:
            # we get rewards, terminals, prev_screen, next screen, and target network
            discounted_rewards = (1. - terminals) * gamma * next_max_q + rewards

        #for i in range(len(rollout)):
        #    print(observations[i],actions[i],discounted_rewards[i])

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        if self.use_lstm:
            rnn_state = ac_network.state_init
            feed_dict = {ac_network.target_q_t: discounted_rewards,
                         ac_network.inputs: np.vstack(observations),
                         ac_network.actions: actions,
                         ac_network.state_in[0]: rnn_state[0],
                         ac_network.state_in[1]: rnn_state[1]}
        else:
            feed_dict = {ac_network.target_q_t: discounted_rewards,
                         ac_network.network.inputs: np.vstack(observations),
                         ac_network.actions: actions}
        v_l, q_value, q_acted, g_n, v_n, _ = sess.run([ac_network.loss, ac_network.network.value, ac_network.q_acted,
                                                       ac_network.grad_norms,
                                                       ac_network.var_norms,
                                                       ac_network.apply_grads],
                                                      feed_dict=feed_dict)

        return v_l / len(rollout), 0, 0, g_n, v_n

    # Take an action using probabilities from policy network output.
    def take_action_from_network(self, sess, network, previous_screen, action_indexes,
                                 action_distribution, value, rnn_state):
        if self.use_lstm:
            action_distribution[0], value[0], rnn_state[0] = sess.run(
                [network.policy, network.value, network.state_out],
                feed_dict={network.inputs: [previous_screen[0]],
                           network.state_in[0]: rnn_state[0][0],
                           network.state_in[1]: rnn_state[0][1]})
        else:
            action_distribution, value = sess.run(
                [network.policy, network.value],
                feed_dict={network.inputs: previous_screen})

        action = np.random.choice(action_indexes, p=action_distribution[0])

        return action, action_distribution, value, rnn_state

    def work(self, max_episode_length, gamma, sess, coord=None, saver=None):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))

        prev_clock = time()
        if coord is None:
            coord = sess

        # Copy local online -> local target networks
        sess.run(self.local_AC_rectangle.assign_op)
        sess.run(self.local_AC_circle.assign_op)

        action_indexes_rectangle = list(range(self.env.action_space.spaces[0].n))
        action_indexes_circle = list(range(self.env.action_space.spaces[1].n))

        end_epsilon_probs = [0.01, 0.1, 0.5]
        my_end_epsilon_prob = end_epsilon_probs[self.number % len(end_epsilon_probs)]

        # print("Copying global networks to local networks")
        if self.rectangle_learning:
            sess.run(self.update_local_ops_rectangle)
        if self.circle_learning:
            sess.run(self.update_local_ops_circle)

        while not coord.should_stop():
            episode_buffer_rectangle = [[]]
            episode_values_rectangle = [[]]
            episode_buffer_circle = [[]]
            episode_values_circle = [[]]
            episode_reward_rectangle = 0
            episode_reward_circle = 0

            v_l_circle, p_l_circle, e_l_circle, g_n_circle, v_n_circle = get_empty_loss_arrays(1)
            v_l, p_l, e_l, g_n, v_n = get_empty_loss_arrays(1)

            # start new epi
            (obs_rect, obs_circ) = self.env.reset()

            random_actions = self.env.action_space.sample()
            actions = [random_actions[0], random_actions[1]]
            # Initial state; initialize values for the LSTM network
            if self.rectangle_learning:
                previous_screen_rectangle = rectangle_output(obs_rect, self.env.obstacles_rectangle, self.env.rectangle_ground,
                                                             self.s_size_rect)

                actions[0], action_distribution_rectangle, value_rectangle, rnn_state_rectangle = \
                    self.take_action_from_network(sess, self.local_AC_rectangle.network,
                                                  previous_screen_rectangle, action_indexes_rectangle,
                                                  [None], [None], [self.local_AC_rectangle.network.state_init])
            # else:
            #    actions[0] = random.randint(0, self.env.numberOfActions_rect - 1)

            if self.circle_learning:
                previous_screen_circle = circle_output(obs_circ, self.env.obstacles_circle, self.env.circle_ground,
                                                       self.s_size_circ)
                actions[1], action_distribution_circle, value_circle, rnn_state_circle = \
                    self.take_action_from_network(sess, self.local_AC_circle.network,
                                                  previous_screen_circle, action_indexes_circle,
                                                  [None], [None], [self.local_AC_circle.network.state_init])
            # else:
            #    actions[1] = random.randint(0, self.env.numberOfActions_circle - 1)

            if self.htg:
                if self.rectangle_learning:
                    previous_htg_rect = get_htg_rect(previous_screen_rectangle, self.env)
                if self.circle_learning:
                    previous_htg_circ = get_htg_circ(previous_screen_circle, self.env)

            batch_size = 25
            for episode_step_count in range(max_episode_length):
                if self.display:
                    self.env.render()

                self.exploration_rate_circle = max(my_end_epsilon_prob, 1 - (1 - my_end_epsilon_prob) * (
                    episode_count / self.trainer_episodes_limit))
                self.exploration_rate_rectangle = max(my_end_epsilon_prob, 1 - (1 - my_end_epsilon_prob) * (
                    episode_count / self.trainer_episodes_limit))

                # Watch environment
                (obs_rect, obs_circ), reward, terminal, info = self.env.step(actions)
                reward_rectangle, reward_circle = reward, reward

                if self.htg:
                    # TEMPERATURE
                    # score = (100 / max_episode_length) * max(0, 1 - episode_count/self.trainer_steps)

                    # SIMPLE/SMALL
                    score = min(100 / max_episode_length, gamma ** (max_episode_length - episode_step_count) * 100)
                    # when 830 steps missing, it will use simple method, otherwise the discounted reward (assuming gamma = .99

                    # SIMPLE
                    # score = (100 / max_episode_length)

                    if self.rectangle_learning:
                        reward_rectangle += previous_htg_rect[actions[0]] * score
                    if self.circle_learning:
                        reward_circle += previous_htg_circ[actions[1]] * score

                episode_reward_rectangle += reward_rectangle
                episode_reward_circle += reward_circle

                previous_action_rectangle = actions[0]
                previous_action_circle = actions[1]

                random_actions = self.env.action_space.sample()
                actions = [random_actions[0], random_actions[1]]

                # Move
                if self.rectangle_learning:
                    current_screen_rectangle = rectangle_output(obs_rect, self.env.obstacles_rectangle, self.env.rectangle_ground,
                                                                self.s_size_rect)

                    # trainer, repick
                    if self.trainer:
                        actions[0] = get_trainer_action_rectangle(current_screen_rectangle, self.env)
                    # random exploration
                    elif np.random.random() < self.exploration_rate_rectangle:
                        pass
                        # actions[0] = random.randint(0, self.env.numberOfActions_rect - 1)
                    # Otherwise, lets judge and execute a move
                    else:
                        actions[0], action_distribution_rectangle, value_rectangle, rnn_state_rectangle = \
                            self.take_action_from_network(sess, self.local_AC_rectangle.network,
                                                          current_screen_rectangle, action_indexes_rectangle,
                                                          action_distribution_rectangle, value_rectangle,
                                                          rnn_state_rectangle)
                # else:
                #    actions[0] = random.randint(0, self.env.numberOfActions_rect - 1)

                if self.circle_learning:
                    current_screen_circle = circle_output(obs_circ, self.env.obstacles_circle, self.env.circle_ground,
                                                          self.s_size_circ)

                    # trainer, repick
                    if self.trainer:
                        actions[1] = get_trainer_action_circle(current_screen_circle, self.env)
                    # random exploration
                    elif np.random.random() < self.exploration_rate_circle:
                        pass
                        # actions[1] = random.randint(0, self.env.numberOfActions_circle - 1)
                    # Otherwise, lets judge and execute a move
                    else:
                        actions[1], action_distribution_circle, value_circle, rnn_state_circle = \
                            self.take_action_from_network(sess, self.local_AC_circle.network,
                                                          current_screen_circle, action_indexes_circle,
                                                          action_distribution_circle, value_circle,
                                                          rnn_state_circle)
                # else:
                #    action_circle = random.randint(0, self.env.numberOfActions_circle - 1)

                if self.htg:
                    if self.rectangle_learning:
                        previous_htg_rect = get_htg_rect(current_screen_rectangle, self.env)
                    if self.circle_learning:
                        previous_htg_circ = get_htg_circ(current_screen_circle, self.env)

                # actions = (action_rectangle, action_circle)

                # Store environment
                if self.rectangle_learning:
                    # get target network values
                    if self.n_step_qlearning:
                        # Store environment
                        episode_buffer_rectangle[0].append(
                            [previous_screen_rectangle[0], previous_action_rectangle, reward_rectangle,
                             current_screen_rectangle[0], terminal, 0])
                    else:
                        next_max_q_rectangle = sess.run(self.local_AC_rectangle.target_network.best_q, feed_dict={
                            self.local_AC_rectangle.target_network.inputs: current_screen_rectangle})

                        # Store environment
                        episode_buffer_rectangle[0].append(
                            [previous_screen_rectangle[0], previous_action_rectangle, reward_rectangle,
                             current_screen_rectangle[0], terminal, next_max_q_rectangle[0]])
                    episode_values_rectangle[0].append(np.max(value_rectangle[0]))
                    previous_screen_rectangle = current_screen_rectangle

                if self.circle_learning:
                    if self.n_step_qlearning:
                        # Store environment
                        episode_buffer_circle[0].append(
                            [previous_screen_circle[0], previous_action_circle, reward_circle,
                             current_screen_circle[0], terminal, 0])
                    else:
                        # get target network values
                        next_max_q_circle = sess.run(self.local_AC_circle.target_network.best_q, feed_dict={
                            self.local_AC_circle.target_network.inputs: current_screen_circle})

                        # Store environment
                        episode_buffer_circle[0].append(
                            [previous_screen_circle[0], previous_action_circle, reward_circle,
                             current_screen_circle[0], terminal, next_max_q_circle[0]])
                    episode_values_circle[0].append(np.max(value_circle[0]))
                    previous_screen_circle = current_screen_circle

                # If the episode hasn't ended, but the experience buffer is full, then we make an update step
                # using that experience rollout.
                if self.rectangle_learning and len(episode_buffer_rectangle[0]) == batch_size and \
                        not terminal and episode_step_count < max_episode_length - 1:
                    if self.n_step_qlearning:
                        next_max_q_rectangle = sess.run(self.local_AC_rectangle.target_network.best_q, feed_dict={
                            self.local_AC_rectangle.target_network.inputs: current_screen_rectangle})
                        v_l[0], p_l[0], e_l[0], g_n[0], v_n[0] = \
                            self.train(episode_buffer_rectangle[0], sess, gamma, self.local_AC_rectangle,
                                       bootstrap_value=next_max_q_rectangle[0])
                    else:
                        v_l[0], p_l[0], e_l[0], g_n[0], v_n[0] = \
                            self.train(episode_buffer_rectangle[0], sess, gamma, self.local_AC_rectangle)
                    episode_buffer_rectangle = [[]]

                    # print("Copying global networks to local networks")
                    sess.run(self.update_local_ops_rectangle)

                # print(self.circle_learning, len(episode_buffer_circle[0]), episode_step_count )
                if self.circle_learning and len(episode_buffer_circle[0]) == batch_size and \
                        not terminal and episode_step_count < max_episode_length - 1:
                    if self.n_step_qlearning:
                        next_max_q_circle = sess.run(self.local_AC_circle.target_network.best_q, feed_dict={
                            self.local_AC_circle.target_network.inputs: current_screen_circle})
                        v_l[0], p_l[0], e_l[0], g_n[0], v_n[0] = \
                            self.train(episode_buffer_circle[0], sess, gamma, self.local_AC_circle,
                                       bootstrap_value=next_max_q_circle[0])
                    else:
                        v_l[0], p_l[0], e_l[0], g_n[0], v_n[0] = \
                            self.train(episode_buffer_circle[0], sess, gamma, self.local_AC_circle, 0)
                    episode_buffer_circle = [[]]

                    # print("Copying global networks to local networks")
                    sess.run(self.update_local_ops_circle)

                # Measure time and increase episode step count
                total_steps += 1
                if total_steps % 2000 == 0:
                    new_clock = time()
                    print(2000.0 / (new_clock - prev_clock), "it/s,   ")
                    prev_clock = new_clock

                if terminal:
                    break

            # print("0ver ",episode_count)
            self.episode_rewards_rectangle.append(episode_reward_rectangle)
            self.episode_rewards_circle.append(episode_reward_circle)

            self.episode_lengths.append(episode_step_count)

            if self.rectangle_learning:
                self.episode_mean_values_rectangle.append(np.mean(episode_values_rectangle))
            else:
                self.episode_mean_values_rectangle.append(0)
            if self.circle_learning:
                self.episode_mean_values_circle.append(np.mean(episode_values_circle))
            else:
                self.episode_mean_values_circle.append(0)

            # self.env.close()

            # Update the network using the experience buffer at the end of the episode.
            if self.rectangle_learning:
                v_l[0], p_l[0], e_l[0], g_n[0], v_n[0] = \
                    self.train(episode_buffer_rectangle[0], sess, gamma, self.local_AC_rectangle, terminal=True)
                # print("Copying global networks to local networks")
                sess.run(self.update_local_ops_rectangle)

            # Update the network using the experience buffer at the end of the episode.
            if self.circle_learning:
                v_l_circle[0], p_l_circle[0], e_l_circle[0], g_n_circle[0], v_n_circle[0] = \
                    self.train(episode_buffer_circle[0], sess, gamma, self.local_AC_circle, terminal=True)
                # print("Copying global networks to local networks")
                sess.run(self.update_local_ops_circle)

            # Copy local online -> local target networks
            if episode_count % 20 == 0:
                sess.run([self.local_AC_rectangle.assign_op, self.local_AC_circle.assign_op])

            # Periodically save gifs of episodes, model parameters, and summary statistics.
            if episode_count % 5 == 0 and episode_count != 0:
                # Save current model
                if self.is_chief and episode_count % 50 == 0 and saver is not None:  #
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    print("Saved Model")

                # Save statistics for TensorBoard
                mean_length = np.mean(self.episode_lengths[-5:])
                mean_reward_rectangle = np.mean(self.episode_rewards_rectangle[-5:])
                mean_value_rectangle = np.mean(self.episode_mean_values_rectangle[-5:])
                mean_reward_circle = np.mean(self.episode_rewards_circle[-5:])
                mean_value_circle = np.mean(self.episode_mean_values_circle[-5:])

                summary = tf.Summary()

                summary.value.add(tag='Perf/Length', simple_value=float(mean_length))  # avg episode length

                if self.rectangle_learning:
                    summary.value.add(tag='Perf/Reward rectangle',
                                      simple_value=float(mean_reward_rectangle))  # avg reward
                    summary.value.add(tag='Perf/Value rectangle',
                                      simple_value=float(mean_value_rectangle))  # avg episode value_rectangle
                    summary.value.add(tag='Losses/Value Loss rectangle', simple_value=float(np.mean(v_l)))  # value_loss
                    summary.value.add(tag='Losses/Policy Loss rectangle',
                                      simple_value=float(np.mean(p_l)))  # policy_loss
                    summary.value.add(tag='Losses/Entropy rectangle', simple_value=float(np.mean(e_l)))  # entropy
                    summary.value.add(tag='Losses/Grad Norm rectangle', simple_value=float(np.mean(g_n)))  # grad_norms
                    summary.value.add(tag='Losses/Var Norm rectangle', simple_value=float(np.mean(v_n)))  # var_norms

                if self.circle_learning:
                    summary.value.add(tag='Perf/Reward circle', simple_value=float(mean_reward_circle))  # avg reward
                    summary.value.add(tag='Perf/Value circle',
                                      simple_value=float(mean_value_circle))  # avg episode value_rectangle
                    summary.value.add(tag='Losses/Value Loss circle',
                                      simple_value=float(np.mean(v_l_circle)))  # value_loss
                    summary.value.add(tag='Losses/Policy Loss circle',
                                      simple_value=float(np.mean(p_l_circle)))  # policy_loss
                    summary.value.add(tag='Losses/Entropy circle', simple_value=float(np.mean(e_l_circle)))  # entropy
                    summary.value.add(tag='Losses/Grad Norm circle',
                                      simple_value=float(np.mean(g_n_circle)))  # grad_norms
                    summary.value.add(tag='Losses/Var Norm circle',
                                      simple_value=float(np.mean(v_n_circle)))  # var_norms

                self.summary_writer.add_summary(summary, episode_count)

                self.summary_writer.flush()

            # Update episode count
            if self.is_chief:
                episode_count = sess.run(self.increment)
                print("Global episodes @", episode_count, " epsilon =", self.exploration_rate_rectangle, "/",
                      self.exploration_rate_circle)
            else:
                episode_count = sess.run(self.global_episodes)

        self.env.close()
