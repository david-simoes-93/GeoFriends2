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
        with tf.variable_scope(self.name):
            self.increment = self.global_episodes.assign_add(1)
            self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

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
        self.trainer_episodes_limit = 8000  # trainer slowly turns off after some games

    def train(self, rollout, sess, gamma, bootstrap_value, ac_network):
        # prev_screen, action, reward, next_screen, terminal, value

        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        # next_observations = rollout[:, 3]
        values = rollout[:, 5]

        #for r in rollout:
        #    print([int(x) for x in r[0]],r[1],r[2],[int(x) for x in r[3]],r[4],r[5])

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(rewards_plus, gamma)[:-1]
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        if self.use_lstm:
            rnn_state = ac_network.state_init
            feed_dict = {ac_network.target_v: discounted_rewards,
                         ac_network.inputs: np.vstack(observations),
                         ac_network.actions: actions,
                         ac_network.advantages: advantages,
                         ac_network.state_in[0]: rnn_state[0],
                         ac_network.state_in[1]: rnn_state[1]}
        else:
            feed_dict = {ac_network.target_v: discounted_rewards,
                         ac_network.inputs: np.vstack(observations),
                         ac_network.actions: actions,
                         ac_network.advantages: advantages}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([ac_network.value_loss,
                                               ac_network.policy_loss,
                                               ac_network.entropy,
                                               ac_network.grad_norms,
                                               ac_network.var_norms,
                                               ac_network.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

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
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))

        prev_clock = time()
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

            if self.htg:
                if self.rectangle_learning:
                    previous_htg_rect = get_htg_rect(previous_screen_rectangle, self.env)
                if self.circle_learning:
                    previous_htg_circ = get_htg_circ(previous_screen_circle, self.env)

            batch_size = 25
            for episode_step_count in range(max_episode_length):
                if self.display and self.is_chief:
                    self.env.render()

                # Watch environment
                (obs_rect, obs_circ), reward, terminal, info = self.env.step(actions)
                reward_rectangle, reward_circle = reward, reward

                if self.htg:
                    # TEMPERATURE
                    # score = (100 / max_episode_length) * max(0, 1 - episode_count/self.trainer_steps)

                    # SIMPLE/SMALL
                    score = min(100 / max_episode_length, gamma ** (max_episode_length - episode_step_count) * 100)
                    # when 830 steps missing, it will use simple method, otherwise the discounted reward (assuming gamma = .99)

                    # SIMPLE
                    #score = (100 / max_episode_length)

                    if self.rectangle_learning:
                        reward_rectangle += previous_htg_rect[actions[0]] * score
                    if self.circle_learning:
                        reward_circle += previous_htg_circ[actions[1]] * score
                episode_reward_rectangle += reward_rectangle
                episode_reward_circle += reward_circle

                previous_action_rectangle = actions[0]
                previous_action_circle = actions[1]

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
                    # trainer, repick
                    if self.trainer:
                        actions[0] = get_trainer_action_rectangle(current_screen_rectangle, self.env)
                    #actions[0]=action_rectangle
                # else:
                #    action_rectangle = random.randint(0, self.env.numberOfActions_rect - 1)

                if self.circle_learning:
                    current_screen_circle = circle_output(obs_circ, self.env.obstacles, self.env.circle_ground,
                                                          self.s_size_circ)

                    actions[1], action_distribution_circle, value_circle, rnn_state_circle = \
                        self.take_action_from_network(sess, self.local_AC_circle,
                                                      current_screen_circle, action_indexes_circle,
                                                      action_distribution_circle, value_circle, rnn_state_circle)
                    # trainer, repick
                    if self.trainer:
                        actions[1] = get_trainer_action_circle(current_screen_circle, self.env)
                    #actions[1] = action_circle
                # else:
                #    action_circle = random.randint(0, self.env.numberOfActions_circle - 1)

                if self.htg:
                    if self.rectangle_learning:
                        previous_htg_rect = get_htg_rect(current_screen_rectangle, self.env)
                    if self.circle_learning:
                        previous_htg_circ = get_htg_circ(current_screen_circle, self.env)

                # Store environment
                if self.rectangle_learning:
                    episode_buffer_rectangle[0].append(
                        [previous_screen_rectangle[0], previous_action_rectangle, reward_rectangle,
                         current_screen_rectangle[0], terminal, value_rectangle[0][0]])
                    episode_values_rectangle[0].append(value_rectangle[0][0])
                    previous_screen_rectangle = current_screen_rectangle

                if self.circle_learning:
                    episode_buffer_circle[0].append(
                        [previous_screen_circle[0], previous_action_circle, reward_circle,
                         current_screen_circle[0], terminal, value_circle[0][0]])
                    episode_values_circle[0].append(value_circle[0][0])
                    previous_screen_circle = current_screen_circle

                # If the episode hasn't ended, but the experience buffer is full, then we make an update step
                # using that experience rollout.
                if self.rectangle_learning and len(episode_buffer_rectangle[0]) == batch_size and \
                        not terminal and episode_step_count < max_episode_length - 1:
                    # Since we don't know what the true final return is, we "bootstrap" from our current
                    # value_square estimation.
                    v_l[0], p_l[0], e_l[0], g_n[0], v_n[0] = \
                        self.train(episode_buffer_rectangle[0], sess, gamma, value_rectangle[0][0],
                                   self.local_AC_rectangle)
                    episode_buffer_rectangle = [[]]

                    # print("Copying global networks to local networks")
                    sess.run(self.update_local_ops_rectangle)

                # print(self.circle_learning, len(episode_buffer_circle[0]), episode_step_count )
                if self.circle_learning and len(episode_buffer_circle[0]) == batch_size and \
                        not terminal and episode_step_count < max_episode_length - 1:
                    # Since we don't know what the true final return is, we "bootstrap" from our current
                    # value_square estimation.
                    v_l[0], p_l[0], e_l[0], g_n[0], v_n[0] = \
                        self.train(episode_buffer_circle[0], sess, gamma, value_circle[0][0], self.local_AC_circle)
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

            # print("ended", episode_step_count)

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
                    self.train(episode_buffer_rectangle[0], sess, gamma, 0.0, self.local_AC_rectangle)
                # print("Copying global networks to local networks")
                sess.run(self.update_local_ops_rectangle)

            # Update the network using the experience buffer at the end of the episode.
            if self.circle_learning:
                v_l_circle[0], p_l_circle[0], e_l_circle[0], g_n_circle[0], v_n_circle[0] = \
                    self.train(episode_buffer_circle[0], sess, gamma, 0.0, self.local_AC_circle)
                # print("Copying global networks to local networks")
                sess.run(self.update_local_ops_circle)

            # Periodically save gifs of episodes, model parameters, and summary statistics.
            if episode_count % 5 == 0 and episode_count != 0:
                # Save current model
                if self.is_chief and episode_count % 50 == 0 and saver is not None:
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
                    summary.value.add(tag='Losses/Value Loss rectangle',
                                      simple_value=float(np.mean(v_l)))  # value_loss
                    summary.value.add(tag='Losses/Policy Loss rectangle',
                                      simple_value=float(np.mean(p_l)))  # policy_loss
                    summary.value.add(tag='Losses/Entropy rectangle',
                                      simple_value=float(np.mean(e_l)))  # entropy
                    summary.value.add(tag='Losses/Grad Norm rectangle',
                                      simple_value=float(np.mean(g_n)))  # grad_norms
                    summary.value.add(tag='Losses/Var Norm rectangle',
                                      simple_value=float(np.mean(v_n)))  # var_norms

                if self.circle_learning:
                    summary.value.add(tag='Perf/Reward circle',
                                      simple_value=float(mean_reward_circle))  # avg reward
                    summary.value.add(tag='Perf/Value circle',
                                      simple_value=float(mean_value_circle))  # avg episode value_circle
                    summary.value.add(tag='Losses/Value Loss circle',
                                      simple_value=float(np.mean(v_l_circle)))  # value_loss
                    summary.value.add(tag='Losses/Policy Loss circle',
                                      simple_value=float(np.mean(p_l_circle)))  # policy_loss
                    summary.value.add(tag='Losses/Entropy circle',
                                      simple_value=float(np.mean(e_l_circle)))  # entropy
                    summary.value.add(tag='Losses/Grad Norm circle',
                                      simple_value=float(np.mean(g_n_circle)))  # grad_norms
                    summary.value.add(tag='Losses/Var Norm circle',
                                      simple_value=float(np.mean(v_n_circle)))  # var_norms

                self.summary_writer.add_summary(summary, episode_count)

                self.summary_writer.flush()

            # Update episode count
            if self.is_chief:
                episode_count = sess.run(self.increment)
                print("Global episodes @", episode_count)
            else:
                episode_count = sess.run(self.global_episodes)
                # episode_count += 1

        self.env.close()
