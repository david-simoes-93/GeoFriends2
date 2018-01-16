# While training is taking place, statistics on agent performance are available from Tensorboard. To launch it use:
# 
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'
#   tensorboard --logdir=worker_0:'./train_0'
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3',worker_4:'./train_4',worker_5:'./train_5',worker_6:'./train_6',worker_7:'./train_7',worker_8:'./train_8',worker_9:'./train_9',worker_10:'./train_10',worker_11:'./train_11'


import argparse
import os
import tensorflow as tf
from DQN.DQNetwork import QNetwork1Step
from Examples.DQN.DQNSlave import WorkerGF2
from simulator.GymEnvGF import GymEnvGF

max_episode_length = 4000
gamma = .99  # discount rate for advantage estimation and reward discounting
state_size_square = 9
state_size_circle = 11
height = 1
number_of_cell_types = 1
learning_rate = 1e-5
action_size_square = 4
action_size_circle = 4

model_path = './model_dist'
use_lstm = False
use_conv_layers = False
display = True

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--task_index",
    type=int,
    default=0,
    help="Index of task within the job"
)
parser.add_argument(
    "--slaves_per_url",
    type=str,
    default="1",
    help="Comma-separated list of maximum tasks within the job"
)
parser.add_argument(
    "--urls",
    type=str,
    default="localhost",
    help="Comma-separated list of hostnames"
)

FLAGS, unparsed = parser.parse_known_args()
circle_learning = False
square_learning = True

# Create a cluster from the parameter server and worker hosts.
hosts = []
for (url, max_per_url) in zip(FLAGS.urls.split(","), FLAGS.slaves_per_url.split(",")):
    for i in range(int(max_per_url)):
        hosts.append(url + ":" + str(2210 + i))
cluster = tf.train.ClusterSpec({"dqn": hosts})
server = tf.train.Server(cluster, job_name="dqn", task_index=FLAGS.task_index)

tf.reset_default_graph()

# Create a directory to save models
if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device(tf.train.replica_device_setter(worker_device="/job:dqn/task:%d" % FLAGS.task_index, cluster=cluster)):
    global_episodes = tf.contrib.framework.get_or_create_global_step()
    trainer_square = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainer_circle = tf.train.AdamOptimizer(learning_rate=learning_rate)

    master_network_square = QNetwork1Step(state_size_square, action_size_square, 'global_square',
                                            None, use_conv_layers, use_lstm)  # Generate global network
    master_network_circle = QNetwork1Step(state_size_circle, action_size_circle, 'global_circle',
                                        None, use_conv_layers, use_lstm)  # Generate global network

    # Master declares worker for all slaves
    for i in range(len(hosts)):
        print("Initializing variables for slave ", i)
        if i == FLAGS.task_index:
            worker = WorkerGF2(GymEnvGF(rectangle=square_learning, circle=circle_learning),
                               i, state_size_square, state_size_circle, action_size_square, action_size_circle,
                               trainer_square, trainer_circle, model_path,
                               global_episodes, use_lstm, use_conv_layers, display,
                               rectangle_learning=square_learning, circle_learning=circle_learning)
        else:
            WorkerGF2(None,
                      i, state_size_square, state_size_circle, action_size_square, action_size_circle,
                      trainer_square, trainer_circle, model_path,
                      global_episodes, use_lstm, use_conv_layers, False,
                      rectangle_learning=square_learning, circle_learning=circle_learning)

print("Starting session", server.target, FLAGS.task_index)
hooks = [tf.train.StopAtStepHook(last_step=100000)]
with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0),
                                       config=tf.ConfigProto(),  # config=tf.ConfigProto(log_device_placement=True),
                                       save_summaries_steps=100,
                                       save_checkpoint_secs=600, checkpoint_dir=model_path, hooks=hooks) as mon_sess:
    print("Started session")
    try:
        worker.work(max_episode_length, gamma, mon_sess)
    except RuntimeError:
        print("Puff")

print("Done")
