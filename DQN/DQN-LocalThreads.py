# While training is taking place, statistics on agent performance are available from Tensorboard. To launch it use:
# 
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'
#   tensorboard --logdir=worker_0:'./train_0'
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3',worker_4:'./train_4',worker_5:'./train_5',worker_6:'./train_6',worker_7:'./train_7',worker_8:'./train_8',worker_9:'./train_9',worker_10:'./train_10',worker_11:'./train_11'


import argparse
import os
import threading
from time import sleep
import tensorflow as tf
from DQN.DQNSlave import WorkerGF2
from DQN.DQNetwork import QNetwork1Step
from simulator.GymEnvGF import GymEnvGF

max_episode_length = 4000
gamma = .99  # discount rate for advantage estimation and reward discounting
state_size_square = 9
state_size_circle = 11
learning_rate = 1e-5
action_size_square = 4
action_size_circle = 4

load_model = False
model_path = './model'
use_lstm = False
use_conv_layers = False
display = True

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--num_slaves",
    type=int,
    default=1,
    help="Set number of available CPU threads"
)
parser.add_argument(
    "--alg",
    type=int,
    default=0,
    help="Which algorithm it is"
)
parser.add_argument(
    "--learning",
    type=int,
    default=5,
    help="0 no one learning; 1 square learning; 2 circle learning; 3 square and circle learning"
)
FLAGS, unparsed = parser.parse_known_args()

if FLAGS.learning == 0:  # load model, dont train further
    load_model = True
    circle_learning = False
    square_learning = False
elif FLAGS.learning == 1:  # load model, train circle
    load_model = True
    circle_learning = True
    square_learning = False
elif FLAGS.learning == 2:  # load model, train squares
    load_model = True
    circle_learning = False
    square_learning = True
elif FLAGS.learning == 3:  # dont load model, train both
    load_model = False
    circle_learning = True
    square_learning = True
elif FLAGS.learning == 4:  # dont load model, train square (rectangle)
    load_model = False
    circle_learning = False
    square_learning = True
elif FLAGS.learning == 5:  # dont load model, train circle (circle)
    load_model = False
    circle_learning = True
    square_learning = False

tf.reset_default_graph()

# Create a directory to save models and episode playback gifs
if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer_square = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainer_circle = tf.train.AdamOptimizer(learning_rate=learning_rate)
    master_network_square = QNetwork1Step(state_size_square, action_size_square, 'global_square',
                                            None, use_conv_layers, use_lstm)  # Generate global network
    master_network_circle = QNetwork1Step(state_size_circle, action_size_circle, 'global_circle',
                                        None, use_conv_layers, use_lstm)  # Generate global network
    workers = []
    # Create worker classes
    for i in range(FLAGS.num_slaves):
        workers.append(WorkerGF2(GymEnvGF(rectangle=square_learning, circle=circle_learning),
                                 i, state_size_square, state_size_circle, action_size_square, action_size_circle,
                                 trainer_square, trainer_circle, model_path,
                                 global_episodes, use_lstm, use_conv_layers, display,
                                 rectangle_learning=square_learning, circle_learning=circle_learning))

    saver = tf.train.Saver()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=worker_work)
        t.start()
        sleep(0.5)

        worker_threads.append(t)

    coord.join(worker_threads)

print("Done")
