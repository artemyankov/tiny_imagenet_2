import tensorflow as tf
import os
import json
import time
from clusterone import get_data_path, get_logs_path
from model_def import make_model_def
import tiny_imagenet_data_utils as data_utils

flags = tf.app.flags

#
# Snippet for distributed learning
#
try:
    task_type = os.environ['JOB_NAME']
    task_index = int(os.environ['TASK_INDEX'])
    ps_hosts = os.environ['PS_HOSTS'].split(',')
    worker_hosts = os.environ['WORKER_HOSTS'].split(',')
    TF_CONFIG = {
        'task': {'type': task_type, 'index': task_index},
        'cluster': {
            'chief': [worker_hosts[0]],
            'worker': worker_hosts,
            'ps': ps_hosts
        },
        'environment': 'cloud'
    }

    local_ip = 'localhost:' + TF_CONFIG['cluster'][task_type][task_index].split(':')[1]
    TF_CONFIG['cluster'][task_type][task_index] = local_ip
    if (task_type in ('chief', 'master')) or (task_type == 'worker' and task_index == 0):
        TF_CONFIG['cluster']['worker'][task_index] = local_ip
        TF_CONFIG['task']['type'] = 'chief'

    os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)
except KeyError as ex:
    print(ex)
    job_name = None
    task_index = 0
    ps_hosts = None
    worker_hosts = None
#
#
#

# Training related flags
flags.DEFINE_integer("N_CLASSES",
                     200,
                     "Number of prediction classes in the model")
flags.DEFINE_string("train_data_dir",
                    get_data_path(
                        dataset_name = 'artem-stable/artem-tiny-imagenet-1',
                        local_root = os.path.expanduser('~/Documents/Scratch/tiny_imagenet/'),
                        local_repo = 'tiny-imagenet-200',
                        path = 'train'
                    ),
                    "Path to store logs and checkpoints. It is recommended"
                    "to use get_logs_path() to define your logs directory."
                    "so that you can switch from local to clusterone without"
                    "changing your code."
                    "If you set your logs directory manually make sure"
                    "to use /logs/ when running on ClusterOne cloud.")
flags.DEFINE_string("val_data_dir",
                    get_data_path(
                        dataset_name = 'artem-stable/artem-tiny-imagenet-1',
                        local_root = os.path.expanduser('~/Documents/Scratch/tiny_imagenet/'),
                        local_repo = 'tiny-imagenet-200',
                        path = 'val/for_keras'
                    ),
                    "Path to store logs and checkpoints. It is recommended"
                    "to use get_logs_path() to define your logs directory."
                    "so that you can switch from local to clusterone without"
                    "changing your code."
                    "If you set your logs directory manually make sure"
                    "to use /logs/ when running on ClusterOne cloud.")
flags.DEFINE_string("log_dir",
                    get_logs_path(
                        os.path.expanduser('~/Documents/Scratch/tiny_imagenet/logs/')
                    ),
                    "Path to dataset. It is recommended to use get_data_path()"
                    "to define your data directory.so that you can switch "
                    "from local to clusterone without changing your code."
                    "If you set the data directory manually makue sure to use"
                    "/data/ as root path when running on ClusterOne cloud.")
FLAGS = flags.FLAGS

class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_time_start = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)

def main(_):
    train_input = lambda: data_utils.input_fn(
        FLAGS.train_data_dir,
        batch_size=64
    )

    val_input = lambda: data_utils.input_fn(
        FLAGS.val_data_dir,
        shuffle=False
    )

    model = make_model_def()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    config = tf.estimator.RunConfig(
        model_dir=FLAGS.log_dir,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        keep_checkpoint_max=3,
        log_step_count_steps=50
    )

    classifier = tf.keras.estimator.model_to_estimator(
        model,
        model_dir=FLAGS.log_dir,
        config=config
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input,
        max_steps=1e6,
        hooks=[TimeHistory()]
    )
    val_spec = tf.estimator.EvalSpec(
        input_fn=val_input,
        steps=None,
        start_delay_secs=60,
        throttle_secs=120
    )

    tf.estimator.train_and_evaluate(classifier, train_spec, val_spec)

if __name__ == '__main__':
    tf.app.run()
