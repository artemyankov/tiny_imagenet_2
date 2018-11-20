import tensorflow as tf
import os
import json
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from clusterone import get_data_path, get_logs_path
from model_def import make_model_def
import tiny_imagenet_data_utils as data_utils

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags

def parse_args():
    """Parse arguments"""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='''Train a convolution neural network with MNIST dataset.
                            For distributed mode, the script will use few environment variables as defaults:
                            JOB_NAME, TASK_INDEX, PS_HOSTS, and WORKER_HOSTS. These environment variables will be
                            available on distributed Tensorflow jobs on Clusterone platform by default.
                            If running this locally, you will need to set these environment variables
                            or pass them in as arguments (i.e. python mnist.py --job_name worker --task_index 0
                            --worker_hosts "localhost:2222,localhost:2223" --ps_hosts "localhost:2224").
                            If these are not set, the script will run in non-distributed (single instance) mode.''')

    # Configuration for distributed task
    parser.add_argument('--job_name', type=str, default=os.environ.get('JOB_NAME', None), choices=['worker', 'ps'],
                        help='Task type for the node in the distributed cluster. Worker-0 will be set as master.')
    parser.add_argument('--task_index', type=int, default=os.environ.get('TASK_INDEX', 0),
                        help='Worker task index, should be >= 0. task_index=0 is the chief worker.')
    parser.add_argument('--ps_hosts', type=str, default=os.environ.get('PS_HOSTS', ''),
                        help='Comma-separated list of hostname:port pairs.')
    parser.add_argument('--worker_hosts', type=str, default=os.environ.get('WORKER_HOSTS', ''),
                        help='Comma-separated list of hostname:port pairs.')

    # Parse args
    opts = parser.parse_args()

    if opts.worker_hosts:
        opts.worker_hosts = opts.worker_hosts.split(',')
    else:
        opts.worker_hosts = []

    if opts.ps_hosts:
        opts.ps_hosts = opts.ps_hosts.split(',')
    else:
        opts.ps_hosts = []

    return opts

def make_tf_config(opts):
    """Returns TF_CONFIG that can be used to set the environment variable necessary for distributed training"""
    if all([opts.job_name is None, not opts.ps_hosts, not opts.worker_hosts]):
        return {}
    elif any([opts.job_name is None, not opts.ps_hosts, not opts.worker_hosts]):
        tf.logging.warn('Distributed setting is incomplete. You must pass job_name, ps_hosts, and worker_hosts.')
        if opts.job_name is None:
            tf.logging.warn('Expected job_name of worker or ps. Received {}.'.format(opts.job_name))
        if not opts.ps_hosts:
            tf.logging.warn('Expected ps_hosts, list of hostname:port pairs. Got {}. '.format(opts.ps_hosts) +
                            'Example: --ps_hosts "localhost:2224" or --ps_hosts "localhost:2224,localhost:2225')
        if not opts.worker_hosts:
            tf.logging.warn('Expected worker_hosts, list of hostname:port pairs. Got {}. '.format(opts.worker_hosts) +
                            'Example: --worker_hosts "localhost:2222,localhost:2223"')
        tf.logging.warn('Ignoring distributed arguments. Running single mode.')
        return {}

    tf_config = {
        'task': {
            'type': opts.job_name,
            'index': opts.task_index
        },
        'cluster': {
            'master': [opts.worker_hosts[0]],
            'worker': opts.worker_hosts,
            'ps': opts.ps_hosts
        },
        'environment': 'cloud'
    }

    # Nodes may need to refer to itself as localhost
    local_ip = 'localhost:' + tf_config['cluster'][opts.job_name][opts.task_index].split(':')[1]
    tf_config['cluster'][opts.job_name][opts.task_index] = local_ip
    if opts.job_name == 'worker' and opts.task_index == 0:
        tf_config['task']['type'] = 'master'
        tf_config['cluster']['master'][0] = local_ip
    return tf_config



#
# Snippet for distributed learning
#
#try:
#    task_type = os.environ['JOB_NAME']
#    task_index = int(os.environ['TASK_INDEX'])
#    ps_hosts = os.environ['PS_HOSTS'].split(',')
#    worker_hosts = os.environ['WORKER_HOSTS'].split(',')
#    TF_CONFIG = {
#        'task': {'type': task_type, 'index': task_index},
#        'cluster': {
#            'chief': [worker_hosts[0]],
#            'worker': worker_hosts,
#            'ps': ps_hosts
#        },
#        'environment': 'cloud'
#    }
#
#    local_ip = 'localhost:' + TF_CONFIG['cluster'][task_type][task_index].split(':')[1]
#    TF_CONFIG['cluster'][task_type][task_index] = local_ip
#    if (task_type in ('chief', 'master')) or (task_type == 'worker' and task_index == 0):
#        TF_CONFIG['cluster']['worker'][task_index] = local_ip
#        TF_CONFIG['task']['type'] = 'chief'
#
#    os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)
#except KeyError as ex:
#    print(ex)
#    job_name = None
#    task_index = 0
#    ps_hosts = None
#    worker_hosts = None
#
#
#

# Training related flags
flags.DEFINE_integer("N_CLASSES",
                     200,
                     "Number of prediction classes in the model")
flags.DEFINE_string("train_data_dir",
                    get_data_path(
                        dataset_name = 'artem-towa/artem-tiny-imagenet-example',
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
                        dataset_name = 'artem-towa/artem-tiny-imagenet-example',
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
                        os.path.expanduser('~/Documents/Scratch/tiny_imagenet_2/logs/')
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
        batch_size=32
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
        keep_checkpoint_max=1,
        log_step_count_steps=50
    )

    classifier = tf.keras.estimator.model_to_estimator(
        model,
        model_dir=FLAGS.log_dir,
        config=config
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input,
        max_steps=1e6
      #  hooks=[TimeHistory()]
    )
    val_spec = tf.estimator.EvalSpec(
        input_fn=val_input,
        steps=None,
        start_delay_secs=600,
        throttle_secs=120
    )

    tf.estimator.train_and_evaluate(classifier, train_spec, val_spec)

if __name__ == '__main__':
    args = parse_args()

    TF_CONFIG = make_tf_config(args)
    tf.logging.debug('=' * 20 + ' TF_CONFIG ' + '=' * 20)
    tf.logging.debug(TF_CONFIG)
    os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)

    main(args)
