"""
Argument parser for TD3.
"""

import argparse
import os
import time


def arg_parse():
    parser = argparse.ArgumentParser()

    # Experiment / logging 
    parser.add_argument('--env_name', type=str, default='Pendulum-v1',
                        help='TD3 requires a continuous action space')
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    parser.add_argument('--logroot', type=str, default='runs',
                        help='Root folder where each run\'s logs and models '
                             'are saved (relative to the code folder, '
                             'or absolute path).')

    # Training schedule 
    parser.add_argument('--max_timesteps', type=int, default=int(1e6))
    parser.add_argument('--start_timesteps', type=int, default=25000,
                        help='Random actions before learning starts')
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--eval_episodes', type=int, default=10)

    # TD3 hyperparameters (paper defaults) 
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--policy_noise', type=float, default=0.2,
                        help='Std of target smoothing noise (scaled by max_action)')
    parser.add_argument('--noise_clip', type=float, default=0.5,
                        help='Clip range for target policy noise (scaled by max_action)')
    parser.add_argument('--policy_freq', type=int, default=2,
                        help='Delayed policy update frequency d (paper: 2)')
    parser.add_argument('--expl_noise', type=float, default=0.1)

    # Network architecture 
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=256)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)

    # Replay buffer 
    parser.add_argument('--buffer_size', type=int, default=int(1e6))

    # Rendering 
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during training')
    parser.add_argument('--render_freq', type=int, default=1,
                        help='Render every N-th episode when --render is set')

    # Saving / loading / plotting 
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--no_save_model', dest='save_model', action='store_false')

    parser.add_argument('--load_model', type=str, default='',
                        help='Path prefix to load a pre-trained model from')

    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no_plot', dest='plot', action='store_false')
    parser.add_argument('--show_plot', action='store_true', default=True,
                        help='Pop up plot window at end of training (interactive). '
                             'Disabled automatically on headless machines.')
    parser.add_argument('--no_show_plot', dest='show_plot', action='store_false')

    parser.add_argument('--scalar_log_freq', type=int, default=5000)
    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()
    params = vars(args)


    code_dir = os.path.dirname(os.path.realpath(__file__))
    logroot = params['logroot']
    if not os.path.isabs(logroot):
        logroot = os.path.join(code_dir, logroot)
    os.makedirs(logroot, exist_ok=True)

    run_name = (args.exp_name + '_' + args.env_name + '_' +
                time.strftime("%d-%m-%Y_%H-%M-%S"))
    logdir = os.path.join(logroot, run_name)
    os.makedirs(logdir, exist_ok=True)

    params['logdir'] = os.path.abspath(logdir)

    return params