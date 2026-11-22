#Command line for illusion_gen, the restructured version of gen.py.
#Usage (any Python works to start it; it relaunches itself with the project's .venv):
#    python generate.py [STEPS] --prompts "PROMPT A" "PROMPT B"     Start a new run (default 5000 steps)
#    python generate.py [STEPS] --illusion flip --prompts ...        Another illusion type (twisting_squares, flip, rotate)
#    python generate.py [STEPS] --resume RUN_FOLDER                  Continue a stopped run for STEPS more steps
#    python generate.py --config runs.yaml                           Go through a series of runs from a config file
#    python generate.py --config runs.yaml --dry-run                 Check the config and list its runs without running them
#    python generate.py --help                                       All options
#Every option is also a setting in config files and a generate() keyword argument (with _ instead of -), see illusion_gen/config.py.
#Results are saved under untracked/<illusion>_runs/

import argparse
import sys

from illusion_gen.env import ensure_venv, fix_console_encoding

ensure_venv(__file__)
fix_console_encoding()

from illusion_gen.config import ConfigError, RunConfig
from illusion_gen.illusions import ILLUSIONS


def on_off(value):
    lowered = value.lower()
    if lowered in ('auto',):
        return 'auto'
    if lowered in ('on', 'true', 'yes', '1'):
        return True
    if lowered in ('off', 'false', 'no', '0'):
        return False
    raise argparse.ArgumentTypeError("expected auto, on or off, not %r" % value)


def make_parser():
    #Options default to SUPPRESS, so only the ones actually given show up (and can override config files)
    parser = argparse.ArgumentParser(
        description='Generate diffusion illusions: one image that shows a different prompt when viewed another way.',
        argument_default=argparse.SUPPRESS,
        epilog='With --config, any other options given apply to every run in the file, overriding its settings '
               '(handy for a quick test, e.g. --steps 20).')

    batch = parser.add_argument_group('batch runs')
    batch.add_argument('--config', metavar='FILE', help='YAML file listing runs to go through in order (see runs.example.yaml).')
    batch.add_argument('--dry-run', action='store_true', help='Check the settings and list the runs, without loading the model or training.')

    what = parser.add_argument_group('what to make')
    what.add_argument('steps_arg', nargs='?', type=int, default=None, metavar='STEPS', help='Training steps to run (when resuming: how many MORE steps). Default 5000.')
    what.add_argument('--steps', type=int, dest='steps', help='Same as the STEPS argument.')
    what.add_argument('--prompts', nargs='+', metavar='PROMPT', help='One prompt per view: image_a, image_b, ... (twisting_squares and flip take 2, rotate takes 4). '
                      'Put "---" in a prompt to add to the negative prompt for that view only.')
    what.add_argument('--illusion', choices=list(ILLUSIONS), help='Illusion type (default twisting_squares).')
    what.add_argument('--squares', type=int, help='twisting_squares: number of squares per side (default 5).')
    what.add_argument('--negative-prompt', help="What you DON'T want to see (default 'blurry ugly').")

    training = parser.add_argument_group('training')
    training.add_argument('--weights', nargs='+', type=float, metavar='W', help='Relative weight of each prompt (default: twisting_squares 1 0.9, others equal).')
    training.add_argument('--guidance-scale', type=float, help='Classifier-free guidance scale (default 100).')
    training.add_argument('--noise-coef', type=float, help='Score distillation strength (default 0.1).')
    training.add_argument('--lr', type=float, help='SGD learning rate (default 1e-4).')
    training.add_argument('--max-step', type=int, help='Highest noise level used in training, out of 1000 (default 990). High noise levels let Stable Diffusion '
                          'repaint the overall layout; lower values (e.g. 500-700) keep the composition of an --init-image and only refine it.')
    training.add_argument('--min-step', type=int, help='Lowest noise level used in training (default 10).')
    training.add_argument('--seed', type=int, help='Random seed, for repeatable runs.')

    start = parser.add_argument_group('starting point')
    start.add_argument('--resume', metavar='RUN_FOLDER', help='Continue a stopped run in this folder. Exact if it has checkpoint.pt (whose prompts and illusion are used); '
                       'for older runs without one, it first retrains the network to reproduce the run\'s image_a.png (close, but not identical).')
    start.add_argument('--init-image', metavar='PICTURE', help='Retrain the network to reproduce this picture before training, e.g. an image_a.png you edited. '
                       'Combine with --resume to continue that run from the edited picture; on its own it starts a new run.')
    start.add_argument('--init-view', metavar='LETTER', help='Which view the --init-image picture shows (default a; b = an edited image_b.png).')
    start.add_argument('--fit-steps', type=int, help='Steps used to retrain the network to reproduce a picture (default 1000).')

    target = parser.add_argument_group('target picture')
    target.add_argument('--target-image', metavar='PICTURE', help='Train one view to look like this picture instead of its prompt. The other views still follow their prompts. '
                        'A new run starts from the picture. Since all views share the same pixels, the match is approximate.')
    target.add_argument('--target-view', metavar='LETTER', help='Which view the target picture is for (default a).')
    target.add_argument('--target-weight', type=float, help='How strongly the view is pulled toward the target picture (default 1). Higher = closer match, less room for the other views.')
    target.add_argument('--target-blur', type=float, help='Blur radius in pixels (of 256) applied to both the view and the picture before comparing them (default 4). 0 = match every pixel.')

    output = parser.add_argument_group('output')
    output.add_argument('--output-root', metavar='FOLDER', help='Runs are saved in FOLDER/<illusion>_runs/ (default untracked).')
    output.add_argument('--name', help='Run folder name (default: the prompts joined with "-"). An ISO 8601 timestamp is added if it already exists.')
    output.add_argument('--save-interval', type=int, help='Save a progress frame and checkpoint every this many steps (default 50).')

    model = parser.add_argument_group('model and GPU memory')
    model.add_argument('--model', help='Stable Diffusion 1.5 model or dreambooth of it (default stable-diffusion-v1-5/stable-diffusion-v1-5).')
    model.add_argument('--full-precision', dest='half_precision', action='store_const', const=False,
                       help='Run the UNet and VAE in 32-bit instead of 16-bit (needs ~10 GB of free GPU memory).')
    model.add_argument('--save-memory', type=on_off, metavar='{auto,on,off}', help='VAE gradient checkpointing: less GPU memory, slower steps (default auto).')
    return parser


def main():
    args = vars(make_parser().parse_args())
    config_path = args.pop('config', None)
    dry_run = args.pop('dry_run', False)
    steps_arg = args.pop('steps_arg')
    if steps_arg is not None:
        args['steps'] = steps_arg
    overrides = args #Everything left is a RunConfig setting

    try:
        if config_path is not None:
            from illusion_gen.batch import load_batch
            configs = load_batch(config_path, overrides)
        else:
            config = RunConfig.from_dict(overrides, where='command line')
            config.check('command line')
            configs = [config]
    except ConfigError as error:
        sys.exit('Error: %s' % error)

    if config_path is not None or dry_run:
        print('%i run(s)%s:' % (len(configs), ' from ' + config_path if config_path else ''))
        for number, config in enumerate(configs, 1):
            print('  %i. %s' % (number, config.describe()))
    if dry_run:
        return

    try:
        if config_path is not None:
            from illusion_gen.batch import run_batch
            ok = run_batch(configs)
            sys.exit(0 if ok else 1)
        else:
            from illusion_gen.runner import run
            run(configs[0])
    except ConfigError as error:
        sys.exit('Error: %s' % error)


if __name__ == '__main__':
    main()
