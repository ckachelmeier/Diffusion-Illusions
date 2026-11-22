#One training run: generate() for calling from Python, run() for a ready-made RunConfig
import dataclasses
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import rp
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .checkpoints import CURRENT_TARGET_WEIGHT_SCALE, OLD_TARGET_WEIGHT_FACTOR, load_checkpoint, save_checkpoint
from .config import (DEFAULT_PROMPTS, DEFAULT_TARGET_BLUR, DEFAULT_TARGET_DETAIL, DEFAULT_TARGET_VIEW,
                     DEFAULT_TARGET_WEIGHT, ConfigError, RunConfig)
from .env import fix_colorama
from .illusions import ILLUSIONS, VIEW_LETTERS, Illusion, resolve_illusion, view_index
from .image_network import fit_to_picture, load_picture, load_state, make_image_network, state_for_checkpoint
from .model import encode_prompts, get_model

fix_colorama() #Also when generate() is called from your own script, not just from generate.py

#Target picture calibration: an SD step's pixel gradient is about as large as the target loss's gradient at a scale of
#6.5e5 with every pixel off by 0.1. At that strength the other view never forms its subject, so weight 1 is 3% of it.
#In 700-step tests (apple target, unicorn prompt, blur 4, detail 0.1), 3% gave a recognizable apple with visible
#unicorns in the other view; 10% gave a clean apple and no unicorn.
TARGET_LOSS_SCALE = 6.5e5 / OLD_TARGET_WEIGHT_FACTOR

MAX_RUN_NAME_LENGTH = 150 #Keeps paths under Windows' 260 character limit


@dataclass
class RunResult:
    output_folder: str
    start_step: int
    end_step: int      #Total training steps the run has after this session
    interrupted: bool  #Stopped early with Ctrl+C (the image was still saved)


def generate(prompts=None, illusion='twisting_squares', steps=5000, **options):
    """Train an illusion and save it. Returns a RunResult.

    prompts:  one prompt per view of the illusion (twisting_squares and flip have 2, rotate has 4)
    illusion: a name from illusions.ILLUSIONS, or an Illusion with your own view functions (see illusions.py)
    steps:    training steps (when resuming: how many more)
    options:  any other RunConfig setting (see config.py), e.g. negative_prompt, resume, init_image, target_image, seed
    """
    return run(RunConfig.from_dict(dict(prompts=prompts, illusion=illusion, steps=steps, **options), where='generate()'))


def timestamp():
    #ISO 8601 local time in the basic format (no colons, which Windows doesn't allow in file names), e.g. 20261121T135501
    return datetime.now().strftime('%Y%m%dT%H%M%S')


def new_output_folder(output_root, illusion, prompts, name=None):
    #Named after the prompts (or the given name), with characters Windows doesn't allow in file names removed
    run_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name if name else '-'.join(prompts))
    run_name = run_name[:MAX_RUN_NAME_LENGTH].strip(' .') or 'run'
    folder = os.path.normpath(os.path.join(output_root, illusion.output_dir_name, run_name))
    if os.path.exists(folder):
        folder = base = folder + '_' + timestamp()
        counter = 2
        while os.path.exists(folder): #Only if two runs start within the same second
            folder = '%s_%i' % (base, counter)
            counter += 1
    os.makedirs(os.path.join(folder, 'frames'))
    return folder


def run(config: RunConfig) -> RunResult:
    config.check()
    c = dataclasses.replace(config) #Work on a copy; the caller's config stays as given

    if c.seed is not None:
        random.seed(c.seed)
        np.random.seed(c.seed)
        torch.manual_seed(c.seed)

    #Settings a checkpoint can override
    prompts, negative_prompt, model_name = c.prompts, c.negative_prompt, c.model
    illusion_spec, squares = c.illusion, c.squares
    init_image, init_view = c.init_image, c.init_view
    target_image, target_view, target_weight, target_blur, target_detail = c.target_image, c.target_view, c.target_weight, c.target_blur, c.target_detail

    checkpoint = None
    if c.resume is not None:
        checkpoint = load_checkpoint(c.resume)
        if checkpoint is not None:
            if c.prompts is not None and c.prompts != checkpoint['prompts']:
                print('Note: using the prompts saved with this run, not the ones given.')
            prompts, negative_prompt = checkpoint['prompts'], checkpoint['negative_prompt']
            saved_illusion = checkpoint['illusion']
            if isinstance(illusion_spec, Illusion) and illusion_spec.name == saved_illusion:
                pass #The caller passed this run's custom illusion again
            elif saved_illusion in ILLUSIONS:
                illusion_spec = saved_illusion
                squares = checkpoint['illusion_params'].get('squares', squares)
            else:
                raise ConfigError('%r uses the custom illusion %r: pass that Illusion object as illusion to resume it' % (c.resume, saved_illusion))
            if checkpoint['model_name']:
                model_name = checkpoint['model_name']
            print('Resuming from checkpoint at step %i' % checkpoint['step'])
            saved_target = checkpoint['target']
            if saved_target is not None and target_image is None:
                #Keep using the run's target picture, unless a different one is given
                target_image = saved_target['path']
                if target_view   is None: target_view   = saved_target['view']
                if target_weight is None:
                    target_weight = saved_target['weight']
                    if 'weight_scale' not in saved_target:
                        target_weight *= OLD_TARGET_WEIGHT_FACTOR
                        print('Converted target weight %s from an older checkpoint to the current scale: %.3g' % (saved_target['weight'], target_weight))
                if target_blur   is None: target_blur   = saved_target['blur']
                if target_detail is None: target_detail = saved_target.get('detail', 0.0) #Older checkpoints had no detail term
        else:
            if init_image is None:
                init_image, init_view = os.path.join(c.resume, 'image_a.png'), 'a'
                if not os.path.exists(init_image):
                    raise ConfigError('%r has no checkpoint.pt or image_a.png to resume from' % c.resume)
            print('This run has no checkpoint, so the network will be retrained to reproduce %r (an approximate restart).' % init_image)
            print('WARNING: Older runs did not save their settings. The given prompts, illusion and squares will be used,')
            print('         so make sure they match this run!')
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    illusion = resolve_illusion(illusion_spec, squares)
    if len(prompts) != illusion.num_views:
        raise ConfigError('%s needs %i prompts (one per view), got %i' % (illusion.name, illusion.num_views, len(prompts)))
    weights = c.weights if c.weights is not None else illusion.default_weights
    if len(weights) != illusion.num_views:
        raise ConfigError('%s needs %i weights (one per view), got %i' % (illusion.name, illusion.num_views, len(weights)))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum() * len(weights)

    target_index = None
    if target_image is not None:
        if not os.path.exists(target_image):
            raise ConfigError('Target picture not found: %r' % target_image)
        target_image  = os.path.abspath(target_image) #Saved in checkpoints, so resuming works from any directory
        target_view   = target_view   if target_view   is not None else DEFAULT_TARGET_VIEW
        target_weight = target_weight if target_weight is not None else DEFAULT_TARGET_WEIGHT
        target_blur   = target_blur   if target_blur   is not None else DEFAULT_TARGET_BLUR
        target_detail = target_detail if target_detail is not None else DEFAULT_TARGET_DETAIL
        target_index  = view_index(target_view, illusion)
        if init_image is None and checkpoint is None:
            init_image, init_view = target_image, target_view #Start a new run from the target picture, rather than from noise
    if init_image is not None and not os.path.exists(init_image):
        raise ConfigError('Picture not found: %r' % init_image)
    init_index = view_index(init_view, illusion)

    print()
    print('Illusion:', illusion.name, ('(%i x %i squares)' % (squares, squares) if 'squares' in illusion.params else ''))
    print('Negative prompt:', repr(negative_prompt))
    print('Prompts:')
    for i, prompt in enumerate(prompts):
        unused = '   (not used: image_%s follows the target picture)' % VIEW_LETTERS[i] if i == target_index else ''
        print('    image_%s = %r%s' % (VIEW_LETTERS[i], prompt, unused))
    if target_image is not None:
        print('Target picture for image_%s: %r  (weight %s, blur %s px, detail %s)' % (target_view, target_image, target_weight, target_blur, target_detail))

    s = get_model(model_name, c.half_precision, c.save_memory)
    device = s.device
    labels = encode_prompts(s, prompts, negative_prompt)

    try:
        image = make_image_network(device)
        if checkpoint is not None:
            load_state(image, checkpoint['image'])
        views = [lambda view=view: view(image()) for view in illusion.views] #Each returns the current image seen through one view

        if init_image is not None:
            fit_to_picture(image, views[init_index], init_image, c.fit_steps, device)

        optim = torch.optim.SGD(image.parameters(), lr=c.lr) #Plain SGD has no internal state, so saving the network alone is enough to resume exactly

        def blur(img):
            if target_blur <= 0:
                return img
            kernel_size = 2 * int(np.ceil(3 * target_blur)) + 1
            return TF.gaussian_blur(img, kernel_size=kernel_size, sigma=target_blur)

        if target_index is not None:
            target_picture = load_picture(target_image, device)
            target_blurred = blur(target_picture)

        def target_step():
            #Instead of a Stable Diffusion step, the target view gets a plain pixel loss toward the picture (after blurring both),
            #plus optionally toward its full detail. The gradient flows back through the view into the same network the other views use.
            view = views[target_index]()
            loss = F.mse_loss(blur(view), target_blurred)
            if target_detail:
                loss = loss + target_detail * F.mse_loss(view, target_picture)
            loss = target_weight * TARGET_LOSS_SCALE * loss
            loss.backward()

        def target_error():
            #Average difference between the blurred target view and the blurred target picture, from 0 to 1
            with torch.no_grad():
                return (blur(views[target_index]()) - target_blurred).abs().mean().item()

        def sd_step(index):
            s.train_step(labels[index].embedding, views[index]()[None],
                         noise_coef=c.noise_coef * weights[index], guidance_scale=c.guidance_scale)

        def display_image():
            return rp.tiled_images([rp.as_numpy_image(view()) for view in views], length=len(views), border_thickness=0)

        s.max_step = c.max_step
        s.min_step = c.min_step
        settled_max_step = min(c.max_step, c.settle_max_step)

        def settle_schedule(step):
            #Highest noise level and learning rate factor for a step: full strength until settle_start, then a smooth
            #(cosine) decline to the settled values at settle_end, which are kept after it
            if not c.settle or step <= c.settle_start:
                return c.max_step, 1.0
            progress = min(1.0, (step - c.settle_start) / (c.settle_end - c.settle_start))
            eased = (1 - math.cos(math.pi * progress)) / 2
            return round(c.max_step + (settled_max_step - c.max_step) * eased), 1 + (c.settle_lr - 1) * eased

        if c.resume is not None:
            #Continue in the same folder, numbering new frames after the existing ones
            output_folder = os.path.normpath(c.resume)
            frames_folder = os.path.join(output_folder, 'frames')
            os.makedirs(frames_folder, exist_ok=True)
            existing_frames = [int(m.group(1)) for f in os.listdir(frames_folder) for m in [re.fullmatch(r'ims_(\d+)\.png', f)] if m]
            frame_index = max(existing_frames) + 1 if existing_frames else 0
            start_step = checkpoint['step'] if checkpoint is not None else frame_index * c.save_interval #Older runs: estimated from the frames
        else:
            output_folder = new_output_folder(c.output_root, illusion, prompts, c.name)
            frame_index = 0
            start_step = 0

        if target_index is not None:
            #Shows what the target view is actually matched against
            rp.save_image(rp.as_numpy_image(target_blurred), os.path.join(output_folder, 'target_blurred.png'))

        saved_target = None if target_image is None else dict(path=target_image, view=target_view, weight=target_weight, weight_scale=CURRENT_TARGET_WEIGHT_SCALE,
                                                              blur=target_blur, detail=target_detail)

        def checkpoint_now(step):
            save_checkpoint(output_folder, state_for_checkpoint(image), step, prompts, negative_prompt, illusion, model_name, saved_target)

        def save_progress(step):
            nonlocal frame_index
            with torch.no_grad():
                im = display_image()
            rp.save_image(im, os.path.join(output_folder, 'frames', 'ims_%04i.png' % frame_index))
            rp.save_image(im, os.path.join(output_folder, 'latest.png')) #Open this file to watch progress
            frame_index += 1
            checkpoint_now(step)

        def save_final(step):
            with torch.no_grad():
                for i, view in enumerate(views):
                    rp.save_image(rp.as_numpy_image(view()), os.path.join(output_folder, 'image_%s.png' % VIEW_LETTERS[i]))
                rp.save_image(display_image(), os.path.join(output_folder, 'final.png'))
            checkpoint_now(step)
            print()
            print('Saved results to folder:', repr(os.path.abspath(output_folder)))
            for i, description in enumerate(illusion.descriptions):
                print('    image_%s.png = %s' % (VIEW_LETTERS[i], description))
            if target_index is not None:
                print('    image_%s differs from the blurred target picture by %.1f%% on average (see target_blurred.png)' % (target_view, 100 * target_error()))
            print('Continue this run later with:')
            print('    python generate.py 1000 --resume "%s"' % output_folder)

        print()
        print('Training steps %i to %i' % (start_step, start_step + c.steps))
        print('Every %i steps we save an image in the form [%s], where' % (c.save_interval, ', '.join('image_' + VIEW_LETTERS[i] for i in range(len(views)))))
        for i, description in enumerate(illusion.descriptions):
            print('    image_%s = (%s)' % (VIEW_LETTERS[i], description))
        if c.settle:
            current_max_step, current_lr_factor = settle_schedule(start_step)
            print('Settling from step %i to %i: highest noise level %i -> %i, learning rate x1 -> x%s (now: %i, x%.2g)'
                  % (c.settle_start, c.settle_end, c.max_step, settled_max_step, c.settle_lr, current_max_step, current_lr_factor))
        print('Watch progress in:', os.path.abspath(os.path.join(output_folder, 'latest.png')))
        print()
        print('Press Ctrl+C at any time to stop early and save the current image')
        print()
        print('Please expect this to take quite a while to get good images! The longer you wait the better they\'ll be')

        #Save the starting point: the random initial image of a new run, or the reproduced picture after a restart
        if start_step == 0 or init_image is not None:
            save_progress(start_step)

        display_eta = rp.eta(c.steps, title='Status') if c.steps else None
        other_indices = [i for i in range(len(views)) if i != target_index]
        completed = start_step #Number of finished training steps
        interrupted = False
        try:
            for iter_num in range(start_step, start_step + c.steps):
                display_eta(iter_num - start_step) #Print the remaining time
                s.max_step, lr_factor = settle_schedule(iter_num)
                for group in optim.param_groups:
                    group['lr'] = c.lr * lr_factor #Applies to the target picture's pull too, so the balance between views stays the same
                if target_index is None:
                    #Train a randomly chosen view toward its prompt
                    sd_step(random.randrange(len(views)))
                else:
                    #Every step: pull the target view toward the picture (cheap), and train another view toward its prompt
                    target_step()
                    sd_step(random.choice(other_indices))
                optim.step()
                optim.zero_grad()
                completed = iter_num + 1
                if not completed % c.save_interval:
                    save_progress(completed)
        except KeyboardInterrupt:
            interrupted = True
            print()
            print('Interrupted early after %i steps' % completed)

        save_final(completed)
        #Clean up the progress display now; if it's left for Python's shutdown, rp prints a harmless but alarming traceback
        del display_eta
        return RunResult(output_folder=output_folder, start_step=start_step, end_step=completed, interrupted=interrupted)
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
