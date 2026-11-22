#Standalone script version of twisting_squares_colab.ipynb
#Usage (any Python works to start it; it relaunches itself with the project's .venv):
#    python gen.py [STEPS]                            Start a new run (default 5000 steps)
#    python gen.py [STEPS] --resume RUN_FOLDER        Continue a stopped run for STEPS more steps
#    python gen.py [STEPS] --init-image PICTURE.png   Start a new run from a picture (e.g. an edited image_a.png)
#    python gen.py [STEPS] --target-image PHOTO.png   Make image_a look like a specific picture instead of prompt_a
#    python gen.py --help                             All options
#Progress frames, final images and a checkpoint are saved under untracked/twisting_squares_runs/

import argparse
import os
import re
import subprocess
import sys
import time
import copy

#The packages this script needs are installed in the project's .venv (see requirements-local.txt). If it was started
#with some other Python (e.g. plain "python" on PATH), rerun it with the venv's Python instead.
_venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv')
_venv_python = os.path.join(_venv_dir, 'Scripts', 'python.exe')
if os.path.exists(_venv_python) and os.path.normcase(os.path.realpath(sys.prefix)) != os.path.normcase(os.path.realpath(_venv_dir)):
    print('Relaunching with the project environment:', _venv_python, flush=True)
    _child = subprocess.Popen([_venv_python, os.path.abspath(__file__), *sys.argv[1:]])
    while True:
        try:
            sys.exit(_child.wait())
        except KeyboardInterrupt:
            pass #Ctrl+C reaches the relaunched script too, which stops and saves; wait for it to finish

#rp prints unicode characters that the default Windows console encoding can't handle
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

#rp's progress display calls colorama.init() on every update, and each call wraps stdout in another layer.
#When output is redirected to a file (e.g. python gen.py > log.txt), the layers pile up until Python's recursion
#limit crashes the run after ~250 steps. Initialize colorama once here and make later calls do nothing.
import colorama
colorama.just_fix_windows_console()
colorama.init = lambda *args, **kwargs: None

#SET YOUR PROMPTS HERE
#(When resuming a run that has a checkpoint, the prompts, negative prompt and square count saved with it are used instead)
#prompt_a = A beautiful mermaid ocean beach Modern Disney style 4k
#prompt_b = A beautiful unicorn Modern Disney style 4k
prompt_a = "A cute bunny in a forest meadow Modern Disney style"
prompt_b = "A beautiful unicorn in a forest meadow Modern Disney style"

# prompt_a = "A red semi truck pulling a gray trailer down the road photograph 4k"
# prompt_b = "The red bold letters \"ATS\" in the center 4k"

NUMBER_OF_SQUARES = 5 # A 5x5 twisting squares illusion

#Optional: Specify what you DON'T want to see
negative_prompt = 'blurry ugly'

DEFAULT_TARGET_WEIGHT = 1.0
DEFAULT_TARGET_BLUR = 4.0
DEFAULT_TARGET_DETAIL = 0.1
#Checkpoints saved before target weights were rescaled (no 'weight_scale' entry) used weights 1/0.03 times smaller
OLD_TARGET_WEIGHT_FACTOR = 1 / 0.03

parser = argparse.ArgumentParser(description='Generate a twisting squares illusion.')
parser.add_argument('steps', nargs='?', type=int, default=5000,
                    help='Number of training steps to run (when resuming: how many MORE steps). Default 5000.')
parser.add_argument('--resume', metavar='RUN_FOLDER',
                    help='Continue a stopped run in this folder. Exact if it has checkpoint.pt; for older runs without one, '
                         'it first retrains the network to reproduce the run\'s image_a.png (close, but not identical).')
parser.add_argument('--init-image', metavar='PICTURE',
                    help='Retrain the network to reproduce this picture before training, e.g. an image_a.png you edited. '
                         'Combine with --resume to continue that run from the edited picture; on its own it starts a new run.')
parser.add_argument('--twisted', action='store_true',
                    help='The --init-image picture is the twisted view (an edited image_b.png) rather than image_a.')
parser.add_argument('--max-step', type=int, default=990,
                    help='Highest noise level used in training, out of 1000 (default 990). High noise levels let Stable Diffusion '
                         'repaint the overall layout; lower values (e.g. 500-700) keep the composition of an --init-image and only refine it.')
parser.add_argument('--fit-steps', type=int, default=1000,
                    help='Steps used to retrain the network to reproduce a picture (default 1000).')
parser.add_argument('--target-image', metavar='PICTURE',
                    help='Train one view to look like this picture instead of its prompt. The other view still follows its prompt. '
                         'A new run starts from the picture. Since both views share the same pixels, the match is approximate: '
                         'see --target-weight and --target-blur.')
parser.add_argument('--target-view', choices=['a', 'b'],
                    help='Which view the target picture is for: a = the original image (default), b = the twisted image.')
parser.add_argument('--target-weight', type=float,
                    help='How strongly the view is pulled toward the target picture (default ' + str(DEFAULT_TARGET_WEIGHT) + '). 1 is tuned so the target view '
                         'stays recognizable (~4%% off the blurred picture) while the other view still forms its subject. '
                         'Around 3 the target view is a clean match but the other view mostly loses its subject.')
parser.add_argument('--target-blur', type=float,
                    help='Blur radius in pixels (of 256) applied to both the view and the picture before comparing them (default ' + str(DEFAULT_TARGET_BLUR) + '). '
                         'The view then only has to match the picture\'s overall shapes and colors, leaving fine detail free for the other view. '
                         '0 = match every pixel.')
parser.add_argument('--target-detail', type=float,
                    help='Strength of an extra pull toward the unblurred picture, relative to the blurred one (default ' + str(DEFAULT_TARGET_DETAIL) + '). '
                         'Keeps fine detail from turning into noisy dot texture. Higher (e.g. 0.3) = sharper target view, less room for the other view; 0 = off.')
args = parser.parse_args()

import torch

checkpoint = None
init_image_path = args.init_image
if args.resume:
    if not os.path.isdir(args.resume):
        sys.exit('Run folder not found: %r' % args.resume)
    checkpoint_path = os.path.join(args.resume, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        prompt_a          = checkpoint['prompt_a']
        prompt_b          = checkpoint['prompt_b']
        negative_prompt   = checkpoint['negative_prompt']
        NUMBER_OF_SQUARES = checkpoint['number_of_squares']
        print('Resuming from checkpoint at step %i' % checkpoint['step'])
        saved_target = checkpoint.get('target')
        if saved_target is not None and args.target_image is None:
            #Keep using the run's target picture, unless overridden on the command line
            args.target_image = saved_target['path']
            if args.target_view   is None: args.target_view   = saved_target['view']
            if args.target_weight is None:
                args.target_weight = saved_target['weight']
                if 'weight_scale' not in saved_target:
                    args.target_weight *= OLD_TARGET_WEIGHT_FACTOR
                    print('Converted target weight %s from an older checkpoint to the current scale: %.3g' % (saved_target['weight'], args.target_weight))
            if args.target_blur   is None: args.target_blur   = saved_target['blur']
            if args.target_detail is None: args.target_detail = saved_target.get('detail', 0.0)
    else:
        if init_image_path is None:
            init_image_path = os.path.join(args.resume, 'image_a.png')
            if not os.path.exists(init_image_path):
                sys.exit('%r has no checkpoint.pt or image_a.png to resume from' % args.resume)
        print('This run has no checkpoint, so the network will be retrained to reproduce %r (an approximate restart).' % init_image_path)
        print('WARNING: Older runs did not save their settings. The prompts and NUMBER_OF_SQUARES set in gen.py will be used,')
        print('         so make sure they match this run!')

if init_image_path is not None and not os.path.exists(init_image_path):
    sys.exit('Picture not found: %r' % init_image_path)

TARGET_IMAGE  = args.target_image
TARGET_VIEW   = args.target_view or 'a'
TARGET_WEIGHT = args.target_weight if args.target_weight is not None else DEFAULT_TARGET_WEIGHT
TARGET_BLUR   = args.target_blur   if args.target_blur   is not None else DEFAULT_TARGET_BLUR
TARGET_DETAIL = args.target_detail if args.target_detail is not None else DEFAULT_TARGET_DETAIL
if TARGET_IMAGE is not None:
    if not os.path.exists(TARGET_IMAGE):
        sys.exit('Target picture not found: %r' % TARGET_IMAGE)
    TARGET_IMAGE = os.path.abspath(TARGET_IMAGE) #Saved in checkpoints, so resuming works from any directory
    if init_image_path is None and checkpoint is None:
        init_image_path = TARGET_IMAGE #Start a new run from the target picture, rather than from noise
        args.twisted = TARGET_VIEW == 'b'

print()
print('Negative prompt:',repr(negative_prompt))
print()
print('Chosen prompts:')
print('    prompt_a =', repr(prompt_a) + ('   (not used: image_a follows the target picture)' if TARGET_IMAGE and TARGET_VIEW == 'a' else '')) #This will be the original image
print('    prompt_b =', repr(prompt_b) + ('   (not used: image_b follows the target picture)' if TARGET_IMAGE and TARGET_VIEW == 'b' else '')) #This will be the twisted image
print('    NUMBER_OF_SQUARES =', NUMBER_OF_SQUARES)
if TARGET_IMAGE:
    print('Target picture for image_%s: %r  (weight %s, blur %s px, detail %s)' % (TARGET_VIEW, TARGET_IMAGE, TARGET_WEIGHT, TARGET_BLUR, TARGET_DETAIL))

from rp import *
import numpy as np
import rp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import source.stable_diffusion as sd
from easydict import EasyDict
from source.learnable_textures import LearnableImageFourier
from source.stable_diffusion_labels import NegativeLabel
from itertools import chain
from PIL import Image


#You can select the original Stable Diffusion 1.5 or some dreambooth of it
# model_name="CompVis/stable-diffusion-v1-4"
model_name="stable-diffusion-v1-5/stable-diffusion-v1-5"
# model_name="nitrosocke/Arcane-Diffusion"
#rp.select_torch_device() crashes on Windows when it can't inspect other processes using the GPU
gpu='cuda' if torch.cuda.is_available() else 'cpu'
s=sd.StableDiffusion(gpu,model_name)
device=s.device

label_a = NegativeLabel(prompt_a,negative_prompt)
label_b = NegativeLabel(prompt_b,negative_prompt)

#GPU MEMORY AND SPEED SETTINGS
#The defaults work on both a ~6GB card like the GTX 1060 and big cloud GPUs.
#If training doesn't fit in GPU memory, Windows silently spills into system RAM and each step gets 3-7x slower.
#
#HALF_PRECISION: Run the UNet and VAE in 16-bit, halving their memory (1.8 GB of weights instead of 3.6 GB).
#   On RTX 20-series and newer cards, including cloud GPUs, it's also much faster thanks to tensor cores.
#   On GTX 10-series cards it computes a bit slower, but it's the only way to fit in 6 GB.
#   Set to False for full 32-bit precision, identical to the original notebook (needs ~10 GB of free GPU memory).
HALF_PRECISION = True
#
#SAVE_MEMORY: Gradient checkpointing in the VAE. Instead of storing intermediate results for the backward pass,
#   it recomputes them: ~0.8 GB less peak memory in half precision (~1.5 GB in full), but each step takes longer.
#   'auto' turns it on only if the GPU doesn't have enough free memory without it. Or set True / False.
SAVE_MEMORY = 'auto'

#Peak extra memory one training step needs on top of the model weights, in GiB, measured on a GTX 1060.
#Keys are (HALF_PRECISION, SAVE_MEMORY).
STEP_MEMORY_GIB = {(True, False): 2.83, (True, True): 2.07, (False, False): 4.55, (False, True): 3.06}

if device.type == 'cuda':
    #The text encoder is only needed to encode the prompts, which happened above, so free its GPU memory
    s.text_encoder.to('cpu')
    if HALF_PRECISION:
        s.unet.half()
        s.vae.half()
        _predict_noise, _encode_imgs, _decode_latents = s.predict_noise, s.encode_imgs, s.decode_latents
        s.predict_noise  = lambda latents, embeddings, t: _predict_noise(latents.half(), embeddings.half(), t).float()
        s.encode_imgs    = lambda imgs: _encode_imgs(imgs.half()).float()
        s.decode_latents = lambda latents: _decode_latents(latents.half()).float()
    torch.cuda.empty_cache()

    free_gib = torch.cuda.mem_get_info()[0] / 2**30
    fits = lambda save_memory: free_gib > STEP_MEMORY_GIB[HALF_PRECISION, save_memory] * 1.15 #Headroom for allocator overhead; on the 1060, 13% spare spilled and 19% didn't
    save_memory = (not fits(False)) if SAVE_MEMORY == 'auto' else SAVE_MEMORY
    if save_memory:
        s.vae.enable_gradient_checkpointing()

    print()
    print('GPU: %s, %.1f GB free after loading the model' % (torch.cuda.get_device_name(device), free_gib))
    print('    Half precision: %s   Save memory: %s%s' % ('on' if HALF_PRECISION else 'off', 'on' if save_memory else 'off', ' (auto)' if SAVE_MEMORY == 'auto' else ''))
    if not fits(save_memory):
        print('    WARNING: Training may not fit in the free GPU memory. If steps are much slower than usual,')
        print('             close other programs using the GPU (web browsers use it too)' +(', or set HALF_PRECISION = True' if not HALF_PRECISION else '') + (', or set SAVE_MEMORY = True' if not save_memory else '') + '.')

#Image Parametrization and Initialization (this section takes vram)

#Select Learnable Image Size (this has big VRAM implications!):
#Note: We use implicit neural representations for better image quality
#They're previously used in our paper "TRITON: Neural Neural Textures make Sim2Real Consistent" (see tritonpaper.github.io)
# ... and that representation is based on Fourier Feature Networks (see bmild.github.io/fourfeat)
learnable_image_maker = lambda: LearnableImageFourier(height=256,width=256,num_features=256,hidden_dim=256,scale=10).to(s.device);SIZE=256

image=learnable_image_maker()

#Checkpoints leave out the network's precomputed Fourier features (134 MB), since they're fixed and can be recomputed
#exactly from its random frequency table (feature_extractor.freqs). That keeps checkpoints ~2 MB instead of ~136 MB.
def image_state_for_checkpoint():
    return {k: v for k, v in image.state_dict().items() if k != 'features'}

def load_image_state(state):
    image.load_state_dict(state, strict='features' in state) #Older checkpoints include the features
    if 'features' not in state:
        #Recompute on the CPU, the same way LearnableImageFourier.__init__ does, so the result is bit-identical
        with torch.no_grad():
            cpu_feature_extractor = copy.deepcopy(image.feature_extractor).cpu()
            image.features.copy_(cpu_feature_extractor(image.uv_grid.cpu()))

if checkpoint is not None:
    load_image_state(checkpoint['image'])


def rotate_tiles(image, num_divisions=NUMBER_OF_SQUARES, inverse=False):
    image=as_torch_image(image)
    # Assuming image is a tensor of shape (num_channels, height, width)
    num_channels, height, width = image.shape

    tile_size=width//num_divisions

    # Calculate the number of tiles in each dimension
    tiles_x = width // tile_size
    tiles_y = height // tile_size

    # Start from a copy, so any leftover edge pixels (when the size isn't divisible by num_divisions) stay put instead of turning black
    output = image.clone()

    direction = -1 if inverse else 1 # inverse=True undoes the twist

    for x in range(tiles_x):
        for y in range(tiles_y):
            # Extract the tile
            tile = image[:, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]

            # Check if the tile should be rotated 90 or -90 degrees (checker pattern)
            if (x + y) % 2 == 0:
                # Rotate 90 degrees
                tile = tile.rot90(direction, [1, 2])
            else:
                # Rotate -90 degrees
                tile = tile.rot90(-direction, [1, 2])

            # Place the rotated tile back in the output tensor
            output[:, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size] = tile

    return output

learnable_image_a=lambda: image()
learnable_image_b=lambda: rotate_tiles(image()) #Twisted squares


def load_picture(path):
    #Load a picture as a [3, SIZE, SIZE] tensor with values between 0 and 1
    picture = Image.open(path).convert('RGB').resize((SIZE, SIZE), Image.LANCZOS)
    return torch.tensor(np.asarray(picture), dtype=torch.float32, device=device).permute(2, 0, 1) / 255

def fit_image_to_picture(path, steps, twisted=False):
    #Retrain the image network so it reproduces a picture (the "approximate restart")
    target = load_picture(path)
    if twisted:
        target = rotate_tiles(target, inverse=True) #Untwist it, since the network stores the original view
    print()
    print('Retraining the network to reproduce %r for %i steps...' % (path, steps))
    fit_optim = torch.optim.Adam(image.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(fit_optim, steps, eta_min=1e-5)
    for i in range(steps):
        loss = F.mse_loss(image(), target)
        fit_optim.zero_grad()
        loss.backward()
        fit_optim.step()
        schedule.step()
    psnr = -10 * np.log10(loss.item())
    print('    Done. Match quality: %.1f dB PSNR (30+ is very close; below ~25 you\'ll see noticeable softening)' % psnr)

if init_image_path is not None:
    fit_image_to_picture(init_image_path, args.fit_steps, twisted=args.twisted)

optim=torch.optim.SGD(image.parameters(),lr=1e-4) #Plain SGD has no internal state, so saving the network alone is enough to resume exactly



labels=[label_a,label_b]
learnable_images=[learnable_image_a,learnable_image_b]

#The weight coefficients for each prompt. For example, if we have [0,1], then only the twisted mode will be optimized
#(These come from this file even when resuming, so you can adjust them mid-run)
weights=[1,0.9]

weights=rp.as_numpy_array(weights)
weights=weights/weights.sum()
weights=weights*len(weights)


#TARGET PICTURE: instead of a Stable Diffusion step, the target view gets a plain pixel loss toward the picture.
#The gradient flows back through the view (e.g. through rotate_tiles) into the same network the other view uses.
#Calibration: an SD step's pixel gradient is about as large as this loss's gradient at a scale of 6.5e5 with every
#pixel off by 0.1. At that strength the other view never forms its subject, so weight 1 is 3% of it. In 700-step
#tests (apple target, unicorn prompt, blur 4, detail 0.1), 3% gave a recognizable apple with visible unicorns in
#the other view; 10% gave a clean apple and no unicorn.
TARGET_LOSS_SCALE = 6.5e5 / OLD_TARGET_WEIGHT_FACTOR

def blur(img):
    if TARGET_BLUR <= 0:
        return img
    kernel_size = 2 * int(np.ceil(3 * TARGET_BLUR)) + 1
    return TF.gaussian_blur(img, kernel_size=kernel_size, sigma=TARGET_BLUR)

if TARGET_IMAGE is not None:
    target_index = 0 if TARGET_VIEW == 'a' else 1
    other_index = 1 - target_index
    target_picture = load_picture(TARGET_IMAGE)
    target_blurred = blur(target_picture)

def target_step():
    #Pull the target view toward the target picture (after blurring both), plus optionally toward its full detail
    view = learnable_images[target_index]()
    loss = F.mse_loss(blur(view), target_blurred)
    if TARGET_DETAIL:
        loss = loss + TARGET_DETAIL * F.mse_loss(view, target_picture)
    loss = TARGET_WEIGHT * TARGET_LOSS_SCALE * loss
    loss.backward()

def sd_step(label, learnable_image, weight):
    return s.train_step(
        label.embedding,
        learnable_image()[None],

        #PRESETS (uncomment one):
        noise_coef=.1*weight,guidance_scale=100,#10
        # noise_coef=0,image_coef=-.01,guidance_scale=50,
        # noise_coef=0,image_coef=-.005,guidance_scale=50,
        # noise_coef=.1,image_coef=-.010,guidance_scale=50,
        # noise_coef=.1,image_coef=-.005,guidance_scale=50,
        # noise_coef=.1*weight, image_coef=-.005*weight, guidance_scale=50,
    )


def get_display_image():
    return rp.tiled_images(
        [
            rp.as_numpy_image(learnable_image_a()),
            rp.as_numpy_image(learnable_image_b()),
        ],
        length=len(learnable_images),
        border_thickness=0,
    )


NUM_ITER = args.steps

#Set the minimum and maximum noise timesteps for the dream loss (aka score distillation loss)
s.max_step=MAX_STEP=args.max_step
s.min_step=MIN_STEP=10

SAVE_INTERVAL = 50 #Save a progress frame and checkpoint every this many steps

if args.resume:
    #Continue in the same folder, numbering new frames after the existing ones
    output_folder = args.resume
    os.makedirs(os.path.join(output_folder, 'frames'), exist_ok=True)
    existing_frames = [int(m.group(1)) for f in os.listdir(os.path.join(output_folder, 'frames')) for m in [re.fullmatch(r'ims_(\d+)\.png', f)] if m]
    frame_index = max(existing_frames) + 1 if existing_frames else 0
    start_step = checkpoint['step'] if checkpoint is not None else frame_index * SAVE_INTERVAL #Older runs: estimated from the frames
else:
    #Output folder, named after the prompts (with characters Windows doesn't allow in filenames removed)
    run_name = re.sub(r'[<>:"/\\|?*]', '', '-'.join([prompt_a, prompt_b])).strip()
    output_folder = os.path.join('untracked', 'twisting_squares_runs', run_name)
    if os.path.exists(output_folder):
        output_folder += '_%i' % time.time()
    os.makedirs(os.path.join(output_folder, 'frames'))
    frame_index = 0
    start_step = 0

if TARGET_IMAGE is not None:
    #Shows what the target view is actually matched against
    rp.save_image(rp.as_numpy_image(target_blurred), os.path.join(output_folder, 'target_blurred.png'))

def target_error():
    #Average difference between the blurred target view and the blurred target picture, from 0 to 1
    with torch.no_grad():
        return (blur(learnable_images[target_index]()) - target_blurred).abs().mean().item()

def save_checkpoint(step):
    #Write to a temporary file first, so stopping mid-save can't corrupt the existing checkpoint
    path = os.path.join(output_folder, 'checkpoint.pt')
    torch.save(dict(
        image             = image_state_for_checkpoint(),
        step              = step,
        prompt_a          = prompt_a,
        prompt_b          = prompt_b,
        negative_prompt   = negative_prompt,
        number_of_squares = NUMBER_OF_SQUARES,
        model_name        = model_name,
        target            = None if TARGET_IMAGE is None else dict(path=TARGET_IMAGE, view=TARGET_VIEW, weight=TARGET_WEIGHT, weight_scale=2, blur=TARGET_BLUR, detail=TARGET_DETAIL),
    ), path + '.tmp')
    os.replace(path + '.tmp', path)

def save_progress(step):
    global frame_index
    with torch.no_grad():
        im = get_display_image()
    rp.save_image(im, os.path.join(output_folder, 'frames', 'ims_%04i.png' % frame_index))
    rp.save_image(im, os.path.join(output_folder, 'latest.png')) #Open this file to watch progress
    frame_index += 1
    save_checkpoint(step)

def save_final(step):
    with torch.no_grad():
        rp.save_image(rp.as_numpy_image(learnable_image_a()), os.path.join(output_folder, 'image_a.png'))
        rp.save_image(rp.as_numpy_image(learnable_image_b()), os.path.join(output_folder, 'image_b.png'))
        rp.save_image(get_display_image(), os.path.join(output_folder, 'final.png'))
    save_checkpoint(step)
    print()
    print('Saved results to folder:', repr(os.path.abspath(output_folder)))
    print('    image_a.png = the illusion image (twist the squares to get image_b.png)')
    if TARGET_IMAGE is not None:
        print('    image_%s differs from the blurred target picture by %.1f%% on average (see target_blurred.png)' % (TARGET_VIEW, 100 * target_error()))
    print('Continue this run later with:')
    print('    .venv\\Scripts\\python gen.py 1000 --resume "%s"' % output_folder)

display_eta=rp.eta(NUM_ITER, title='Status')

print()
print('Training steps %i to %i' % (start_step, start_step + NUM_ITER))
print('Every %i steps we save an image in the form [image_a, image_b], where'%SAVE_INTERVAL)
print('    image_a = (the original image)')
print('    image_b = (image_a, with its squares twisted)')
print('Watch progress in:', os.path.abspath(os.path.join(output_folder, 'latest.png')))
print()
print('Press Ctrl+C at any time to stop early and save the current image')
print()
print('Please expect this to take quite a while to get good images! The longer you wait the better they\'ll be')

#Save the starting point: the random initial image of a new run, or the reproduced picture after a restart
if start_step == 0 or init_image_path is not None:
    save_progress(start_step)

completed = start_step #Number of finished training steps
try:
    for iter_num in range(start_step, start_step + NUM_ITER):
        display_eta(iter_num - start_step) #Print the remaining time

        if TARGET_IMAGE is None:
            #Train a randomly chosen view toward its prompt
            for label,learnable_image,weight in rp.random_batch(list(zip(labels,learnable_images,weights)), batch_size=1):
                sd_step(label, learnable_image, weight)
        else:
            #Every step: pull the target view toward the picture (cheap), and train the other view toward its prompt
            target_step()
            sd_step(labels[other_index], learnable_images[other_index], weights[other_index])

        optim.step()
        optim.zero_grad()
        completed = iter_num + 1

        if not completed % SAVE_INTERVAL:
            save_progress(completed)
except KeyboardInterrupt:
    print()
    print('Interrupted early after %i steps'%completed)

save_final(completed)

#Clean up the progress display now; if it's left for Python's shutdown, rp prints a harmless but alarming traceback
del display_eta
