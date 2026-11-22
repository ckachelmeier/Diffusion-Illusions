#checkpoint.pt in each run folder: the image network plus the settings needed to continue the run
import os

import torch

CHECKPOINT_NAME = 'checkpoint.pt'


def load_checkpoint(folder):
    #Returns the checkpoint as a dict in the current format, or None if the folder has none.
    #Checkpoints written by gen.py (prompt_a / prompt_b / number_of_squares) are converted.
    path = os.path.join(folder, CHECKPOINT_NAME)
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location='cpu')
    if 'prompts' not in checkpoint:
        checkpoint['prompts']  = [checkpoint['prompt_a'], checkpoint['prompt_b']]
        checkpoint['illusion'] = 'twisting_squares'
        checkpoint['illusion_params'] = dict(squares=checkpoint['number_of_squares'])
    checkpoint.setdefault('illusion_params', {})
    checkpoint.setdefault('target', None)
    checkpoint.setdefault('model_name', None)
    return checkpoint


def save_checkpoint(folder, image_state, step, prompts, negative_prompt, illusion, model_name, target):
    #Write to a temporary file first, so stopping mid-save can't corrupt the existing checkpoint
    path = os.path.join(folder, CHECKPOINT_NAME)
    data = dict(
        image           = image_state,
        step            = step,
        prompts         = list(prompts),
        negative_prompt = negative_prompt,
        illusion        = illusion.name,
        illusion_params = dict(illusion.params),
        model_name      = model_name,
        target          = target,
    )
    if illusion.name == 'twisting_squares' and 'squares' in illusion.params and len(prompts) == 2:
        #Also in gen.py's format, so gen.py can resume these runs too
        data.update(prompt_a=prompts[0], prompt_b=prompts[1], number_of_squares=illusion.params['squares'])
    torch.save(data, path + '.tmp')
    os.replace(path + '.tmp', path)
