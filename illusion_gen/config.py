#Every setting of one run. The same names are used as generate() keyword arguments, batch config keys and (with dashes)
#command line options. No torch needed here, so configs can be checked before the model loads.
import dataclasses
import os
from dataclasses import dataclass
from typing import Optional

from .illusions import Illusion, resolve_illusion, view_index


class ConfigError(Exception):
    pass


#Used when a new run doesn't name its prompts (only fits 2-view illusions like twisting_squares and flip)
DEFAULT_PROMPTS = ["A cute bunny in a forest meadow Modern Disney style",
                   "A beautiful unicorn in a forest meadow Modern Disney style"]

DEFAULT_TARGET_VIEW = 'a'
DEFAULT_TARGET_WEIGHT = 1.0
DEFAULT_TARGET_BLUR = 4.0


@dataclass
class RunConfig:
    #What to make
    prompts: Optional[list] = None   #One prompt per view (image_a, image_b, ...). When resuming a run with a checkpoint, its saved prompts are used
    illusion: object = 'twisting_squares' #A name from illusions.ILLUSIONS, or an Illusion object with your own view functions
    squares: int = 5                 #twisting_squares only: a squares x squares grid
    negative_prompt: str = 'blurry ugly' #What you DON'T want to see
    steps: int = 5000                #Training steps to run (when resuming: how many MORE steps)

    #Training
    weights: Optional[list] = None   #Relative weight of each prompt; default depends on the illusion (twisting_squares: 1, 0.9)
    guidance_scale: float = 100
    noise_coef: float = 0.1
    lr: float = 1e-4                 #SGD learning rate
    max_step: int = 990              #Highest noise level used in training, out of 1000. Lower values (500-700) keep the composition of an init_image and only refine it
    min_step: int = 10
    seed: Optional[int] = None       #Set for repeatable runs

    #Starting point
    resume: Optional[str] = None     #Continue a stopped run in this folder (exact with checkpoint.pt; approximate restart from image_a.png without)
    init_image: Optional[str] = None #Retrain the network to reproduce this picture before training (e.g. an edited image_a.png)
    init_view: str = 'a'             #Which view init_image shows (b = an edited image_b.png)
    fit_steps: int = 1000            #Steps used to retrain the network to reproduce a picture

    #Target picture: train one view toward a picture instead of its prompt (pixel loss after blurring both)
    target_image: Optional[str] = None
    target_view: Optional[str] = None    #Default 'a'
    target_weight: Optional[float] = None #Default 1.0. Higher = closer match, less room for the other views
    target_blur: Optional[float] = None   #Default 4.0 px (of 256). 0 = match every pixel

    #Output
    output_root: str = 'untracked'   #Runs are saved in <output_root>/<illusion>_runs/<name>
    name: Optional[str] = None       #Run folder name; default is the prompts joined with '-'
    save_interval: int = 50          #Save a progress frame and checkpoint every this many steps

    #Model and GPU memory
    model: str = 'stable-diffusion-v1-5/stable-diffusion-v1-5' #Or a dreambooth of SD 1.5. When resuming with a checkpoint, its saved model is used
    half_precision: bool = True      #16-bit UNet/VAE: half the memory; False needs ~10 GB free GPU memory
    save_memory: object = 'auto'     #VAE gradient checkpointing: True, False, or 'auto' (on only if the GPU needs it)

    @classmethod
    def field_names(cls):
        return [f.name for f in dataclasses.fields(cls)]

    @classmethod
    def from_dict(cls, values, where='config'):
        #Build from a dict (e.g. a batch file entry), rejecting unknown keys so typos don't go unnoticed
        values = {str(k).replace('-', '_'): v for k, v in values.items()}
        unknown = sorted(set(values) - set(cls.field_names()))
        if unknown:
            raise ConfigError('%s: unknown setting(s) %s. Valid settings: %s' % (where, ', '.join(unknown), ', '.join(cls.field_names())))
        config = cls(**values)
        config.normalize(where)
        return config

    def normalize(self, where='config'):
        #Coerce types (YAML and the command line give strings, ints for floats, etc.) and check value ranges
        def fail(message):
            raise ConfigError('%s: %s' % (where, message))

        def number(name, kind, minimum=None, optional=False):
            value = getattr(self, name)
            if value is None and optional:
                return
            try:
                if isinstance(value, bool) or (kind is int and float(value) != int(float(value))):
                    raise ValueError
                value = kind(value)
            except (TypeError, ValueError):
                fail('%s must be a%s number, not %r' % (name, 'n integer' if kind is int else '', value))
            if minimum is not None and value < minimum:
                fail('%s must be at least %s, not %r' % (name, minimum, value))
            setattr(self, name, value)

        if self.prompts is not None:
            if isinstance(self.prompts, str) or not all(isinstance(p, str) for p in self.prompts):
                fail('prompts must be a list of strings, one per view, not %r' % (self.prompts,))
            self.prompts = list(self.prompts)
        if self.weights is not None:
            try:
                self.weights = [float(w) for w in self.weights]
            except (TypeError, ValueError):
                fail('weights must be a list of numbers, not %r' % (self.weights,))
            if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
                fail('weights must be non-negative and not all zero, not %r' % (self.weights,))

        number('steps', int, 0)
        number('squares', int, 1)
        number('guidance_scale', float)
        number('noise_coef', float)
        number('lr', float, 0)
        number('max_step', int, 0)
        number('min_step', int, 0)
        number('seed', int, optional=True)
        number('fit_steps', int, 0)
        number('target_weight', float, 0, optional=True)
        number('target_blur', float, 0, optional=True)
        number('save_interval', int, 1)
        if not self.min_step <= self.max_step < 1000:
            fail('need min_step <= max_step < 1000, not min_step=%r, max_step=%r' % (self.min_step, self.max_step))

        if isinstance(self.save_memory, str):
            lowered = self.save_memory.lower()
            self.save_memory = {'auto': 'auto', 'on': True, 'true': True, 'off': False, 'false': False}.get(lowered, self.save_memory)
        if self.save_memory not in ('auto', True, False):
            fail("save_memory must be 'auto', true or false, not %r" % (self.save_memory,))
        if not isinstance(self.half_precision, bool):
            fail('half_precision must be true or false, not %r' % (self.half_precision,))

        if not isinstance(self.illusion, Illusion):
            try:
                resolve_illusion(self.illusion, self.squares)
            except ValueError as e:
                fail(str(e))

    def check(self, where='config'):
        #Checks that can be done without loading anything: files exist, prompt counts fit the illusion.
        #(When resuming with a checkpoint, prompts and the illusion come from the checkpoint, so they're checked later.)
        self.normalize(where)

        def fail(message):
            raise ConfigError('%s: %s' % (where, message))

        for name in ['init_image', 'target_image']:
            path = getattr(self, name)
            if path is not None and not os.path.isfile(path):
                fail('%s not found: %r' % (name, path))
        if self.resume is not None and not os.path.isdir(self.resume):
            fail('run folder to resume not found: %r' % self.resume)

        illusion = resolve_illusion(self.illusion, self.squares)
        try:
            view_index(self.init_view, illusion)
            if self.target_view is not None:
                view_index(self.target_view, illusion)
        except ValueError as e:
            fail(str(e))
        if self.resume is None:
            prompts = self.prompts if self.prompts is not None else DEFAULT_PROMPTS
            if len(prompts) != illusion.num_views:
                fail('%s needs %i prompts (one per view), got %i' % (illusion.name, illusion.num_views, len(prompts)))
        if self.weights is not None and len(self.weights) != illusion.num_views:
            fail('%s needs %i weights (one per view), got %i' % (illusion.name, illusion.num_views, len(self.weights)))

    def describe(self):
        #One line summary, for batch listings
        illusion = self.illusion.name if isinstance(self.illusion, Illusion) else self.illusion
        if self.resume is not None:
            return 'resume "%s" for %i more steps' % (os.path.normpath(self.resume), self.steps)
        prompts = self.prompts if self.prompts is not None else DEFAULT_PROMPTS
        extra = ' from "%s"' % self.init_image if self.init_image else ''
        extra += ' with target "%s"' % self.target_image if self.target_image else ''
        return 'new %s, %i steps%s: %s' % (illusion, self.steps, extra, ' | '.join(prompts))
