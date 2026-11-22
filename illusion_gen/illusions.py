#Illusion types. An illusion is one learnable image seen several ways ("views"); each view is trained toward its own
#prompt. A view is just a function from the image tensor [3, H, W] to another image tensor, so you can make your own:
#
#    Illusion('mirror', [lambda img: img, lambda img: img.flip(2)], ['the original image', 'the mirrored image'])
#
#and pass it to generate(illusion=...). Views must be differentiable (rotations, flips, crops, ... are).
#No torch import is needed here: views only call tensor methods, and config validation can load this module cheaply.
from dataclasses import dataclass, field

VIEW_LETTERS = 'abcdefghijklmnopqrstuvwxyz' #Views are named image_a, image_b, ... in output files and options


@dataclass
class Illusion:
    name: str               #Also names the output folder: untracked/<name>_runs
    views: list             #One function per prompt: image tensor [3, H, W] -> view tensor [3, H, W]
    descriptions: list = None   #Human-readable description of each view, for printouts
    default_weights: list = None #Relative weight of each prompt, if the run doesn't set weights
    params: dict = field(default_factory=dict) #Settings that shape the views, saved in checkpoints

    def __post_init__(self):
        if self.descriptions is None:
            self.descriptions = ['view %s' % VIEW_LETTERS[i] for i in range(len(self.views))]
        if self.default_weights is None:
            self.default_weights = [1] * len(self.views)

    @property
    def num_views(self):
        return len(self.views)

    @property
    def output_dir_name(self):
        return self.name + '_runs'


def rotate_tiles(image, num_divisions, inverse=False):
    #Rotate each square of a num_divisions x num_divisions grid by 90 degrees, alternating direction in a checker pattern
    num_channels, height, width = image.shape
    tile_size = width // num_divisions
    tiles_x = width // tile_size
    tiles_y = height // tile_size

    #Start from a copy, so any leftover edge pixels (when the size isn't divisible by num_divisions) stay put instead of turning black
    output = image.clone()
    direction = -1 if inverse else 1 #inverse=True undoes the twist

    for x in range(tiles_x):
        for y in range(tiles_y):
            tile = image[:, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
            tile = tile.rot90(direction if (x + y) % 2 == 0 else -direction, [1, 2])
            output[:, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size] = tile
    return output


def twisting_squares(squares=5):
    return Illusion(
        name='twisting_squares',
        views=[lambda img: img, lambda img: rotate_tiles(img, squares)],
        descriptions=['the original image', 'image_a, with its squares twisted'],
        default_weights=[1, 0.9],
        params=dict(squares=squares),
    )


def flip():
    return Illusion(
        name='flip',
        views=[lambda img: img, lambda img: img.rot90(2, [1, 2])],
        descriptions=['the right-side up image', 'image_a, upside down'],
    )


def rotate():
    return Illusion(
        name='rotate',
        views=[lambda img, k=k: img.rot90(k, [1, 2]) for k in range(4)],
        descriptions=['the original image', 'image_a, rotated 90 degrees', 'image_a, rotated 180 degrees', 'image_a, rotated 270 degrees'],
    )


#name -> function(squares) that builds the illusion. Add new illusion types here.
ILLUSIONS = {
    'twisting_squares': lambda squares: twisting_squares(squares),
    'flip'            : lambda squares: flip(),
    'rotate'          : lambda squares: rotate(),
}


def resolve_illusion(illusion, squares=5):
    #Accepts an illusion name or a ready-made Illusion
    if isinstance(illusion, Illusion):
        return illusion
    if illusion not in ILLUSIONS:
        raise ValueError('Unknown illusion %r. Choose one of: %s' % (illusion, ', '.join(ILLUSIONS)))
    return ILLUSIONS[illusion](squares)


def view_index(letter, illusion):
    index = VIEW_LETTERS.find(letter) if isinstance(letter, str) and len(letter) == 1 else -1
    if not 0 <= index < illusion.num_views:
        raise ValueError('View %r does not exist: %s has views %s' % (letter, illusion.name, ', '.join(VIEW_LETTERS[:illusion.num_views])))
    return index
