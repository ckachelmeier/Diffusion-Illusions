#The learnable image: an implicit neural representation (Fourier features fed into a per-pixel MLP).
#Previously used in "TRITON: Neural Neural Textures make Sim2Real Consistent" (tritonpaper.github.io),
#based on Fourier Feature Networks (bmild.github.io/fourfeat)
import copy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from source.learnable_textures import LearnableImageFourier

SIZE = 256 #Learnable image size (big VRAM implications!). Checkpoints only fit networks of the same size.


def make_image_network(device):
    return LearnableImageFourier(height=SIZE, width=SIZE, num_features=256, hidden_dim=256, scale=10).to(device)


#Checkpoints leave out the network's precomputed Fourier features (134 MB), since they're fixed and can be recomputed
#exactly from its random frequency table (feature_extractor.freqs). That keeps checkpoints ~2 MB instead of ~136 MB.
def state_for_checkpoint(image):
    return {k: v for k, v in image.state_dict().items() if k != 'features'}


def load_state(image, state):
    image.load_state_dict(state, strict='features' in state) #Older checkpoints include the features
    if 'features' not in state:
        #Recompute on the CPU, the same way LearnableImageFourier.__init__ does, so the result is bit-identical
        with torch.no_grad():
            cpu_feature_extractor = copy.deepcopy(image.feature_extractor).cpu()
            image.features.copy_(cpu_feature_extractor(image.uv_grid.cpu()))


def load_picture(path, device):
    #Load a picture as a [3, SIZE, SIZE] tensor with values between 0 and 1
    picture = Image.open(path).convert('RGB').resize((SIZE, SIZE), Image.LANCZOS)
    return torch.tensor(np.asarray(picture), dtype=torch.float32, device=device).permute(2, 0, 1) / 255


def fit_to_picture(image, view, path, steps, device):
    #Retrain the image network so that view() (e.g. the twisted view) reproduces a picture: the "approximate restart"
    target = load_picture(path, device)
    print()
    print('Retraining the network to reproduce %r for %i steps...' % (path, steps))
    optim = torch.optim.Adam(image.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max(steps, 1), eta_min=1e-5)
    loss = None
    for _ in range(steps):
        loss = F.mse_loss(view(), target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        schedule.step()
    if loss is not None:
        psnr = -10 * np.log10(max(loss.item(), 1e-12))
        print('    Done. Match quality: %.1f dB PSNR (30+ is very close; below ~25 you\'ll see noticeable softening)' % psnr)
