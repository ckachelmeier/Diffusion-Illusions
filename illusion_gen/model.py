#Loading Stable Diffusion with the GPU memory settings, kept loaded between runs of a batch
import gc

import torch

#Peak extra memory one training step needs on top of the model weights, in GiB, measured on a GTX 1060.
#Keys are (half_precision, save_memory).
STEP_MEMORY_GIB = {(True, False): 2.83, (True, True): 2.07, (False, False): 4.55, (False, True): 3.06}

#half_precision: Run the UNet and VAE in 16-bit, halving their memory (1.8 GB of weights instead of 3.6 GB).
#   On RTX 20-series and newer cards, including cloud GPUs, it's also much faster thanks to tensor cores.
#   On GTX 10-series cards it computes a bit slower, but it's the only way to fit in 6 GB.
#   False = full 32-bit precision, identical to the original notebook (needs ~10 GB of free GPU memory).
#save_memory: Gradient checkpointing in the VAE. Instead of storing intermediate results for the backward pass,
#   it recomputes them: ~0.8 GB less peak memory in half precision (~1.5 GB in full), but each step takes longer.
#   'auto' turns it on only if the GPU doesn't have enough free memory without it.
#If training doesn't fit in GPU memory, Windows silently spills into system RAM and each step gets 3-7x slower.

_loaded = None #(key, StableDiffusion) of the currently loaded model


def get_model(model_name, half_precision=True, save_memory='auto'):
    #Returns the loaded StableDiffusion, loading it only if these settings differ from the last call
    global _loaded
    key = (model_name, half_precision, save_memory)
    if _loaded is not None and _loaded[0] == key:
        return _loaded[1]
    unload_model()

    import source.stable_diffusion as sd
    #rp.select_torch_device() crashes on Windows when it can't inspect other processes using the GPU
    gpu = 'cuda' if torch.cuda.is_available() else 'cpu'
    s = sd.StableDiffusion(gpu, model_name)
    device = s.device

    if device.type == 'cuda':
        #The text encoder is only needed to encode prompts (see encode_prompts), so keep it off the GPU
        s.text_encoder.to('cpu')
        if half_precision:
            s.unet.half()
            s.vae.half()
            _predict_noise, _encode_imgs, _decode_latents = s.predict_noise, s.encode_imgs, s.decode_latents
            s.predict_noise  = lambda latents, embeddings, t: _predict_noise(latents.half(), embeddings.half(), t).float()
            s.encode_imgs    = lambda imgs: _encode_imgs(imgs.half()).float()
            s.decode_latents = lambda latents: _decode_latents(latents.half()).float()
        torch.cuda.empty_cache()

        free_gib = torch.cuda.mem_get_info()[0] / 2**30
        fits = lambda use_save_memory: free_gib > STEP_MEMORY_GIB[half_precision, use_save_memory] * 1.15 #Headroom for allocator overhead; on the 1060, 13% spare spilled and 19% didn't
        use_save_memory = (not fits(False)) if save_memory == 'auto' else save_memory
        if use_save_memory:
            s.vae.enable_gradient_checkpointing()

        print()
        print('GPU: %s, %.1f GB free after loading the model' % (torch.cuda.get_device_name(device), free_gib))
        print('    Half precision: %s   Save memory: %s%s' % ('on' if half_precision else 'off', 'on' if use_save_memory else 'off', ' (auto)' if save_memory == 'auto' else ''))
        if not fits(use_save_memory):
            print('    WARNING: Training may not fit in the free GPU memory. If steps are much slower than usual,')
            print('             close other programs using the GPU (web browsers use it too)' + (', or use half_precision' if not half_precision else '') + (', or set save_memory to true' if not use_save_memory else '') + '.')

    _loaded = (key, s)
    return s


def unload_model():
    global _loaded
    if _loaded is None:
        return
    _loaded = None
    import source.stable_diffusion as sd
    sd._stable_diffusion_singleton = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_prompts(s, prompts, negative_prompt):
    #Returns one label (holding the text embedding) per prompt. The text encoder visits the GPU only for this.
    from source.stable_diffusion_labels import NegativeLabel
    on_gpu = s.device.type == 'cuda'
    if on_gpu:
        s.text_encoder.to(s.device)
    try:
        return [NegativeLabel(prompt, negative_prompt) for prompt in prompts]
    finally:
        if on_gpu:
            s.text_encoder.to('cpu')
            torch.cuda.empty_cache()
