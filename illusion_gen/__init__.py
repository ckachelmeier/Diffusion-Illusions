"""Restructured version of gen.py: illusion generation as a reusable function, plus batch runs from a config file.

Use it from Python (start Python with the project's .venv):
    from illusion_gen.runner import generate
    generate(['A cute bunny in a forest meadow', 'A beautiful unicorn in a forest meadow'], 'twisting_squares', 1000)

Or from the command line: see generate.py.

Modules:
    env.py            Relaunching with the project's .venv, console encoding
    config.py         RunConfig: every setting of one run, its defaults and validation (no torch needed)
    illusions.py      The illusion types: which views of the image each prompt is trained on
    model.py          Loading Stable Diffusion once (with the GPU memory settings) and encoding prompts
    image_network.py  The learnable image network: creating it, fitting it to a picture
    checkpoints.py    Saving and loading checkpoint.pt (also reads checkpoints written by gen.py)
    runner.py         generate() / run(): one training run
    batch.py          Reading a batch config file into a list of runs, and running them in order

This file stays empty of imports on purpose: generate.py imports from this package before it has relaunched
itself with the .venv's Python, so nothing here may need torch.
"""
