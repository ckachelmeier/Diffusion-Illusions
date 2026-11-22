# Running on vast.ai

A vast.ai instance is a Docker container on a rented GPU machine. The flow here: the instance clones this repo from
GitHub and runs `vast/setup.sh`, you SSH in to start runs with `vast/run.sh`, and you copy the results back to your PC
with `vast/pull_results.ps1`. The Stable Diffusion weights download from Hugging Face during setup, so there's nothing
big to upload.

| File | What it does |
|---|---|
| `setup.sh` | Installs everything on the instance: system libraries, Python 3.10 in `.venv`, torch for its GPU, the requirements, the model weights. Then checks it all works. |
| `run.sh` | Starts `generate.py` in a tmux session (keeps running if SSH drops) and logs to `untracked/logs/`. |
| `pull_results.ps1` | On your PC: copies the instance's `untracked/` folder (runs and logs) into yours. |
| `constraints.txt` | Package versions from the local Windows `.venv`, so the instance runs the same versions. |

## Once

1. **Push your code.** The instance clones the `local-changes` branch from GitHub, so it only gets what you've committed
   and pushed.
2. **Add your SSH key to vast.ai** (Account → SSH Keys). If you don't have one, run `ssh-keygen -t ed25519` in
   PowerShell, then paste in the contents of `~\.ssh\id_ed25519.pub`.

## Each time

1. **Rent an instance.** On the vast.ai Templates page, pick the PyTorch template (any image with CUDA works), and set:
   - **Disk:** 40 GB or more (torch is ~6 GB, the model ~5 GB, plus the base image).
   - **On-start script:** paste this. On first boot it clones the repo and runs the setup, which takes around 5–10 minutes:
     ```bash
     cd /workspace && { [ -d diffusion-illusion ] || git clone -b local-changes https://github.com/ckachelmeier/Diffusion-Illusions.git diffusion-illusion; } && bash diffusion-illusion/vast/setup.sh > /workspace/setup.log 2>&1
     ```
   - **GPU:** anything with 8 GB+ VRAM works. An RTX 3090/4090 is a good price per step.

2. **Connect.** The instance's Connect button shows the SSH command, e.g. `ssh -p 41234 root@123.45.67.89`. Then wait
   for the setup to finish:
   ```bash
   tail -f /workspace/setup.log     # ends with "Setup finished"; Ctrl+C to stop watching
   ```
   vast.ai opens SSH sessions inside tmux. That's fine: `run.sh` handles it.

3. **Put your run config there.** Either write it on the instance (`nano runs.yaml`, starting from `runs.example.yaml`),
   or copy it up from your PC:
   ```powershell
   scp -P 41234 runs.yaml root@123.45.67.89:/workspace/diffusion-illusion/
   ```
   To continue a run you started locally, copy its folder up the same way (`scp -P PORT -r "untracked\twisting_squares_runs\RUN NAME" root@ADDRESS:/workspace/diffusion-illusion/untracked/twisting_squares_runs/`)
   and list it under `resume:` with forward slashes.

4. **Run.**
   ```bash
   cd /workspace/diffusion-illusion
   bash vast/run.sh --config runs.yaml --dry-run    # check the config
   bash vast/run.sh --config runs.yaml
   ```
   Ctrl+C in the tmux window saves the current run and stops the batch, same as on Windows.

5. **Copy the results back** to your PC, from the project folder. Run it as often as you like while runs are going:
   ```powershell
   .\vast\pull_results.ps1 -Address 123.45.67.89 -Port 41234
   ```

6. **Destroy the instance** once the results are on your PC. A stopped instance still bills for its disk, and when
   you restart it the GPU may be rented by someone else.

## Notes

- **Interruptible instances** are cheaper but can be stopped at any time. Runs can resume from their `checkpoint.pt`,
  so they work here if you pull results regularly.
- **Blackwell GPUs** (RTX 50xx, B200) need a CUDA 13 build of torch, which needs NVIDIA driver 580+. `setup.sh` picks
  the build and warns if the driver is too old. In vast.ai's filters, set max CUDA version to 13.0+ for those.
  Older GPUs get the CUDA 12.6 build (driver 525+).
- **Setup failed?** Read `/workspace/setup.log`, fix what it says, and run `bash vast/setup.sh` again. Steps that are
  already done get skipped.
- **Hugging Face login:** the default model (SD 1.5) doesn't need one. If you switch to a gated model, run
  `export HF_TOKEN=...` before `setup.sh` / `run.sh`.
