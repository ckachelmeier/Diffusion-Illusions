#Environment setup that has to happen before torch and friends are imported
import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_venv(script_path):
    #The packages this project needs are installed in the project's .venv (see requirements-local.txt). If the script
    #was started with some other Python (e.g. plain "python" on PATH), rerun it with the venv's Python instead.
    venv_dir = os.path.join(PROJECT_DIR, '.venv')
    venv_python = os.path.join(venv_dir, 'Scripts', 'python.exe')
    if os.path.exists(venv_python) and os.path.normcase(os.path.realpath(sys.prefix)) != os.path.normcase(os.path.realpath(venv_dir)):
        print('Relaunching with the project environment:', venv_python, flush=True)
        child = subprocess.Popen([venv_python, os.path.abspath(script_path), *sys.argv[1:]])
        while True:
            try:
                sys.exit(child.wait())
            except KeyboardInterrupt:
                pass #Ctrl+C reaches the relaunched script too, which stops and saves; wait for it to finish


def fix_console_encoding():
    #rp prints unicode characters that the default Windows console encoding can't handle
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    fix_colorama()


def fix_colorama():
    #rp's progress display calls colorama.init() on every update, and each call wraps stdout in another layer.
    #When output is redirected to a file (e.g. python generate.py > log.txt), the layers pile up until Python's recursion
    #limit crashes the run after ~250 steps. Initialize colorama once here and make later calls do nothing.
    import colorama
    colorama.just_fix_windows_console()
    colorama.init = lambda *args, **kwargs: None
