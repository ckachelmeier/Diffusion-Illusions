#Batch config files: a list of runs to go through in order (see runs.example.yaml)
import gc
import glob
import os
import traceback

import yaml

from .config import ConfigError, RunConfig

BATCH_KEYS = {'repeat'} #Run-entry keys that aren't RunConfig settings


def load_batch(path, overrides=None):
    """Read a batch config file into a list of RunConfigs, in the order they'll run.

    The file is YAML with an optional `defaults` mapping (settings for every run) and a `runs` list. A file that is
    just a list is treated as the runs. Each run entry can use any RunConfig setting, plus:
        repeat: N      Do this entry N times (N separate new runs, or resuming the same folder N times in a row)
        resume: [...]  A list of run folders and/or glob patterns with *, each resumed with the entry's settings
    Precedence, highest first: overrides (command line options) > the run entry > defaults > RunConfig defaults.
    Relative paths are relative to the current directory, like on the command line.
    """
    try:
        with open(path, encoding='utf-8') as file:
            data = yaml.safe_load(file)
    except OSError as error:
        raise ConfigError('Could not read %s: %s' % (path, error))
    except yaml.YAMLError as error:
        raise ConfigError('%s is not valid YAML:\n%s' % (path, error))
    if data is None:
        raise ConfigError('%s is empty' % path)
    if isinstance(data, list):
        defaults, runs = {}, data
    elif isinstance(data, dict):
        unknown = sorted(set(data) - {'defaults', 'runs'})
        if unknown:
            raise ConfigError("%s: unknown top-level key(s) %s. Expected 'defaults' and 'runs'" % (path, ', '.join(map(str, unknown))))
        defaults, runs = data.get('defaults') or {}, data.get('runs')
    else:
        raise ConfigError('%s: expected a mapping with defaults/runs, or a list of runs' % path)
    if not isinstance(defaults, dict):
        raise ConfigError('%s: defaults must be a mapping of settings' % path)
    if not isinstance(runs, list) or not runs:
        raise ConfigError('%s: runs must be a non-empty list' % path)
    for key in BATCH_KEYS | {'resume'}:
        if key in defaults:
            raise ConfigError('%s: %r belongs in a run entry, not in defaults' % (path, key))

    configs = []
    for number, entry in enumerate(runs, 1):
        where = '%s, run %i' % (path, number)
        if not isinstance(entry, dict):
            raise ConfigError('%s: each run must be a mapping of settings, not %r' % (where, entry))
        entry = dict(entry)
        repeat = entry.pop('repeat', 1)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ConfigError('%s: repeat must be a positive whole number, not %r' % (where, repeat))
        resumes = expand_resume(entry.pop('resume', None), where)
        for resume in resumes:
            for _ in range(repeat):
                settings = {**defaults, **entry, **(overrides or {})}
                if resume is not None:
                    settings['resume'] = resume
                config = RunConfig.from_dict(settings, where=where)
                config.check(where)
                configs.append(config)
    return configs


def expand_resume(resume, where):
    #A folder, a glob pattern with *, or a list of either -> list of folders ([None] for a new run)
    if resume is None:
        return [None]
    patterns = [resume] if isinstance(resume, str) else resume
    if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
        raise ConfigError('%s: resume must be a folder or a list of folders, not %r' % (where, resume))
    folders = []
    for pattern in patterns:
        if '*' in pattern:
            matches = sorted(m for m in glob.glob(pattern) if os.path.isdir(m))
            if not matches:
                raise ConfigError('%s: no run folders match %r' % (where, pattern))
            folders += matches
        else:
            folders.append(pattern)
    return folders


def run_batch(configs):
    """Run each config in order. A failed run is reported and skipped; Ctrl+C saves the current run and stops the batch.
    Returns True if every run finished."""
    from .runner import run #Imported here so the config can be checked without loading torch

    outcomes = [] #(config, result or error text)
    stopped = False
    for number, config in enumerate(configs, 1):
        print()
        print('=' * 100)
        print('Run %i of %i: %s' % (number, len(configs), config.describe()))
        print('=' * 100)
        try:
            result = run(config)
        except KeyboardInterrupt:
            print()
            print('Stopped during run %i.' % number)
            outcomes.append((config, 'stopped before finishing'))
            stopped = True
            break
        except Exception as error:
            traceback.print_exc()
            outcomes.append((config, 'FAILED: %s' % error))
            continue
        finally:
            gc.collect()
        outcomes.append((config, result))
        if result.interrupted:
            print()
            print('Batch stopped by Ctrl+C after run %i.' % number)
            stopped = True
            break

    print()
    print('=' * 100)
    print('Batch summary:')
    for number, (config, outcome) in enumerate(outcomes, 1):
        if isinstance(outcome, str):
            print('  %i. %s\n       %s' % (number, config.describe(), outcome))
        else:
            print('  %i. %s\n       steps %i -> %i%s, saved in %r' % (number, config.describe(), outcome.start_step, outcome.end_step,
                                                             ' (interrupted)' if outcome.interrupted else '', outcome.output_folder))
    not_started = len(configs) - len(outcomes)
    if not_started:
        print('  %i run(s) not started.' % not_started)
    return not stopped and all(not isinstance(outcome, str) for _, outcome in outcomes)
