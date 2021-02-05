"""
Environment management tool to instantiate Conda from Flit.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import argparse
from flit_core.config import read_flit_config, toml
from packaging.requirements import Requirement
from packaging.markers import default_environment


def write_env(obj, out):
    try:
        import yaml
        yaml.safe_dump(obj, out)
    except ImportError:
        import json
        json.dump(obj, out, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Manage development environments.')
    parser.add_argument('--python-version', '-V', metavar='VER',
                        help='use Python version VER')
    parser.add_argument('--extra', '-E', metavar='EXTRA', action='append',
                        help='include EXTRA')
    parser.add_argument('--name', '-n', metavar='NAME',
                        help='name Conda environment NAME')
    parser.add_argument('--no-dev', action='store_true', help='skip dev dependencies')
    parser.add_argument('--save-env', metavar='FILE',
                        help='save environment to FILE')
    parser.add_argument('--create-env', action='store_true',
                        help='create Conda environment')
    parser.add_argument('--update-env', action='store_true',
                        help='update Conda environment')
    args = parser.parse_args()
    return args


def load_project():
    tp = Path('pyproject.toml')
    fc = read_flit_config(tp)
    pyp = toml.loads(tp.read_text())
    return pyp, fc


def conda_config(project):
    cfg = project.get('tool', {})
    cfg = cfg.get('envtool', {})
    return cfg.get('conda', {})


def marker_env(args):
    "Get the marker environment"
    env = {}
    env.update(default_environment())
    if args.python_version:
        env['python_version'] = args.python_version
        env['python_full_version'] = args.python_version
    return env


def req_active(env, req):
    if req.marker:
        return req.marker.evaluate(env)
    else:
        return True


def dep_str(cfg, req):
    dep = req.name
    map = cfg.get('rename', {})
    dep = str(map.get(dep, dep))
    if req.specifier:
        dep += f' {req.specifier}'
    return dep


def conda_env(args, pyp, flp):
    cfg = conda_config(pyp)
    conda_extras = cfg.get('extra', {})
    mkenv = marker_env(args)
    name = args.name
    if name is None:
        name = cfg.get('name', 'dev-env')

    env = {'name': str(name)}
    channels = cfg.get('channels')
    if channels:
        env['channels'] = [str(c) for c in channels]

    deps = []
    if args.python_version:
        deps.append(f'python ={args.python_version}')
    elif flp.metadata['requires_python']:
        deps.append('python ' + str(flp.metadata['requires_python']))

    extras = set(['.none'])
    if not args.no_dev:
        extras |= set(['dev', 'doc', 'test'])
    if args.extra == 'all':
        extras |= set(flp.reqs_by_extra.keys())
    elif args.extra:
        extras |= set(args.extra)

    for e in extras:
        for req in flp.reqs_by_extra.get(e, []):
            req = Requirement(req)
            if req_active(mkenv, req):
                deps.append(dep_str(cfg, req))
        for cr in conda_extras.get(e, []):
            deps.append(str(cr))

    env['dependencies'] = deps

    return env


def env_command(env, cmd):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        ef = path / 'environment.yml'
        with ef.open('w') as f:
            write_env(env, f)
        print(cmd, 'environment', ef)
        subprocess.run(['conda', 'env', cmd, '-q', '-f', os.fspath(ef)], check=True)


def main(args):
    py_p, flit_p = load_project()
    env = conda_env(args, py_p, flit_p)
    if args.save_env:
        with open(args.save_env, 'w') as ef:
            write_env(env, ef)
    elif args.create_env:
        env_command(env, 'create')
    elif args.update_env:
        env_command(env, 'update')
    else:
        write_env(env, sys.stdout)


if __name__ == '__main__':
    main(parse_args())
