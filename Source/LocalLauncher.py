# Author: Qidong Yang & Jonathan Giezendanner

import sys
from pathlib import Path


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == '__main__':
    # Settings now live in configs/*.yaml; see Source/run.py
    import run

    run.main(sys.argv[1:] or ['--config', str(Path(__file__).parent.parent / 'configs' / 'neus_repro.yaml')])
