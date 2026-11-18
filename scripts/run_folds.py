"""Leave-one-player-out fold runner shared by every model.

Team convention (agreed 2026-11-17): each of the 6 GuitarSet players is the test
player once, and the validation player is the one just before it, wrapping round:

    test        00  01  02  03  04  05
    validation  05  00  01  02  03  04

The other 4 players train. Use FOLDS / fold_players() from here so every model
(Head A, Head C, TabCNN, the decoder) runs on exactly the same splits.
`load_split(root, val, test)` in src/data/loader.py builds a fold from the
manifest, so one features/ folder serves all six.

Run commands for every fold; {val} and {test} are filled in, and commands run
from the repo root, in order, stopping at the first failure:

    python scripts/run_folds.py \\
        --cmd "python src/headC/train_head_c.py --val_player {val} --test_player {test}" \\
        --cmd "python src/headC/eval_head_c.py --run_dir runs/head_c/stage1_val{val}_test{test}"

    python scripts/run_folds.py --folds 00 03 --cmd "..."     # only some test players
    python scripts/run_folds.py --dry_run --cmd "..."         # print, don't run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

PLAYERS = ["00", "01", "02", "03", "04", "05"]
REPO = Path(__file__).resolve().parents[1]


def validation_player(test_player):
    """The player just before `test_player`, wrapping round (test 00 -> val 05)."""
    test = str(test_player).zfill(2)
    return PLAYERS[(PLAYERS.index(test) - 1) % len(PLAYERS)]


FOLDS = {test: validation_player(test) for test in PLAYERS}   # test player -> validation player


def fold_players(test_player):
    """{"train": [...], "validation": v, "test": t} for one fold."""
    test = str(test_player).zfill(2)
    val = FOLDS[test]
    return {"train": [p for p in PLAYERS if p not in (val, test)], "validation": val, "test": test}


def fold_commands(templates, test_players=PLAYERS):
    """[(test, val, [command strings])] with {val} / {test} filled in."""
    out = []
    for test in test_players:
        test = str(test).zfill(2)
        val = FOLDS[test]
        out.append((test, val, [t.format(val=val, test=test) for t in templates]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Run commands for every leave-one-player-out fold.",
                                epilog="See the module docstring for the fold table.")
    p.add_argument("--cmd", action="append", required=True,
                   help="command template with {val} and {test}; repeat for several")
    p.add_argument("--folds", nargs="+", default=PLAYERS, help="test players to run")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)

    t_all = time.time()
    for test, val, commands in fold_commands(args.cmd, args.folds):
        print(f"\n=== fold: test {test}, validation {val}, "
              f"train {' '.join(fold_players(test)['train'])} ===", flush=True)
        for command in commands:
            print(f"$ {command}", flush=True)
            if args.dry_run:
                continue
            t0 = time.time()
            done = subprocess.run(shlex.split(command), cwd=REPO)
            if done.returncode != 0:
                print(f"failed (exit {done.returncode}) on fold test {test}: {command}")
                return done.returncode
            print(f"(done in {time.time() - t0:.0f} s)", flush=True)
    print(f"\nall {len(args.folds)} folds done in {time.time() - t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
