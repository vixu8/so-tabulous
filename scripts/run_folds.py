"""Leave-one-player-out fold runner shared by every model.

Team convention (2026-11-18, same as TabCNN): each of the 6 GuitarSet players is
the test player once and the other 5 players train. There is no validation
player, so every model trains for a fixed budget and keeps its last epoch.

    test   00  01  02  03  04  05
    train  the other five players

Use PLAYERS / fold_players() from here so every model (Head A, Head C, TabCNN,
the decoder) runs on exactly the same splits. `load_split(root, test)` in
src/data/loader.py builds a fold from the manifest, so one features/ folder
serves all six.

Run commands for every fold; {test} is filled in, and commands run from the repo
root, in order, stopping at the first failure:

    python scripts/run_folds.py \\
        --cmd "python src/headA/train_head_a.py --stage 1 --test_player {test}" \\
        --cmd "python src/headA/train_head_a.py --stage 2 --test_player {test}"

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


def fold_players(test_player):
    """{"train": [5 players], "test": t} for one fold."""
    test = str(test_player).zfill(2)
    if test not in PLAYERS:
        raise ValueError(f"unknown player {test!r}; choose from {PLAYERS}")
    return {"train": [p for p in PLAYERS if p != test], "test": test}


def fold_commands(templates, test_players=PLAYERS):
    """[(test, [command strings])] with {test} filled in."""
    out = []
    for test in test_players:
        test = fold_players(test)["test"]
        commands = []
        for t in templates:
            try:
                commands.append(t.format(test=test))
            except KeyError as exc:
                raise ValueError(f"unknown placeholder {exc} in {t!r}: only {{test}} is "
                                 "filled in (there is no validation player)") from None
        out.append((test, commands))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Run commands for every leave-one-player-out fold.",
                                epilog="See the module docstring for the fold table.")
    p.add_argument("--cmd", action="append", required=True,
                   help="command template with {test}; repeat for several")
    p.add_argument("--folds", nargs="+", default=PLAYERS, help="test players to run")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)

    t_all = time.time()
    for test, commands in fold_commands(args.cmd, args.folds):
        print(f"\n=== fold: test {test}, train {' '.join(fold_players(test)['train'])} ===",
              flush=True)
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
