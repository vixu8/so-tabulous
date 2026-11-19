"""The leave-one-player-out fold convention (scripts/run_folds.py): 5 train + 1 test."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_folds import PLAYERS, fold_commands, fold_players, main  # noqa: E402


def test_every_player_tested_once_and_trains_on_the_rest():
    assert PLAYERS == ["00", "01", "02", "03", "04", "05"]
    for test in PLAYERS:
        f = fold_players(test)
        assert set(f) == {"train", "test"} and f["test"] == test
        assert len(f["train"]) == 5 and test not in f["train"]
    assert fold_players(3)["test"] == "03"
    with pytest.raises(ValueError):
        fold_players("07")


def test_fold_commands_fill_test_player():
    cmds = fold_commands(["x --test {test}", "y run_test{test}"], ["0", "03"])
    assert cmds == [("00", ["x --test 00", "y run_test00"]),
                    ("03", ["x --test 03", "y run_test03"])]


def test_val_placeholder_is_rejected():
    with pytest.raises(ValueError, match="no validation player"):
        fold_commands(["x --val {val} --test {test}"], ["00"])


def test_dry_run(capsys):
    assert main(["--dry_run", "--folds", "01", "--cmd", "echo {test}"]) == 0
    out = capsys.readouterr().out
    assert "$ echo 01" in out and "train 00 02 03 04 05" in out


def test_failure_stops(capsys):
    assert main(["--folds", "00", "01", "--cmd", "false"]) != 0
    assert "fold test 00" in capsys.readouterr().out
