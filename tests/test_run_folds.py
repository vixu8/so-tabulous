"""The leave-one-player-out fold convention (scripts/run_folds.py)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_folds import FOLDS, fold_commands, fold_players, main  # noqa: E402


def test_rotation():
    assert FOLDS == {"00": "05", "01": "00", "02": "01", "03": "02", "04": "03", "05": "04"}


def test_every_player_tested_and_validated_once():
    assert sorted(FOLDS) == sorted(FOLDS.values()) == ["00", "01", "02", "03", "04", "05"]
    for test in FOLDS:
        f = fold_players(test)
        assert len(f["train"]) == 4
        assert len({*f["train"], f["validation"], f["test"]}) == 6


def test_fold_commands_fill_players():
    cmds = fold_commands(["x --val {val} --test {test}", "y run_val{val}_test{test}"], ["0", "03"])
    assert cmds == [("00", "05", ["x --val 05 --test 00", "y run_val05_test00"]),
                    ("03", "02", ["x --val 02 --test 03", "y run_val02_test03"])]


def test_dry_run(capsys):
    assert main(["--dry_run", "--folds", "01", "--cmd", "echo {val} {test}"]) == 0
    assert "$ echo 00 01" in capsys.readouterr().out


def test_failure_stops(capsys):
    assert main(["--folds", "00", "01", "--cmd", "false"]) != 0
    assert "fold test 00" in capsys.readouterr().out
