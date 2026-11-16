
"""Run a real pretrained-weight smoke test (no training required)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from kong_encoder import KongEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seconds", type=int, default=2)
    args = parser.parse_args()

    encoder = KongEncoder.from_checkpoint(
        args.checkpoint, repo_path=args.repo, use_adapter=True, backbone_frozen=True
    ).to(args.device)
    encoder.eval()
    audio = torch.zeros(1, 16000 * args.seconds, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        out = encoder(audio)
    expected_frames = args.seconds * 100 + 1
    assert out["features"].shape == (1, expected_frames, 512)
    assert out["pretrained_features"].shape == (1, expected_frames, 512)
    assert all(x.shape == (1, expected_frames, 512) for x in out["branch_features"].values())
    assert all(x.shape == (1, expected_frames, 88) for x in out["piano"].values())
    torch.testing.assert_close(out["features"], out["pretrained_features"], rtol=0, atol=0)
    print("PASS: checkpoint loaded strictly; 4 CRNNs + 2 fusion GRUs captured.")
    print("features:", tuple(out["features"].shape))
    print("piano outputs:", {k: tuple(v.shape) for k, v in out["piano"].items()})
    print("fusion adapter starts as an exact identity residual")


if __name__ == "__main__":
    main()
