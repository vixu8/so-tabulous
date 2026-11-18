
"""Single-batch training sanity test for Head A."""

from pathlib import Path

import torch

from kong_wrapper.kong_encoder import KongEncoder
from headA.head_a import KongHeadA
from data.guitarset_loader import (
    GuitarSetDataset,
    load_split,
    collate,
)


def main():
    # Paths relative to the repository structure
    src_root = Path(__file__).resolve().parents[1]
    project_root = src_root.parent

    checkpoint = (
        src_root / "kong_wrapper" /
        "checkpoints" / "kong_note_pedal.pth"
    )
    data_root = project_root / "features" / "v1"

    if not (data_root / "manifest.csv").exists():
        raise FileNotFoundError(
            f"GuitarSet data not found: {data_root}\n"
            "Update data_root to your features/v1 folder."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # 1. Load one real GuitarSet example
    splits = load_split(
        data_root,
        val_player="04",
        test_player="05",
    )

    dataset = GuitarSetDataset(
        splits["train"],
        data_root,
        mode="random",
        targets=(),
        crops_per_clip=1,
    )

    batch = collate([dataset[0]])

    # Use 2 seconds to keep the CPU test manageable
    waveform = batch["waveform"][:, :32000].to(device)
    targets = batch["tab"][:, :201].to(device).clone()

    # Ignore any padded frames
    valid = min(
        int(batch["n_valid_frames"][0]),
        targets.shape[1],
    )
    targets[:, valid:] = -100

    print("Waveform:", waveform.shape)
    print("Targets:", targets.shape)
    print("Active string-frames:",
          (targets > 0).sum().item())

    # 2. Load pretrained Kong + adapter + Head A
    encoder = KongEncoder.from_checkpoint(
        checkpoint_path=checkpoint,
        use_adapter=True,
        backbone_frozen=True,
    )

    model = KongHeadA(encoder).to(device)
    model.set_stage(1)
    model.train()

    # 3. Train only the adapter and head
    optimizer = torch.optim.AdamW(
        list(model.encoder.fusion_adapter.parameters())
        + list(model.head.parameters()),
        lr=1e-4,
    )

    assert not any(
        p.requires_grad
        for p in model.encoder.note_model.parameters()
    )

    # 4. A few steps on the SAME batch
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)

        logits = model(waveform)
        assert logits.shape == (1, 201, 6, 21)

        loss = model.loss(logits, targets)
        assert torch.isfinite(loss), "Non-finite loss"

        loss.backward()

        # 5. Verify gradients on the first step
        if step == 0:
            head_grad = model.head.dense[-1].weight.grad

            # Adapter's final projection is zero-initialized.
            # Its earlier layers may have zero gradients
            # on the FIRST step, which is expected.
            adapter_grad = (
                model.encoder.fusion_adapter
                .adapter[-1].weight.grad
            )

            assert head_grad is not None
            assert adapter_grad is not None
            assert torch.isfinite(head_grad).all()
            assert torch.isfinite(adapter_grad).all()

            assert all(
                p.grad is None
                for p in model.encoder.note_model.parameters()
            )

            print(
                "Head gradient norm:",
                head_grad.norm().item()
            )
            print(
                "Adapter gradient norm:",
                adapter_grad.norm().item()
            )

        optimizer.step()

        print(
            f"Step {step + 1} | "
            f"Loss: {loss.item():.4f}"
        )

    print("Single-batch training test passed!")


if __name__ == "__main__":
    main()
