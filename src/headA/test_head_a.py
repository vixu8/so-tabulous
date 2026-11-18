
import torch

from kong_wrapper.kong_encoder import KongEncoder
from headA.head_a import KongHeadA


def test_head_a():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load Kong with the trainable fusion adapter
    encoder = KongEncoder.from_checkpoint(
        checkpoint_path="kong_wrapper/checkpoints/kong_note_pedal.pth",
        use_adapter=True,
        backbone_frozen=True,
    )

    model = KongHeadA(encoder).to(device)
    model.set_stage(1)
    model.eval()

    # Test with 2 seconds of synthetic audio
    x = torch.randn(1, 32000, device=device)

    with torch.no_grad():
        logits = model(x)

    # Verify output shape
    assert logits.shape == (1, 201, 6, 21)

    # Verify Stage 1 freezes Kong
    assert not any(
        p.requires_grad
        for p in model.encoder.note_model.parameters()
    )

    # Verify adapter + head remain trainable
    assert all(
        p.requires_grad
        for p in model.encoder.fusion_adapter.parameters()
    )
    assert all(
        p.requires_grad
        for p in model.head.parameters()
    )

    print("Stage 1 checks passed!")

    # Verify Stage 2 unfreezes Kong
    model.set_stage(2)

    assert all(
        p.requires_grad
        for p in model.encoder.note_model.parameters()
    )

    print("Stage 2 checks passed!")
    print("Head A connected successfully!")


if __name__ == "__main__":
    test_head_a()
