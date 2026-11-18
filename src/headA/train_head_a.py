
from torch.utils.data import DataLoader

from data.loader import (
    GuitarSetDataset,
    load_split,
    collate,
)

DATA_ROOT = "features/v1"

splits = load_split(
    DATA_ROOT,
    val_player="04",
    test_player="05",
)

train_dataset = GuitarSetDataset(
    splits["train"],
    DATA_ROOT,
    mode="random",
    targets=(),
    crops_per_clip=4,
)

val_dataset = GuitarSetDataset(
    splits["validation"],
    DATA_ROOT,
    mode="windows",
    targets=(),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    collate_fn=collate,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate,
)

model = KongHeadA(encoder).to(DEVICE)



def run_epoch(loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0

    for batch in loader:
        waveform = batch["waveform"].to(DEVICE)
        targets = batch["tab"].to(DEVICE).clone()
        n_valid = batch["n_valid_frames"].to(DEVICE)

        # Ignore frames corresponding to padded audio.
        frame_ids = torch.arange(
            targets.shape[1], device=DEVICE
        )
        invalid = frame_ids[None, :] >= n_valid[:, None]
        targets.masked_fill_(invalid.unsqueeze(-1), -100)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(waveform)
            loss = model.loss(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



def train_stage(stage, epochs, optimizer, filename):
    model.set_stage(stage)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = run_epoch(train_loader, optimizer)
        val_loss = run_epoch(val_loader)

        print(
            f"Stage {stage} | Epoch {epoch + 1} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)


# STAGE 1: Train adapter and head, freeze Kong
model.set_stage(1)

optimizer1 = torch.optim.AdamW(
    list(model.encoder.fusion_adapter.parameters())
    + list(model.head.parameters()),
    lr=1e-4,
)

train_stage(
    stage=1,
    epochs=5,
    optimizer=optimizer1,
    filename="head_a_stage1.pt",
)


# STAGE 2: Load best Stage 1 model, fine-tune everything
model.load_state_dict(
    torch.load(
        "head_a_stage1.pt",
        map_location=DEVICE,
        weights_only=True,
    )
)

model.set_stage(2)

optimizer2 = torch.optim.AdamW([
    {
        "params": model.encoder.note_model.parameters(),
        "lr": 1e-5,
    },
    {
        "params": model.encoder.fusion_adapter.parameters(),
        "lr": 3e-5,
    },
    {
        "params": model.head.parameters(),
        "lr": 3e-5,
    },
])

train_stage(
    stage=2,
    epochs=5,
    optimizer=optimizer2,
    filename="head_a_stage2.pt",
)
