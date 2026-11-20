# so-tabulous
Generative AI final project

Head A stage 1 requires the trained TabCNN checkpoint for the same held-out
player (the TabCNN trainer saves fold 5 under `5/weights.pt` for player `05`):

```bash
python src/headA/train_head_a.py --stage 1 --test_player 05 --tabcnn_checkpoint checkpoints/tabcnn/5/weights.pt --device cuda --resume
python src/headA/train_head_a.py --stage 2 --test_player 05 --device cuda --resume
```

Both TabCNN classifier layers initialize from the checkpoint and remain
trainable. A fresh linear projection maps Kong's 512 features to the pretrained
classifier's input size. Stage 2 restores the projection and classifier from
stage 1. Use a new output directory for this experiment if previous randomly
initialized Head A runs exist; `--resume` restores those previous weights.
