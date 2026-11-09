# Model Checkpoints

This directory stores training checkpoints for model recovery and experimentation.

## Checkpoint Strategy

- Saved every 5 epochs during training
- Best model saved when validation accuracy improves
- Allows resuming interrupted training
- Enables model version comparison

## File Naming Convention

```
checkpoints/
├── epoch_05.h5         # Checkpoint at epoch 5
├── epoch_10.h5         # Checkpoint at epoch 10
├── epoch_15.h5         # Checkpoint at epoch 15
├── best_model.h5       # Best validation accuracy
└── last_checkpoint.h5  # Most recent checkpoint
```

## Using Checkpoints

### Resume Training

```python
from tensorflow.keras.models import load_model

# Load last checkpoint
model = load_model('models/checkpoints/last_checkpoint.h5')

# Continue training
model.fit(X_train, y_train, initial_epoch=15, epochs=50)
```

### Evaluate Checkpoint

```python
model = load_model('models/checkpoints/epoch_10.h5')
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Checkpoint accuracy: {accuracy:.2%}")
```

## Auto-Checkpoint Setup

Training scripts use ModelCheckpoint callback:

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'models/checkpoints/epoch_{epoch:02d}.h5',
    save_best_only=False,
    save_freq='epoch',
    period=5
)
```

## Storage

- Each checkpoint: ~2-5 MB
- Keep best 5 checkpoints + best model
- Auto-cleanup of old checkpoints
