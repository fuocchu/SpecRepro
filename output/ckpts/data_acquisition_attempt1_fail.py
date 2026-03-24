import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-100 mean and std (per channel)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# ── Transforms ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    # Random crop: pad 4 pixels on each side, then crop back to 32×32
    transforms.RandomCrop(32, padding=4),
    # Random horizontal flip for data augmentation
    transforms.RandomHorizontalFlip(),
    # Convert PIL image to tensor (scales [0,255] → [0.0,1.0])
    transforms.ToTensor(),
    # Normalize each channel: (x - mean) / std
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

val_transform = transforms.Compose([
    # No augmentation for validation – just convert and normalize
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

# ── Datasets ─────────────────────────────────────────────────────────────────
train_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True,  download=True, transform=train_transform
)
val_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=val_transform
)

# ── DataLoaders ───────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
NUM_WORKERS = 0  # Use 0 to avoid multiprocessing overhead/timeout issues

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# ── Verification ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Dataset split sizes")
print(f"  Train samples : {len(train_dataset):,}  (expected 50,000)")
print(f"  Val   samples : {len(val_dataset):,}  (expected 10,000)")
print("=" * 60)

# Inspect the first batch from the training split
train_images, train_labels = next(iter(train_loader))
print("\n[Train split] first batch")
print(f"  images shape : {train_images.shape}")
#   Dimensions: (batch_size, channels, height, width)
#     batch_size = 64   – number of samples per mini-batch
#     channels   = 3    – RGB colour channels
#     height     = 32   – image height in pixels
#     width      = 32   – image width in pixels
print(f"  images dtype : {train_images.dtype}")   # expected: torch.float32
print(f"  labels shape : {train_labels.shape}")   # (batch_size,)
print(f"  labels dtype : {train_labels.dtype}")   # expected: torch.int64
print(f"  pixel min/max: {train_images.min():.4f} / {train_images.max():.4f}")

# Inspect the first batch from the validation split
val_images, val_labels = next(iter(val_loader))
print("\n[Val split] first batch")
print(f"  images shape : {val_images.shape}")
#   Same dimension layout as training
print(f"  images dtype : {val_images.dtype}")
print(f"  labels shape : {val_labels.shape}")
print(f"  labels dtype : {val_labels.dtype}")
print(f"  pixel min/max: {val_images.min():.4f} / {val_images.max():.4f}")

# Sanity-check: number of unique classes using dataset targets directly
# (avoids iterating through all batches which causes timeout)
all_labels = torch.tensor(train_dataset.targets)
print(f"\nUnique classes in train set : {all_labels.unique().numel()} (expected 100)")

print("\nDataloader verification complete ✓")