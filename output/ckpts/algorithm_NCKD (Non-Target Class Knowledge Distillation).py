import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-100 mean and std (per channel)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# ── Transforms ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    # Random crop: pad 4 pixels on each side, then crop back to 32x32
    transforms.RandomCrop(32, padding=4),
    # Random horizontal flip for data augmentation
    transforms.RandomHorizontalFlip(),
    # Convert PIL image to tensor (scales [0,255] -> [0.0,1.0])
    transforms.ToTensor(),
    # Normalize each channel: (x - mean) / std
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

val_transform = transforms.Compose([
    # No augmentation for validation - just convert and normalize
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
#     batch_size = 64   - number of samples per mini-batch
#     channels   = 3    - RGB colour channels
#     height     = 32   - image height in pixels
#     width      = 32   - image width in pixels
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

print("\nDataloader verification complete [OK]")

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(nn.Module):
    """ResNet for CIFAR with configurable depth and width multiplier."""

    def __init__(self, block, num_blocks, width_multiplier=1, num_classes=100):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16 * width_multiplier

        self.conv1 = nn.Conv2d(3, 16 * width_multiplier, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * width_multiplier)

        self.layer1 = self._make_layer(block, 16 * width_multiplier, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32 * width_multiplier, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width_multiplier, num_blocks[2], stride=2)

        self.linear = nn.Linear(64 * width_multiplier * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet32x4(nn.Module):
    """
    Teacher: ResNet32x4
    ResNet with depth=32 and width multiplier=4, for CIFAR-100.
    Pre-trained on CIFAR-100 achieving 79.42% top-1 accuracy.
    """

    def __init__(self, architecture: str = "ResNet32x4", top1_accuracy: float = 79.42, num_classes: int = 100):
        super(ResNet32x4, self).__init__()
        self.architecture = architecture
        self.top1_accuracy = top1_accuracy

        # ResNet-32 has [5, 5, 5] blocks (depth = 1 + 3*5*2 + 1 = 32... actually 6n+2 where n=5)
        # depth=32 => n=5 => [5, 5, 5] blocks per stage
        # width_multiplier=4
        self.model = ResNet_CIFAR(
            block=BasicBlock,
            num_blocks=[5, 5, 5],
            width_multiplier=4,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


# ── Smoke Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    teacher = ResNet32x4(architecture="ResNet32x4", top1_accuracy=79.42)
    teacher.eval()

    B = 4
    dummy_input = torch.randn(B, 3, 32, 32)
    with torch.no_grad():
        output = teacher(dummy_input)

    print(f"SHAPE_CHECK Teacher: ResNet32x4: input={tuple(dummy_input.shape)} output={tuple(output.shape)}")
    assert output.shape == (B, 100), f"Expected output shape ({B}, 100), got {output.shape}"

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, width_multiplier=1, num_classes=100):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16 * width_multiplier

        self.conv1 = nn.Conv2d(3, 16 * width_multiplier, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * width_multiplier)

        self.layer1 = self._make_layer(block, 16 * width_multiplier, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32 * width_multiplier, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width_multiplier, num_blocks[2], stride=2)

        self.linear = nn.Linear(64 * width_multiplier * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet8x4(nn.Module):
    """
    Student: ResNet8x4
    ResNet with depth=8 and width multiplier=4, for CIFAR-100.
    Baseline (vanilla) top-1 accuracy: 72.50%.
    """

    def __init__(self, architecture: str = "ResNet8x4", baseline_top1: float = 72.50, num_classes: int = 100):
        super(ResNet8x4, self).__init__()
        self.architecture = architecture
        self.baseline_top1 = baseline_top1

        # ResNet-8 has depth=8 => 6n+2=8 => n=1 => [1, 1, 1] blocks per stage
        # width_multiplier=4
        self.model = ResNet_CIFAR(
            block=BasicBlock,
            num_blocks=[1, 1, 1],
            width_multiplier=4,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


# ── Smoke Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    student = ResNet8x4(architecture="ResNet8x4", baseline_top1=72.50)
    student.eval()

    B = 4
    dummy_input = torch.randn(B, 3, 32, 32)
    with torch.no_grad():
        output = student(dummy_input)

    print(f"SHAPE_CHECK Student: ResNet8x4: input={tuple(dummy_input.shape)} output={tuple(output.shape)}")
    assert output.shape == (B, 100), f"Expected output shape ({B}, 100), got {output.shape}"

import torch
import torch.nn as nn
import torch.nn.functional as F


def tckd_loss(student_logits: torch.Tensor,
              teacher_logits: torch.Tensor,
              labels: torch.Tensor,
              T: float = 4.0) -> torch.Tensor:
    """
    TCKD: Target Class Knowledge Distillation
    
    Binary logit distillation for the target class.
    KL(b^T || b^S) where b = [p_t, p_{\\t}] are binary probabilities
    of target vs. all non-target classes.
    
    p_t = exp(z_t) / sum_j exp(z_j)  (i.e., the softmax probability of the target class)
    p_{\\t} = 1 - p_t
    
    Args:
        student_logits: (B, C) student logits
        teacher_logits: (B, C) teacher logits
        labels: (B,) ground truth class indices
        T: temperature for distillation
    
    Returns:
        TCKD_loss: scalar tensor
    """
    batch_size = student_logits.size(0)
    
    # Compute softmax probabilities at temperature T
    # Shape: (B, C)
    student_probs = F.softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    
    # Extract target class probabilities p_t
    # Shape: (B,)
    student_pt = student_probs[torch.arange(batch_size), labels]
    teacher_pt = teacher_probs[torch.arange(batch_size), labels]
    
    # Compute non-target probabilities p_{\\t} = 1 - p_t
    # Shape: (B,)
    student_pnt = 1.0 - student_pt
    teacher_pnt = 1.0 - teacher_pt
    
    # Stack into binary distributions: shape (B, 2)
    # b = [p_t, p_{\\t}]
    student_binary = torch.stack([student_pt, student_pnt], dim=1)  # (B, 2)
    teacher_binary = torch.stack([teacher_pt, teacher_pnt], dim=1)  # (B, 2)
    
    # Clamp to avoid log(0)
    student_binary = student_binary.clamp(min=1e-8)
    teacher_binary = teacher_binary.clamp(min=1e-8)
    
    # KL divergence: KL(teacher || student) = sum(teacher * log(teacher / student))
    # Using F.kl_div which expects log-probabilities for input and probabilities for target
    # F.kl_div(input, target) = target * (log(target) - input)
    # So input = log(student_binary), target = teacher_binary
    kl = F.kl_div(
        torch.log(student_binary),
        teacher_binary,
        reduction='batchmean'
    )
    
    # Scale by T^2 as is standard in knowledge distillation
    loss = kl * (T ** 2)
    
    return loss


# ── Smoke Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    torch.manual_seed(42)
    
    B = 8
    C = 100
    T = 4.0
    
    student_logits = torch.randn(B, C)
    teacher_logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))
    
    loss = tckd_loss(student_logits, teacher_logits, labels, T=T)
    
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"KL divergence should be non-negative, got {loss.item()}"
    
    print(f"ALGO_CHECK TCKD (Target Class Knowledge Distillation): OK  output_shape={loss.shape}")
    print(f"  loss value: {loss.item():.6f}")

import torch
import torch.nn as nn
import torch.nn.functional as F


def nckd_loss(student_logits: torch.Tensor,
              teacher_logits: torch.Tensor,
              labels: torch.Tensor,
              T: float = 4.0) -> torch.Tensor:
    """
    NCKD: Non-Target Class Knowledge Distillation
    
    Multi-category logit distillation among non-target classes.
    KL(p_hat^T || p_hat^S) where:
        p_hat_i = exp(z_i) / sum_{j != t} exp(z_j)  for i != t
    
    Args:
        student_logits: (B, C) student logits
        teacher_logits: (B, C) teacher logits
        labels: (B,) ground truth class indices
        T: temperature for distillation
    
    Returns:
        NCKD_loss: scalar tensor
    """
    batch_size, num_classes = student_logits.shape
    
    # Scale logits by temperature
    student_logits_T = student_logits / T
    teacher_logits_T = teacher_logits / T
    
    # Create a mask to zero out the target class
    # mask shape: (B, C), True for target classes
    arange = torch.arange(batch_size, device=student_logits.device)
    
    mask = torch.zeros(batch_size, num_classes, device=student_logits.device, dtype=torch.bool)
    mask[arange, labels] = True  # True at target positions
    
    # Apply mask: set target class logit to -inf
    neg_inf = torch.full_like(student_logits_T, float('-inf'))
    
    student_logits_masked = torch.where(mask, neg_inf, student_logits_T)
    teacher_logits_masked = torch.where(mask, neg_inf, teacher_logits_T)
    
    # Compute softmax over non-target classes only
    # p_hat_i = exp(z_i) / sum_{j != t} exp(z_j)
    # The target class will have probability 0 after softmax with -inf logit
    student_phat = F.softmax(student_logits_masked, dim=1)  # (B, C)
    teacher_phat = F.softmax(teacher_logits_masked, dim=1)  # (B, C)
    
    # Use log_softmax for numerical stability
    student_log_phat = F.log_softmax(student_logits_masked, dim=1)  # (B, C)
    
    # The target class position will have:
    # - teacher_phat = 0
    # - student_log_phat = -inf (or nan after softmax normalization)
    # We need to handle 0 * (-inf) = nan -> should be 0
    # Replace nan in student_log_phat with 0 (these correspond to target class positions)
    student_log_phat = torch.nan_to_num(student_log_phat, nan=0.0, neginf=0.0)
    
    # KL(teacher || student) = sum(teacher * (log(teacher) - log(student)))
    # Using F.kl_div with reduction='batchmean'
    # F.kl_div(log_input, target) computes target * (log(target) - log_input)
    kl = F.kl_div(
        student_log_phat,
        teacher_phat,
        reduction='batchmean'
    )
    
    # Scale by T^2 as standard in knowledge distillation
    loss = kl * (T ** 2)
    
    return loss


# ── Smoke Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    torch.manual_seed(42)
    
    B = 8
    C = 100
    T = 4.0
    
    student_logits = torch.randn(B, C)
    teacher_logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))
    
    loss = nckd_loss(student_logits, teacher_logits, labels, T=T)
    
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"KL divergence should be non-negative, got {loss.item()}"
    
    print(f"ALGO_CHECK NCKD (Non-Target Class Knowledge Distillation): OK  output_shape={loss.shape}")
    print(f"  loss value: {loss.item():.6f}")