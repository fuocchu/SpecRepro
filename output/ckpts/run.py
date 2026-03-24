import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-100 mean and std (per channel)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# ── Transforms ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

val_transform = transforms.Compose([
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
NUM_WORKERS = 0

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
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet32x4(nn.Module):
    """Teacher: ResNet32x4"""
    def __init__(self, architecture: str = "ResNet32x4", top1_accuracy: float = 79.42, num_classes: int = 100):
        super(ResNet32x4, self).__init__()
        self.architecture = architecture
        self.top1_accuracy = top1_accuracy
        self.model = ResNet_CIFAR(
            block=BasicBlock,
            num_blocks=[5, 5, 5],
            width_multiplier=4,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


class ResNet8x4(nn.Module):
    """Student: ResNet8x4"""
    def __init__(self, architecture: str = "ResNet8x4", baseline_top1: float = 72.50, num_classes: int = 100):
        super(ResNet8x4, self).__init__()
        self.architecture = architecture
        self.baseline_top1 = baseline_top1
        self.model = ResNet_CIFAR(
            block=BasicBlock,
            num_blocks=[1, 1, 1],
            width_multiplier=4,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


def dkd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=1.0, beta=1.0):
    batch_size = student_logits.size(0)
    num_classes = student_logits.size(1)
    arange = torch.arange(batch_size, device=student_logits.device)

    # TCKD
    student_probs = F.softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    student_pt = student_probs[arange, labels]
    teacher_pt = teacher_probs[arange, labels]
    student_binary = torch.stack([student_pt, 1.0 - student_pt], dim=1).clamp(min=1e-8)
    teacher_binary = torch.stack([teacher_pt, 1.0 - teacher_pt], dim=1).clamp(min=1e-8)
    tckd = F.kl_div(torch.log(student_binary), teacher_binary, reduction='batchmean') * (T ** 2)

    # NCKD
    mask = torch.zeros(batch_size, num_classes, device=student_logits.device, dtype=torch.bool)
    mask[arange, labels] = True
    neg_inf = torch.full_like(student_logits, float('-inf'))
    student_logits_masked = torch.where(mask, neg_inf, student_logits / T)
    teacher_logits_masked = torch.where(mask, neg_inf, teacher_logits / T)
    teacher_phat = F.softmax(teacher_logits_masked, dim=1)
    student_log_phat = F.log_softmax(student_logits_masked, dim=1)
    student_log_phat = torch.nan_to_num(student_log_phat, nan=0.0, neginf=0.0)
    nckd = F.kl_div(student_log_phat, teacher_phat, reduction='batchmean') * (T ** 2)

    return alpha * tckd + beta * nckd


def total_loss(student_logits, teacher_logits, labels, T=4.0, alpha=1.0, beta=8.0, ce_weight=1.0):
    ce_loss = F.cross_entropy(student_logits, labels)
    dkd = dkd_loss(student_logits, teacher_logits, labels, T=T, alpha=alpha, beta=beta)
    return ce_weight * ce_loss + dkd


# ── Training and Evaluation Functions ─────────────────────────────────────────

def train_one_epoch(student, teacher, loader, optimizer, device, T, alpha, beta,
                    ce_weight, epoch, warmup_epochs, base_lr, debug=False):
    """
    Train student for one epoch using DKD loss with teacher guidance.
    Includes linear warmup for the first warmup_epochs epochs.
    """
    student.train()
    teacher.eval()

    total_loss_val = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if debug and batch_idx >= 2:
            break

        images = images.to(device)
        labels = labels.to(device)

        # Linear warmup: scale lr linearly from 0 to base_lr over warmup_epochs
        if epoch < warmup_epochs:
            # epoch is 0-indexed here
            warmup_factor = (epoch + (batch_idx + 1) / len(loader)) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * warmup_factor

        optimizer.zero_grad()

        # Forward pass
        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)

        # Compute loss
        loss = total_loss(
            student_logits, teacher_logits, labels,
            T=T, alpha=alpha, beta=beta, ce_weight=ce_weight
        )

        loss.backward()
        optimizer.step()

        total_loss_val += loss.item()
        _, predicted = student_logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss_val / (min(len(loader), 2) if debug else len(loader))
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate(model, loader, device):
    """
    Evaluate model on the given loader.
    Returns a dict with 'top1_accuracy'.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    top1 = 100.0 * correct / total
    return {'top1_accuracy': top1}


def main():
    import os

    # ── Config ────────────────────────────────────────────────────────────────
    EPOCHS = 240
    BASE_LR = 0.05
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    T = 4.0
    ALPHA = 1.0
    BETA = 8.0
    CE_WEIGHT = 1.0
    WARMUP_EPOCHS = 20
    LR_DECAY_EPOCHS = [150, 180, 210]
    LR_DECAY_FACTOR = 0.1
    NUM_CLASSES = 100

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Models ────────────────────────────────────────────────────────────────
    teacher = ResNet32x4(num_classes=NUM_CLASSES).to(device)
    student = ResNet8x4(num_classes=NUM_CLASSES).to(device)

    # Load pretrained teacher weights if available
    teacher_ckpt_paths = [
        './teacher_resnet32x4.pth',
        './resnet32x4_cifar100.pth',
        './pretrained/resnet32x4.pth',
    ]
    teacher_loaded = False
    for ckpt_path in teacher_ckpt_paths:
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            if 'model' in state:
                state = state['model']
            elif 'state_dict' in state:
                state = state['state_dict']
            teacher.load_state_dict(state, strict=False)
            print(f"Loaded teacher weights from {ckpt_path}")
            teacher_loaded = True
            break
    if not teacher_loaded:
        print("Warning: No pretrained teacher weights found. Using random initialization.")
        print("For proper DKD results, please provide a pretrained ResNet32x4 teacher.")

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=BASE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=False
    )

    # ── LR Scheduler (MultiStepLR) ────────────────────────────────────────────
    # We handle warmup manually; scheduler handles decay after warmup
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=LR_DECAY_EPOCHS,
        gamma=LR_DECAY_FACTOR
    )

    # ── Debug run: 1 epoch with 2 batches ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running debug training loop (1 epoch, 2 batches)...")
    print("=" * 60)

    # Reset LR for debug
    for param_group in optimizer.param_groups:
        param_group['lr'] = BASE_LR

    train_loss, train_acc = train_one_epoch(
        student=student,
        teacher=teacher,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        T=T,
        alpha=ALPHA,
        beta=BETA,
        ce_weight=CE_WEIGHT,
        epoch=0,
        warmup_epochs=WARMUP_EPOCHS,
        base_lr=BASE_LR,
        debug=True
    )

    print(f"Debug epoch - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # ── Evaluation ────────────────────────────────────────────────────────────
    metrics = evaluate(student, val_loader, device)

    print("\n" + "=" * 60)
    print("Evaluation Results (after debug run - not fully trained):")
    for name, value in metrics.items():
        print(f"METRIC {name}: {value:.2f}")
    print("=" * 60)

    print("\nTRAINING_LOOP: OK")

    # ── Full Training (optional, uncomment to run) ────────────────────────────
    # Uncomment below to run full 240-epoch training
    """
    print("\nStarting full training for 240 epochs...")
    
    # Reset student weights
    student = ResNet8x4(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=BASE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=LR_DECAY_EPOCHS,
        gamma=LR_DECAY_FACTOR
    )
    
    best_acc = 0.0
    for epoch in range(EPOCHS):
        # During warmup, we set lr manually in train_one_epoch
        # After warmup, scheduler handles lr
        if epoch >= WARMUP_EPOCHS:
            # Restore base lr before scheduler step (in case warmup modified it)
            pass
        
        train_loss, train_acc = train_one_epoch(
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            T=T,
            alpha=ALPHA,
            beta=BETA,
            ce_weight=CE_WEIGHT,
            epoch=epoch,
            warmup_epochs=WARMUP_EPOCHS,
            base_lr=BASE_LR,
            debug=False
        )
        
        # Step scheduler after warmup
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            metrics = evaluate(student, val_loader, device)
            top1 = metrics['top1_accuracy']
            if top1 > best_acc:
                best_acc = top1
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{EPOCHS}] LR: {current_lr:.5f} "
                  f"Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
                  f"Val Acc: {top1:.2f}% Best: {best_acc:.2f}%")
    
    print(f"\nFinal Best Top-1 Accuracy: {best_acc:.2f}% (expected ~76.32%)")
    """


if __name__ == "__main__":
    main()