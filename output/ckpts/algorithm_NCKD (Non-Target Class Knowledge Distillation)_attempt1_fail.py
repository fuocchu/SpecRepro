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