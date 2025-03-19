import torch
from torch.nn.functional import kl_div, softmax, log_softmax, mse_loss, cosine_similarity, one_hot

def fn_KLDivergence(student_logits, teacher_logits, temperature):
    student_logprobs = log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = softmax(teacher_logits / temperature, dim=-1)
    return kl_div(student_logprobs, teacher_probs, reduction='none', log_target=False) * (temperature ** 2)


def fn_MSE(student_logits, teacher_logits):
    return mse_loss(student_logits, teacher_logits, reduction='none')


def fn_SoftTargetXEntropy(student_logits, teacher_logits, temperature):
    teacher_probs = softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = log_softmax(student_logits / temperature, dim=-1)
    return -(teacher_probs * student_log_probs)


def fn_Slim(student_logits, teacher_logits, temperature, hard_labels):
    student_probs, teacher_probs = softmax(student_logits, dim=-1), softmax(teacher_logits, dim=-1)
    kd_loss = fn_SoftTargetXEntropy(student_logits, teacher_logits, temperature)
    hard_labels = one_hot(hard_labels, num_classes=student_logits.size(-1)).to(device=student_logits.device,
                                                                               dtype=student_logits.dtype)
    filtered_student_probs = hard_labels * student_probs
    filtered_teacher_probs = hard_labels * teacher_probs
    diff = filtered_teacher_probs / torch.clamp(filtered_student_probs, min=1e-9)
    coef = 1 - torch.exp(-diff)
    return coef * kd_loss


def fn_CosSim(student_logits, teacher_logits):
    return 1 - cosine_similarity(student_logits, teacher_logits, dim=-1)


def fn_JensenShannonDiv(student_logits, teacher_logits, temperature):
    teacher_probs = softmax(teacher_logits / temperature, dim=-1)
    student_probs = softmax(student_logits / temperature, dim=-1)
    m = 0.5 * (teacher_probs + student_probs)
    jsd = 0.5 * (kl_div(torch.log(student_probs), m, reduction='none') + kl_div(torch.log(teacher_probs), m,
                                                                                reduction='none'))
    return jsd


def fn_EarthMoverDistance(student_logits, teacher_logits):
    student_probs = softmax(student_logits, dim=-1)
    teacher_probs = softmax(teacher_logits, dim=-1)
    return torch.cdist(student_probs, teacher_probs, p=1)


def fn_AlphaBetaDiv(student_logits, teacher_logits, alpha=1.0, beta=2.0):
    student_probs = softmax(student_logits, dim=-1)
    teacher_probs = softmax(teacher_logits, dim=-1)

    if alpha == beta:
        divergence = (1 / alpha) * (torch.sum(teacher_probs ** alpha) - 1)
    else:
        divergence = (1 / (alpha * (beta - alpha))) * (
                    torch.sum(teacher_probs ** alpha) - torch.sum(teacher_probs ** beta))

    return divergence