from transformers import Trainer
from transformers import AutoModelForCausalLM

from collections import defaultdict
from typing import Literal, Dict

import torch

from src.utils.model import prepare_ref_model_for_deepspeed
from src.utils.attn import *
from src.utils.losses import *

class DistillationTrainer(Trainer):
    def __init__(self, 
                 teacher_model: AutoModelForCausalLM, 
                 distill_loss: str, 
                 temperature: float, 
                 alpha: float, 
                 arguments: dict,
                 *args, 
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.teacher_model = prepare_ref_model_for_deepspeed(teacher_model, self.accelerator)
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_loss_fn = self.select_distillation_loss(distill_loss)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.args = arguments

    def select_distillation_loss(self, loss_type):
        distillation_losses = {
            "kl_divergence": fn_KLDivergence,
            "mse": fn_MSE,
            "soft_cross_entropy": fn_SoftTargetXEntropy,
            "cosine_similarity": fn_CosSim,
            "jensen_shannon": fn_JensenShannonDiv,
            "earth_mover_distance": fn_EarthMoverDistance,
            "alpha_beta_divergence": fn_AlphaBetaDiv,
            "slim": fn_Slim
        }
        if loss_type not in distillation_losses:
            raise ValueError(f"Unsupported distillation loss type: {loss_type}")
        return distillation_losses[loss_type]

    def compute_loss(self, model, inputs, return_outputs=False):
        attention_mask = inputs['attention_mask'][..., 1:].contiguous()
        hard_labels = inputs['labels'][..., 1:].contiguous()
        kd_coef = None

        student_outputs = model(**inputs)
        student_logits = student_outputs.logits[..., :-1, :].contiguous()

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits[..., :-1, :].contiguous()

        # Calculate the distillation loss
        if self.distillation_loss_fn in [fn_KLDivergence,
                                        fn_SoftTargetXEntropy,
                                        fn_JensenShannonDiv]:
            distillation_loss = self.distillation_loss_fn(student_logits, teacher_logits, self.temperature)
        elif self.distillation_loss_fn in [fn_Slim]:
            distillation_loss = self.distillation_loss_fn(student_logits, teacher_logits, self.temperature,
                                                            hard_labels)
        else:
            distillation_loss = self.distillation_loss_fn(student_logits, teacher_logits)

        # Apply hard labels coefficient to loss if needed
        if self.args.apply_hard_labels_coef and not self.distillation_loss_fn in [fn_Slim]:
            kd_coef = hard_labels_coefficient(student_logits, teacher_logits, self.temperature, hard_labels)
            distillation_loss = kd_coef * distillation_loss

        # We need to calculate distill losses only on non -100 labels
        distillation_loss = apply_hard_labels_mask(distillation_loss, hard_labels)

        # We need to calculate distill losses only on attending tokens
        distillation_loss = apply_attention_mask(distillation_loss, attention_mask)

        # CLM loss - already computed by model
        sft_loss = student_outputs.loss

        # Total loss computation
        # loss = self.alpha * distillation_loss + (1 - self.alpha) * sft_loss
        if self.args.dont_learn_clm:
            loss = self.alpha * distillation_loss
        else:
            loss = self.alpha * distillation_loss + sft_loss

        metrics = {
            "distillation_loss": distillation_loss.cpu(),
            "sft_loss": sft_loss.cpu()
        }
        if kd_coef is not None:
            kd_coef = apply_attention_mask(kd_coef, attention_mask)  # for logging
            metrics['kd_coef'] = kd_coef.cpu()

        self.store_metrics(metrics, train_eval='train' if self.model.training else 'eval')

        if return_outputs:
            return loss, student_outputs
        return loss

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)