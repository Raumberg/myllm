#pragma once

#include <torch/extension.h>

// rstd is now an output parameter for the forward pass
void rms_norm_forward(torch::Tensor &output, torch::Tensor &rstd,
                      const torch::Tensor &input, const torch::Tensor &weight,
                      float eps);

// grad_weight now matches the input type, no longer float*
void rms_norm_backward(torch::Tensor &grad_input, torch::Tensor &grad_weight,
                       const torch::Tensor &grad_output,
                       const torch::Tensor &input, const torch::Tensor &weight,
                       const torch::Tensor &rstd);
