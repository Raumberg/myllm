def get_liger_kernel():
    from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_mistral, apply_liger_kernel_to_qwen2
    apply_liger_kernel_to_llama(
        rope=False,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=False
    )
    apply_liger_kernel_to_mistral(
        rope=False,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=False
    )
    apply_liger_kernel_to_qwen2(
        rope=False,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=False
    )