# Kimi K2.5 Model Adapter for vLLM

## Why This Is Needed

Kimi K2.5's `config.json` declares architecture `KimiK25ForConditionalGeneration`,
which is not in vLLM's model registry. Additionally:

1. The config wraps a `DeepseekV3Config` inside a `text_config` field — vLLM's
   DeepseekV3 loader doesn't know to unwrap it.
2. Kimi K2.5 weight files use the prefix `language_model.` on all tensor names,
   while DeepseekV3ForCausalLM expects unprefixed names.

The adapter handles both issues by subclassing `DeepseekV3ForCausalLM`.

## File 1: kimi_k25.py

Create at:
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py`

```python
# SPDX-License-Identifier: Apache-2.0
# vLLM adapter for Kimi K2.5 (moonshotai/Kimi-K2.5)
# Architecture: KimiK25ForConditionalGeneration
# Backbone: DeepseekV3 MoE with MLA attention

from vllm.model_executor.models.deepseek_v2 import DeepseekV3ForCausalLM


class KimiK25ForConditionalGeneration(DeepseekV3ForCausalLM):
    """
    Thin adapter that makes Kimi K2.5 compatible with vLLM's DeepseekV3 loader.

    Two adjustments:
      1. Extracts text_config (DeepseekV3Config) from the outer KimiK25Config
         so vLLM sees the right architecture config.
      2. Strips the 'language_model.' prefix from weight names so they match
         DeepseekV3ForCausalLM's expected parameter names.
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        kimi_cfg = vllm_config.model_config.hf_config
        text_cfg = getattr(kimi_cfg, "text_config", None)
        if text_cfg is not None:
            vllm_config.model_config.hf_config = text_cfg
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        _PREFIX = "language_model."
        _LEN = len(_PREFIX)

        def _strip(weights_iter):
            for name, tensor in weights_iter:
                if name.startswith(_PREFIX):
                    yield name[_LEN:], tensor
                # weights without the prefix are passed through unchanged
                # (handles embed_tokens etc. if they lack the prefix)

        return super().load_weights(_strip(weights))
```

## File 2: registry.py patch

Edit:
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py`

Find the `_VLLM_MODELS` dict (around line 240–260) and add:
```python
"KimiK25ForConditionalGeneration": ("kimi_k25", "KimiK25ForConditionalGeneration"),
```

Place it near other `K` entries or near the DeepSeek entries. Example context:
```python
    # ... existing entries ...
    "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    "KimiK25ForConditionalGeneration": ("kimi_k25", "KimiK25ForConditionalGeneration"),  # ADD THIS
    # ... existing entries ...
```

## Verification

```bash
python3 -c "
from vllm.model_executor.models import ModelRegistry
print('registered:', 'KimiK25ForConditionalGeneration' in ModelRegistry.get_supported_archs())
"
```
