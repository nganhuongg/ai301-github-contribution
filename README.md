# Contribution 1: Support for Sparse MoE models like Camelidae and Sparsetral

**Contribution Number:** 1  

**Student:** Ngan Huong Nguyen

**Issue:** [Support for Sparse MoE models like Camelidae and Sparsetral](https://github.com/ggml-org/llama.cpp/issues/5365)  

**Status:** Phase III - In Progress

---

## Why I Chose This Issue

I chose this issue because it is closely related to areas that I want to explore deeply, including large language model inference, model architectures, and systems programming. I am currently learning NLP, transformers, and modern deep learning systems, so contributing to llama.cpp provides an opportunity to understand how production-grade inference engines support different model architectures.

This issue also matches my technical background. I am comfortable with C++ and Python and have experience reading and modifying machine learning codebases. Through this contribution, I hope to learn how model conversion, tensor mapping, and inference support are implemented in real-world LLM infrastructure.

---

## Understanding the Issue

### Problem Description

Currently, the project supports many standard Mixture-of-Expert (MoE) architectures, but does not provide explicit support for Sparse MoE architectures. These architectures combine dense base model computation with routed lightweight adapter experts, requiring different model conversion, metadata, and runtime handling compared to traditional MoE implementations.

Paper: [Parameter-Efficient Sparsity Crafting from Dense to Mixture-of-Experts for Instruction Tuning on General Tasks](http://arxiv.org/pdf/2401.02731 ) 

### Expected Behavior

Support for sparse adapter experts: a dense base model remains active while token routing selects lightweight LoRA-like expert MLPs.

### Current Behavior

Support for Sparse MoE is not available, preventing these models from being used through the standard llama.cpp workflow.

### Affected Components

GGUF Conversion:
- `convert_hf_to_gguf.py`, `conversion/`, `gguf-py/gguf/`
- The converter needs to recognize the architecture, normalize its config, map its custom tensor names and write GGUF metadata before runtime code can be exercised.

GGUF architecture, tensor, and metadata names:
- `gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py`, `src/llama-arch.h`, `src/llama-arch.cpp`
- These files define the contract between converted files and C++ loading code. Sparse adapter experts need stable tensor names and metadata for expert count, top-k routing, adapter dimensions, and any weighting behavior; otherwise conversion may produce tensors the loader cannot find or the graph cannot interpret.

Runtime loading:
- `src/llama-model.cpp`, `src/models/`
- This layer decides which model class handles the GGUF, validates hparams, creates expected tensors, and reports missing or mismatched tensors. Fixing sparse LoRA MoE support requires the loader to distinguish dense base FFN tensors from routed adapter-expert tensors and to create both sets with the correct shapes.

MoE graph construction:
- `src/llama-graph.cpp`, `build_moe_ffn()`
- Sparsetral-like models should reuse these routing primitives, but their adapter experts appear to be added on top of the dense MLP instead of replacing the whole FFN like many existing MoE models.

Adapter:
- `convert_lora_to_gguf.py`, `src/llama-adapter.h`, `src/llama-adapter.cpp`, `src/llama-context.cpp`
- This code shows how llama.cpp represents low-rank A/B weights, applies scaling, and matches adapter weights to base tensors. Sparse MoE adapters are LoRA-like expert weights selected per token, so the implementation should reuse or mirror these semantics rather than inventing an unrelated adapter mechanism.

Quantization:
- `src/llama-quant.cpp`
- This matters because routed experts are likely stored as per-expert 3D tensors or paired low-rank expert tensors. If the tensor categories are wrong, conversion might work but quantization could skip important tensors, quantize router tensors incorrectly, or produce a GGUF that no longer loads or matches the reference model.

### Defined Scope
The goal of this contribution is not to support only a single model, but to investigate support for the Sparse LoRA-MoE architecture family described in the PESC paper. [Sparsetral](https://huggingface.co/serpdotai/sparsetral-16x7B-v2/tree/main) will be used as the initial investigation target because its public artifacts (configuration, custom modeling code, and checkpoints) make the architecture easier to inspect and reproduce. [Camelidae](https://huggingface.co/hywu/Camelidae-8x13B) will be used later to validate whether the implementation generalizes across models from the same architecture family.

---

## Reproduction Process

### Environment Setup

I installed requirements for the project and found out some version mismatches due to my use of Python 3.13. That caused repeated install failures because llama.cpp requirements pin packages like numpy~=1.26.4, which do not have suitable wheels for Python 3.13. Pip tried to build NumPy from source, then failed because no C/C++ compiler was installed. The fix was to use Python 3.12.10 and recreate the venv with that version. I also needed to install PyTorch version for CPU individually because the project requirements require platform_machine == "s390x".

### Steps to Reproduce

1. Download the Sparsetral model: Since the initial goal is to determine whether the current conversion pipeline recognizes and supports the Sparsetral architecture, downloading the entire model checkpoint is not necessary at this stage. Therefore, to reproduce the problem, I only downloaded `config.json`, `tokenizer_config.json`, `tokenizer.model`, `modeling_sparsetral.py`, `model.safetensors.index.json` and `configuration_sparsetral.py`

2. Create a local model directory and place the downloaded files inside it.

   ````bash
   mkdir -p models/sparsetral-minimal`

3. Attempt to run the standard Hugging Face to GGUF conversion pipeline

`python convert_hf_to_gguf.py models/sparsetral-minimal --outfile sparsetral.gguf` 

4. Converter can read config, load architecture name, and correctly define model as `modeling_sparsetral.MistralForCausalLM`. It fails at architecture support.

### Reproduction Evidence

- **Commit showing reproduction:** [Link](https://github.com/ggml-org/llama.cpp/commit/7f2954dcb64ac5a1ab2f7e8c21b93aa5feca6560)
- **Screenshots/logs:** 
<img width="493" height="175" alt="image" src="https://github.com/user-attachments/assets/c690bf3e-ce34-4c1d-813b-7237f82e1dcc" />

- **My findings:** My initial hypothesis was that the conversion pipeline might fail to recognize the Sparsetral architecture. However, reproduction showed that the converter successfully loads the configuration and identifies the architecture as `modeling_sparsetral.MistralForCausalLM`. 

The failure occurs at the architecture support stage:

`ERROR: Model modeling_sparsetral.MistralForCausalLM is not supported`

This suggests that the issue is not basic architecture recognition, but rather the absence of a registered conversion path for Sparsetral models within the Hugging Face → GGUF conversion pipeline.

---

## Solution Approach

### Analysis

After reproducing the issue, I found that the converter can read the local Sparsetral model configuration, but fails when it tries to match the model architecture to a supported converter implementation. This suggests that the first root cause is the absence of a registered conversion path for Sparsetral in the Hugging Face to GGUF conversion pipeline. The converter identifies the architecture as `modeling_sparsetral.MistralForCausalLM`, but does not know which existing converter class or architecture mapping should handle it.

### Proposed Solution

I will first investigate whether Sparsetral can reuse an existing Mistral or Mixtral-style converter path with modifications, or whether it needs a dedicated converter class. The first fix should be narrowly scoped to architecture recognition and conversion support before attempting runtime inference or quantization.

If Sparsetral’s base tensors match an existing Mistral-style layout, I will reuse the existing conversion logic as much as possible and only add the missing mappings for Sparsetral-specific router and sparse adapter expert tensors.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** The current converter does not support Sparsetral. It can identify the model architecture, but fails because `modeling_sparsetral.MistralForCausalLM` is not registered or mapped to a supported converter path.

**Match:** I will study similar model conversion patterns in the codebase, especially:

- Existing Mistral conversion logic
- Existing Mixtral or MoE conversion logic
- Tensor mapping rules for MoE expert weights
- Architecture registration patterns in the GGUF conversion pipeline

**Plan:** 
1. Inspect how `convert_hf_to_gguf.py` selects a converter class from the Hugging Face model architecture.
2. Find where supported architecture names are registered.
3. Compare Sparsetral’s `config.json` and `modeling_sparsetral.py` against existing Mistral/Mixtral-style models.
4. Add architecture recognition for `modeling_sparsetral.MistralForCausalLM`.
5. Add or extend tensor mappings for Sparsetral-specific router and sparse expert tensors if needed.
6. Run the converter again on the minimal Sparsetral files. If architecture recognition succeeds, continue investigating the next failure point, likely tensor mapping or missing weight files.

**Implement:** https://github.com/nganhuongg/llama.cpp/tree/fix-issue-5365

**Review:** 
Before submitting a PR, I will verify that:

- [ ] The issue is not already being addressed by another open PR.
- [ ] The implementation is limited to a single feature (Sparse MoE architecture support) and does not include unrelated changes.
- [ ] The solution follows existing llama.cpp architecture registration and converter patterns whenever possible.
- [ ] No unnecessary third-party dependencies, files, or frameworks are introduced.
- [ ] New code follows the project's coding guidelines (naming conventions, formatting, and existing code style).
- [ ] The implementation remains cross-platform and does not introduce platform-specific assumptions.
- [ ] I can explain every line of code that I submit.
- [ ] Any AI assistance was limited to code navigation, explanation, debugging support, and drafting ideas; the final implementation and reasoning are manually reviewed and understood.
- [ ] Existing functionality for supported models is not broken.
- [ ] Relevant conversion and loading tests have been executed successfully.
- [ ] The change is focused on CPU support only unless additional backend support is explicitly required.
- [ ] Commit messages are clear and follow project conventions.
- [ ] The PR description clearly explains the problem, implementation approach, and validation results.

**Evaluate:** I will verify the fix by:

- Running the converter on the minimal Sparsetral reproduction directory.
- Confirming that the previous architecture support error no longer occurs.
- Checking whether the converter progresses to the next expected stage.
- Running existing relevant conversion or architecture tests if available.
- Later, when full model files are available, testing full GGUF conversion and model loading.

---

## Testing Strategy

### Unit Tests

- [ ] Test case 1: [Description]
- [ ] Test case 2: [Description]
- [ ] Test case 3: [Description]

### Integration Tests

- [ ] Integration scenario 1
- [ ] Integration scenario 2

### Manual Testing

[What you tested manually and results]

---

## Implementation Notes

### Week 3 Progress

This week, I moved from reproduction into implementation preparation. Since `llama.cpp` is a large codebase with many conversion, GGUF, and runtime files, I first traced the converter failure to understand the smallest safe implementation path.

#### Navigating Error
- Reproduced the issue again using a minimal Sparsetral model directory.
- Confirmed that `get_model_architecture()` successfully identifies the architecture as `modeling_sparsetral.MistralForCausalLM`.
- Located the failure point in `convert_hf_to_gguf.py`: the converter raises an error when `get_model_class()` cannot find a supported conversion path for the architecture (Line 233 -- 235).
- **Understanding Codebase**: `convert_hf_to_gguf.py` loaded hyperparameters from chosen model's `config.json` and passed them through shared functions and classes from `conversion` package. Two related functions are `get_model_architecture(hparams, model_type)` from `conversion/base.py` and `get_model_class(model_architecture)` from `conversion/__init__.py`. `get_model_class()` decides whether llama.cpp has a converter registered for that exact architecture string. In this case, `get_model_class(model_architecture)` tries to find the converter class in `TEXT_MODEL_MAP` but finds nothing.

#### Sparsetral Architecture
- Sparsetral is MoE-style, but it is not Mixtral-style full MoE. It is a Mistral-style dense model with routed adapter experts added inside the MLP.
- This distinction is important for conversion support. Routing Sparsetral through a standard `MixtralForCausalLM` conversion path would likely be incorrect because Mixtral expects full expert FFN tensors. In contrast, Sparsetral contains standard Mistral FFN tensors together with additional Sparse MoE components such as router and adapter expert tensors (`adapter_down`, `adapter_up`, etc.).

As a result, Sparsetral may require either:
- a dedicated converter implementation, or
- an extension of the existing Mistral/LLaMA conversion pipeline to handle routed adapter experts and their associated metadata.

#### Investigating Converter Assumptions

I compared Sparsetral's configuration against the assumptions used by the existing LLaMA/Mistral conversion pipeline. The dense base transformer configuration appears compatible with the existing `LlamaModel` converter:

| Field | Status |
|---------|---------|
| `hidden_size` | Supported |
| `intermediate_size` | Supported |
| `num_hidden_layers` | Supported |
| `num_attention_heads` | Supported |
| `num_key_value_heads` | Supported |
| `max_position_embeddings` | Supported |
| `rms_norm_eps` | Supported |
| `rope_theta` | Supported |
| `vocab_size` | Supported |

This suggests that Sparsetral's base architecture is largely compatible with existing Mistral/LLaMA conversion logic.

However, Sparsetral introduces several custom Sparse MoE adapter fields:

| Field | Status | Notes |
|---------|---------|---------|
| `num_experts` | Partially supported | Existing converter recognizes expert count |
| `topk` | Not mapped | Existing code expects `num_experts_per_tok`, `num_experts_per_token`, or `top_k_experts` |
| `adapter_dim` | Not supported | Adapter bottleneck dimension |
| `moe_scaling` | Not supported | Adapter output scaling |
| `moe_dtype` | Likely unsupported | Implementation-specific dtype |
| `model_type: sparsetral` | Unsupported | No dedicated converter path |
| `architectures: modeling_sparsetral.MistralForCausalLM` | Unsupported | Current conversion failure |

The issue is likely not caused by the dense transformer architecture itself. Instead, the missing support appears to be related to Sparsetral-specific Sparse MoE adapter metadata and tensor mappings. 

Current GGUF MoE support primarily targets full FFN experts. Existing tensor mappings cover structures such as:

```py
model.layers.{bid}.block_sparse_moe.gate
model.layers.{bid}.block_sparse_moe.experts.w1
model.layers.{bid}.block_sparse_moe.experts.w2
model.layers.{bid}.block_sparse_moe.experts.w3
```
These correspond to full expert feed-forward networks used by models such as Mixtral and PhiMoE. In contrast, Sparsetral introduces a different tensor structure:

```py
model.layers.{bid}.mlp.moe_adapter.router.weight
model.layers.{bid}.mlp.moe_adapter.experts.{xid}.adapter_down.weight
model.layers.{bid}.mlp.moe_adapter.experts.{xid}.adapter_up.weight
```
The adapter experts are added on top of the dense Mistral FFN rather than replacing it with full expert FFNs. 

**Findings**: The current GGUF schema and converter logic already support:

- Dense Mistral/LLaMA FFN tensors
- Full MoE expert tensors (`w1`, `w2`, `w3`)
- Several router implementations used by existing MoE models

However, Sparsetral introduces `moe_adapter.router`, `adapter_down`, and `adapter_up`. These tensor names and their semantics are not currently represented by existing tensor mappings. As a result, simply routing `modeling_sparsetral.MistralForCausalLM` through the existing Mixtral conversion path would likely be incomplete or incorrect.

Therefore, my current implementation decision is to avoid treating Sparsetral as plain Mixtral. The safer direction is to create a Sparsetral-specific conversion path that reuses the existing LLaMA/Mistral converter for dense backbone tensors, then adds separate handling for Sparsetral’s routed adapter tensors.

#### Next Steps

1. Registering `modeling_sparsetral.MistralForCausalLM` with an appropriate converter path.
2. Creating or extending a converter class for Sparsetral.
3. Reusing existing LLaMA/Mistral conversion logic for the dense backbone.
4. Adding Sparsetral-specific handling for `moe_adapter.router`, `adapter_down`, and `adapter_up` tensors.
5. Checking whether new GGUF tensor names or metadata are needed for routed adapter experts.

### Week 4 Progress

#### Registering `modeling_sparsetral.MistralForCausalLM` 
- **Commit showing progress:** [Link](https://github.com/ggml-org/llama.cpp/commit/244ef9be1f2c23aa0cfbec563e5abd9657482296) 
- **Screenshots/logs:**
<img width="814" height="859" alt="image" src="https://github.com/user-attachments/assets/31087838-658d-41c3-8501-c1aae082d122" />

- **Descriptions:**
The first implementation milestone was enabling the Hugging Face to GGUF converter to recognize Sparsetral. I added a dedicated Sparsetral conversion path. The converter now recognizes the exact Hugging Face architecture string: `modeling_sparsetral.MistralForCausalLM`. I also normalized Sparsetral's router metadata:
      - `num_experts` is read as the total expert count
      - `topk` is mapped to the existing `experts_used_per_token` metadata
      - the router scoring function is recorded as softmax, matching the HF forward implementation

- **Validations:**
I reran conversion on the minimal Sparsetral directory. Before this change, conversion stopped at:
```PowerShell
  Model modeling_sparsetral.MistralForCausalLM is not supported
```
After this change, the converter progressed into GGUF metadata export and logged:

```
  expert count = 16
  experts used count = 4
  expert score gating function = softmax
```
This confirms that architecture registration and metadata normalization are working.

- **Current limitation:**
The minimal test directory does not contain actual safetensors weight shards, so the converter produced a metadata-only GGUF:

  `n_tensors` = 0

This means the current validation proves architecture registration and metadata handling, but it does not yet prove tensor conversion.

### Week 5 Progress

#### Implement Sparsetral Tensor Conversion

- **Description:**

After enabling architecture registration, I designed the GGUF tensor mapping for Sparsetral's Sparse MoE adapter tensors. Sparsetral keeps the standard dense Mistral FFN and adds routed adapter tensors. These tensors cannot be represented accurately using the existing Mixtral full-expert mappings, so I introduced dedicated GGUF tensor identifiers and conversion logic.

The existing GGUF schema already supports:

- Dense FFN tensors (`FFN_GATE`, `FFN_DOWN`, `FFN_UP`)
- Full MoE expert tensors (`FFN_GATE_EXP`, `FFN_DOWN_EXP`, `FFN_UP_EXP`)

However, Sparsetral uses a different architecture. Instead of replacing the dense FFN with full experts as in Mixtral, it keeps the dense Mistral FFN and adds routed adapter experts. Because these tensors have different semantics from Mixtral experts, I decided not to reuse the existing `FFN_*_EXP` tensor names. This preserves the distinction between full FFN experts and routed adapter experts while remaining consistent with the existing GGUF naming conventions.

- **Implementation:**

**`conversion/sparsetral.py`**

     - Added buffering and stacking logic for Sparsetral adapter expert tensors.
     - Matches Hugging Face tensors of the form:
    ```text
    model.layers.{bid}.mlp.moe_adapter.experts.{xid}.{adapter_down|adapter_up}.weight
    ```
     - Buffers per-layer expert tensors until all `num_experts` are collected.
     - Merges experts into a single 3D tensor using `torch.stack(..., dim=0)`.
     - Emits merged tensor names:
    ```text
    model.layers.{bid}.mlp.moe_adapter.experts.adapter_down.weight
    model.layers.{bid}.mlp.moe_adapter.experts.adapter_up.weight
    ```
     - Added validation to ensure no incomplete expert buffers remain after conversion.

**`gguf-py/gguf/constants.py`**

  - Added new GGUF tensor identifiers: `FFN_MOE_ADAPTER_GATE`, `FFN_MOE_ADAPTER_DOWN`, `FFN_MOE_ADAPTER_UP`
  - Registered corresponding GGUF tensor names:
    ```text
    blk.{bid}.ffn_moe_adapter_gate.weight
    blk.{bid}.ffn_moe_adapter_down_exps.weight
    blk.{bid}.ffn_moe_adapter_up_exps.weight
    ```

**`gguf-py/gguf/tensor_mapping.py`**

  - Added Hugging Face → GGUF mappings:
    `moe_adapter.router` → `FFN_MOE_ADAPTER_GATE`
    merged `adapter_down` → `FFN_MOE_ADAPTER_DOWN`
    merged `adapter_up` → `FFN_MOE_ADAPTER_UP`

**`src/llama-arch.h`**
  - Added corresponding C++ tensor enums: `LLM_TENSOR_FFN_MOE_ADAPTER_GATE`, `LLM_TENSOR_FFN_MOE_ADAPTER_DOWN`, `LLM_TENSOR_FFN_MOE_ADAPTER_UP`

**`src/llama-arch.cpp`**
  - Registered the corresponding GGUF tensor names for the new adapter tensors.

- **Naming Decision**

The new tensor names follow existing llama.cpp naming conventions:

  - `blk.{bid}` for per-layer tensors.
  - `ffn_` because the tensors belong to the feed-forward (MLP) module.
  - `moe_adapter` preserves the Hugging Face module name.
  - `gate`, `down`, and `up` follow existing projection terminology.
  - `_exps` is consistent with existing expert-stacked tensor naming.

Dedicated names are used instead of the existing FFN_*_EXP tensors because Sparsetral adapters are added to the dense FFN output; they are not replacement-style full FFN experts like Mixtral.

#### Add Adapter-Dimension Metadata

- **Description:** I added GGUF metadata support for Sparsetral's adapter bottleneck size. Sparsetral adapter experts are not full FFN experts. Each adapter uses a low-rank bottleneck:

```text
    hidden_size -> adapter_dim -> hidden_size
    4096        -> 512         -> 4096
```

The loader needs `adapter_dim` to allocate the expert-stacked adapter tensors with the correct shapes:

```PowerShell
blk.{bid}.ffn_moe_adapter_down_exps.weight  { n_embd, adapter_dim, n_expert }
blk.{bid}.ffn_moe_adapter_up_exps.weight    { adapter_dim, n_embd, n_expert }
```

- **Validation**:

I converted the full local Sparsetral model directory:

`python convert_hf_to_gguf.py .\models\sparsetral-minimal --outfile .\sparsetral.gguf`

Conversion completed successfully:

```PowerShell
    expert count = 16
    experts used count = 4
    expert score gating function = softmax
    n_tensors = 387, total_size = 18.8G
    Model successfully exported to sparsetral.gguf
```

This confirms the full model shards are available and the converter can export a real GGUF, not only metadata.

- **Current limitations**
Although conversion succeeded, llama.cpp could not yet load the model correctly. The runtime treated every model with `expert_count > 0` as a full Mixtral-style MoE model.

The next milestone was to implement a dedicated sparse-adapter loading path.

### Week 6 Progress

#### Implement the Sparsetral sparse-adapter loading path

After completing GGUF conversion, I implemented C++ runtime support for loading Sparsetral's dense FFN and routed adapter tensors. This is the checkpoint between GGUF conversion support and full inference support: the model now converts to GGUF with adapter metadata and loads successfully instead of being mistaken for a full Mixtral-style MoE model.

**Observing Errors:**

The original LLaMA loader used two paths:

```
if n_expert == 0:
    load dense FFN
else:
    load full MoE FFN
```

Because Sparsetral reports:

```
expert_count = 16
```

the runtime incorrectly selected the full-MoE path and attempted to load:

```
blk.0.ffn_gate_inp.weight
blk.0.ffn_gate_exps.weight
blk.0.ffn_down_exps.weight
blk.0.ffn_up_exps.weight
```

Loading failed with:

```
missing tensor 'blk.0.ffn_gate_inp.weight'
```

Sparsetral instead contains both dense FFN tensors and sparse-adapter tensors:

```
blk.0.ffn_gate.weight
blk.0.ffn_down.weight
blk.0.ffn_up.weight
blk.0.ffn_moe_adapter_gate.weight
blk.0.ffn_moe_adapter_down_exps.weight
blk.0.ffn_moe_adapter_up_exps.weight
```

**Implementation:**

I added a third loading path:

```
if sparse adapter model:
    load dense FFN
    load sparse-adapter router/down/up tensors
else if n_expert == 0:
    load dense FFN
else:
    load full MoE FFN
```

The following files were updated:

- `gguf-py/gguf/constants.py`

Added the adapter_feed_forward_length metadata key and registered the new sparse-adapter tensor names.

- `gguf-py/gguf/gguf_writer.py`

Added a writer method for adapter_feed_forward_length.

- `conversion/sparsetral.py`

Wrote adapter_dim from the Sparsetral config into GGUF metadata. Also kept the adapter expert stacking logic that merges per-expert HF tensors into GGUF expert-stacked tensors.

- `gguf-py/gguf/tensor_mapping.py`

Mapped Sparsetral HF tensor names to the new GGUF tensor names: ffn_moe_adapter_gate, ffn_moe_adapter_down_exps, and ffn_moe_adapter_up_exps.

- `src/llama-arch.h and src/llama-arch.cpp`

Added C++ metadata and tensor enum entries, GGUF name strings, and tensor op metadata for the sparse-adapter tensors:

```
FFN_MOE_ADAPTER_GATE -> GGML_OP_MUL_MAT
FFN_MOE_ADAPTER_DOWN -> GGML_OP_MUL_MAT_ID
FFN_MOE_ADAPTER_UP -> GGML_OP_MUL_MAT_ID
```

- `src/llama-hparams.h`

Added `n_ff_adapter` to store the adapter bottleneck dimension from GGUF metadata.

- `src/llama-model.h`

Added three tensor fields to `llama_layer` for the adapter router, down projection, and up projection.

- `src/models/llama.cpp`

Added a third loading path for sparse adapter MoE models. Sparsetral has `expert_count > 0`, but it is not a full MoE model. The new path loads the dense FFN tensors plus optional sparse-adapter tensors, avoiding the previous failure where `llama.cpp` looked for `blk.0.ffn_gate_inp.weight`.

**Validation**

I used a convert-build-load validation loop:

1. Reconverted the complete Sparsetral checkpoint with the adapter metadata.
2. Built an x64 Release binary:
```
cmake -B build-msvc-x64 -DCMAKE_BUILD_TYPE=Release 
cmake --build build-msvc-x64 --config Release
```
3. Confirmed the converter exported a complete GGUF:
```
n_tensors = 387, total_size = 18.8G
Model successfully exported to sparsetral-new.gguf
```
4. Tested loading with:
```
.\build-msvc-x64\bin\llama-cli.exe -m .\sparsetral-new.gguf -p "Hello" -n 16
```
The previous error `missing tensor 'blk.0.ffn_gate_inp.weight'` is now resolved.

**Current Checkpoint**

The current implementation confirms that:

- the full Sparsetral checkpoint converts to GGUF;
- adapter metadata and tensors are preserved;
- the C++ loader recognizes the sparse-adapter layout;
- dense FFN and adapter tensors are allocated successfully;
- the model no longer enters the incorrect full-MoE loading path.

<img width="1333" height="748" alt="image" src="https://github.com/user-attachments/assets/5b3e6009-a696-40de-9d83-a052e8b56c1e" />

<img width="1333" height="801" alt="image" src="https://github.com/user-attachments/assets/86506fd1-79ef-4dd8-a9aa-984ab595ec1b" />

**Current Limitation**

Runtime loading is now supported, but correct sparse-adapter inference is not yet implemented.

The adapter tensors are loaded into memory, but the FFN computation graph still needs to:

- compute the normal dense FFN output;
- calculate sparse-adapter router logits;
- select the top-k adapter experts;
- apply the selected adapter_down and adapter_up projections;
- weight and sum the adapter outputs;
- add the routed adapter contribution to the dense FFN output.

The next milestone is therefore sparse-adapter graph construction and numerical validation against the Hugging Face implementation.

### Week 7 Progress

#### Implement routed sparse-adapter inference graph

**Observing Errors:**

The Sparsetral conversion and tensor-loading paths were implemented successfully. The converted GGUF could be loaded, and inference could run without reporting missing tensors. However, the sparse-adapter tensors were only loaded into memory. They were not connected to the computation graph. Consequently, inference executed the ordinary dense Mistral FFN and ignored:

```
  blk.*.ffn_moe_adapter_gate.weight
  blk.*.ffn_moe_adapter_down_exps.weight
  blk.*.ffn_moe_adapter_up_exps.weight
```

The model therefore behaved like its dense base model rather than Sparsetral. Successful generation at this stage only validated conversion and loading, not inference correctness.

**Implementation**

In `src\models\llama.cpp`, I did not modify any graph helper or shared ggml operator. I only reused the existing `build_ffn()`, `build_lora_mm()`, and `build_moe_ffn()` infrastructure.

1. Preserve the Router Input

The normalized FFN input is stored separately:

```cpp
  ggml_tensor * ffn_norm = cur;
```

This is necessary because Sparsetral routes tokens using the input to the dense MLP, not the dense MLP output. Equivalent reference operation:

```cpp
  router_logits = router(ffn_norm)
```

2. Compute the Dense FFN Output

The normal LLaMA/Mistral FFN is evaluated first:

```cpp
  ggml_tensor * dense_out = build_ffn(ffn_norm,
          model.layers[il].ffn_up,   model.layers[il].ffn_up_b,
          model.layers[il].ffn_up_s,
          model.layers[il].ffn_gate, model.layers[il].ffn_gate_b,
          model.layers[il].ffn_gate_s,
          model.layers[il].ffn_down, model.layers[il].ffn_down_b,
          model.layers[il].ffn_down_s,
          NULL,
          LLM_FFN_SILU, LLM_FFN_PAR, il);

  cb(dense_out, "ffn_out", il);

  cur = dense_out;
```

The dense FFN remains active because Sparsetral adapters augment it instead of replacing it.

3. Compute Router Logits

The adapter path is activated only when the layer contains an adapter router:

```cpp
  if (model.layers[il].ffn_moe_adapter_gate != nullptr) {
      ggml_tensor * router_logits = build_lora_mm(
              model.layers[il].ffn_moe_adapter_gate, ffn_norm);

      cb(router_logits, "ffn_moe_adapter_logits", il);
```

`build_lora_mm()` performs the router matrix multiplication. The input is ffn_norm, matching router_hidden_states = x in the Sparsetral reference implementation.

4. Evaluate the Selected Adapter Experts

The existing MoE helper is reused to route and evaluate the adapter experts:

```cpp
  ggml_tensor * adapter_out = build_moe_ffn(
          dense_out,
          nullptr,
          model.layers[il].ffn_moe_adapter_down,
          nullptr,
          model.layers[il].ffn_moe_adapter_up,
          nullptr,
          n_expert,
          n_expert_used,
          LLM_FFN_GELU,
          true,
          hparams.expert_weights_scale,
          LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
          il,
          router_logits);

  cb(adapter_out, "ffn_moe_adapter_out", il);
```

The arguments represent:

```tex
  Expert input:       dense_out
  First projection:   adapter_down
  Activation:         GELU
  Second projection:  adapter_up
  Routing:            softmax followed by top-k
  Weight handling:    normalize selected expert weights
```

This produces:

```cpp
  adapter_out = sum(weight_i * adapter_up_i(GELU(adapter_down_i(dense_out))))
```

5. Add the Adapter Contribution

The routed adapter output is added to the dense FFN output:

```cpp
  cur = ggml_add(ctx0, dense_out, adapter_out);
  cb(cur, "ffn_out", il);
```

The complete computation is therefore:

```cpp
  router_logits = router(ffn_norm)
  dense_out     = dense_ffn(ffn_norm)
  adapter_out   = routed_adapter_moe(dense_out, router_logits)
  ffn_out       = dense_out + adapter_out
```
This is equivalent to the reference implementation because the selected routing weights are normalized to sum to one.

### Code Changes

- **Files modified:** [List]
- **Key commits:** [Links to important commits]
- **Approach decisions:** [Why you chose certain approaches]

---

## Pull Request

**PR Link:** [GitHub PR URL when submitted]

**PR Description:** [Draft or final PR description - much of the content above can be adapted]

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

[What was hard and how you solved it]

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- [Link to helpful documentation]
- [Tutorial or Stack Overflow post that helped]
- [GitHub issues or discussions that helped]
