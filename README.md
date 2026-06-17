# Contribution 1: Support for Sparse MoE models like Camelidae and Sparsetral

**Contribution Number:** 1  

**Student:** Ngan Huong Nguyen

**Issue:** [Support for Sparse MoE models like Camelidae and Sparsetral](https://github.com/ggml-org/llama.cpp/issues/5365)  

**Status:** Phase II - In Progress

---

## Why I Chose This Issue

I chose this issue because it is closely related to areas that I want to explore deeply, including large language model inference, model architectures, and systems programming. I am currently learning NLP, transformers, and modern deep learning systems, so contributing to llama.cpp provides an opportunity to understand how production-grade inference engines support different model architectures.

This issue also matches my technical background. I am comfortable with C++ and Python and have experience reading and modifying machine learning codebases. Through this contribution, I hope to learn how model conversion, tensor mapping, and inference support are implemented in real-world LLM infrastructure.

---

## Understanding the Issue

### Problem Description

Currently, the project is supporting lots of standard Mixture-of-Expert (MoE) architectures, but does not provide explicit support for Sparse MoE architectures. These architectures combine dense base model computation with routed lightweight adapter experts, requiring different model conversion, metadata, and runtime handling compared to traditional MoE implementations.

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

After reproducing the issue, I found that the converter can read the local Sparsetral model configuration, but fails when it tries to match the model architecture to a supported converter implementation. This suggests that the first root cause is not missing model files or tokenizer files, but the absence of a registered conversion path for Sparsetral in the Hugging Face to GGUF conversion pipeline. The converter identifies the architecture as `modeling_sparsetral.MistralForCausalLM`, but does not know which existing converter class or architecture mapping should handle it.

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

### Week [X] Progress

[What you built this week, challenges faced, decisions made]

### Week [Y] Progress

[Continue documenting as you work]

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
