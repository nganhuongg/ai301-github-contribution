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
The goal of this contribution is not to support only a single model, but to investigate support for the Sparse LoRA-MoE architecture family described in the PESC paper. Sparsetral will be used as the initial investigation target because its public artifacts (configuration, custom modeling code, and checkpoints) make the architecture easier to inspect and reproduce. Camelidae will be used later to validate whether the implementation generalizes across models from the same architecture family.

---

## Reproduction Process

### Environment Setup

[Notes on setting up your local development environment - challenges you faced, how you solved them]

### Steps to Reproduce

1. [Step 1]
2. [Step 2]
3. [Observed result]

### Reproduction Evidence

- **Commit showing reproduction:** [Link to commit in your fork]
- **Screenshots/logs:** [If applicable]
- **My findings:** [What you discovered during reproduction]

---

## Solution Approach

### Analysis

[Your analysis of the root cause - what's causing the issue?]

### Proposed Solution

[High-level description of your fix approach]

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** [Restate the problem]

**Match:** [What similar patterns/solutions exist in the codebase?]

**Plan:** [Step-by-step implementation plan]
1. [Modify file X to do Y]
2. [Add function Z]
3. [Update tests]

**Implement:** [Link to your branch/commits as you work]

**Review:** [Self-review checklist - does it follow the project's contribution guidelines?]

**Evaluate:** [How will you verify it works?]

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
