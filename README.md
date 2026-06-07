# Contribution 1: Support for Sparse MoE models like Camelidae and Sparsetral

**Contribution Number:** 1  
**Student:** Ngan Huong Nguyen
**Issue:** [[GitHub issue link]](https://github.com/ggml-org/llama.cpp/issues/5365)  
**Status:** Phase I - In Progress

---

## Why I Chose This Issue

I chose this issue because it is closely related to large language model inference, model architectures, and systems programming, which are areas I want to explore more deeply. I am currently learning NLP, transformers, and modern LLM systems, so contributing to llama.cpp provides an opportunity to understand how production-grade inference engines support different model architectures.

This issue also matches my technical background. I am comfortable with C++ and Python and have experience reading and modifying machine learning codebases. Through this contribution, I hope to learn how model conversion, tensor mapping, and inference support are implemented in real-world LLM infrastructure.

---

## Understanding the Issue

### Problem Description

Currently, the project is supporting lots of standard Mixture-of-Expert (MoE) architectures, but does not provide explicit support for Sparse MoE architectures although sparse MoE architectures require less memory but perform better compared to standard ones.

### Expected Behavior

Users should be able to convert, load, and run Sparse MoE models in llama.cpp in the same way they can run other supported MoE architectures.

### Current Behavior

Support for Sparse MoE is not available, preventing these models from being used through the standard llama.cpp workflow.

### Affected Components

[Which parts of the codebase are involved?]

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
