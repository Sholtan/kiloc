# PROJECT_CONTEXT.md

## Project identity

### Project name
[TO FILL]

### One-sentence description
[TO FILL]

### Primary goal
Describe the main technical goal of the project in one precise paragraph.

[TO FILL]

### Non-goals
List things this project is explicitly not trying to solve yet.

- [TO FILL]
- [TO FILL]

---

## Current status

### Current stage
baseline implementation

### Current active task
Loss function — `src/kiloc/losses/`

### Current blockers
- Must decide MSE vs focal loss before implementing
- `HeatmapHead` outputs raw logits (no sigmoid) — loss must use `BCEWithLogitsLoss` or apply sigmoid explicitly

### Next milestone
Loss function implemented; single training step runs without error and produces a finite loss value.

---

## Problem definition

### Input
What data enters the system?

[TO FILL]

### Output
What should the system predict / generate / return?

[TO FILL]

### Task type
Examples:
- classification
- detection
- segmentation
- regression
- retrieval
- generation
- multimodal ranking

[TO FILL]

### Success condition
What would count as a successful result at the current stage?

[TO FILL]

---

## Datasets

### Main dataset(s)
For each dataset, specify:
- name
- source
- modality
- split structure
- target format
- known issues

#### Dataset 1
- Name: [TO FILL]
- Source: [TO FILL]
- Modality: [TO FILL]
- Splits: [TO FILL]
- Targets: [TO FILL]
- Known issues: [TO FILL]

#### Dataset 2
- Name: [TO FILL]
- Source: [TO FILL]
- Modality: [TO FILL]
- Splits: [TO FILL]
- Targets: [TO FILL]
- Known issues: [TO FILL]

### Data conventions
Document conventions that must stay consistent across the project.

Examples:
- image shape ordering
- coordinate convention
- normalization
- label indexing
- resize rules
- unit definitions

- Image/tensor shape convention: `(C, H, W)` — channels first
- Coordinate convention: `(x, y)` in annotation files; heatmap indexed as `heatmap[y, x]`
- Label convention: channel 0 = positive cells, channel 1 = negative cells
- Preprocessing convention: BGR→RGB, divide by 255, `torch.float32`
- Augmentation convention: [TO FILL]

---

## Metrics and evaluation

### Primary metric(s)
List the metrics that determine success.

- [TO FILL]
- [TO FILL]

### Secondary metric(s)
Useful diagnostics but not the main optimization target.

- [TO FILL]
- [TO FILL]

### Metric definitions
Briefly define each important metric in project-specific terms.

#### Metric 1
- Name: [TO FILL]
- Definition: [TO FILL]
- Why it matters: [TO FILL]

#### Metric 2
- Name: [TO FILL]
- Definition: [TO FILL]
- Why it matters: [TO FILL]

### Evaluation protocol
Specify:
- train/val/test policy
- thresholding/decoding rules
- matching rules if detection task
- aggregation level
- seed policy

[TO FILL]

---

## Current architecture

### High-level pipeline
Describe the end-to-end pipeline from raw input to final output.

[TO FILL]

### Main modules
List the main code modules and their responsibilities.

#### Module: `src/...`
- Responsibility: [TO FILL]
- Inputs: [TO FILL]
- Outputs: [TO FILL]
- Notes: [TO FILL]

#### Module: `src/...`
- Responsibility: [TO FILL]
- Inputs: [TO FILL]
- Outputs: [TO FILL]
- Notes: [TO FILL]

### Entry points
List the main scripts/commands used in the project.

- `python scripts/train.py ...` — [TO FILL]
- `python scripts/evaluate.py ...` — [TO FILL]
- `python scripts/debug_sample.py ...` — [TO FILL]

---

## Constraints

### Technical constraints
Examples:
- limited GPU memory
- limited annotation quality
- only public datasets
- must run on Linux
- must support mixed precision

- [TO FILL]
- [TO FILL]

### Research constraints
Examples:
- need publishable novelty
- need reproducibility
- need fair baseline comparison
- no private clinical data

- [TO FILL]
- [TO FILL]

### Codebase constraints
Examples:
- preserve backward compatibility
- avoid new heavy dependencies
- keep module boundaries stable

- [TO FILL]
- [TO FILL]

---

## Known issues and technical debt

### Known bugs
- [TO FILL]
- [TO FILL]

### Suspected weak points
- [TO FILL]
- [TO FILL]

### Technical debt
- [TO FILL]
- [TO FILL]

### Areas needing refactor
- [TO FILL]
- [TO FILL]

---

## Design decisions currently in force

Use this section for active decisions that the AI should respect unless explicitly reconsidered.

### Decision 1
- Decision: [TO FILL]
- Reason: [TO FILL]
- Consequence: [TO FILL]

### Decision 2
- Decision: [TO FILL]
- Reason: [TO FILL]
- Consequence: [TO FILL]

---

## Preferred workflow for AI collaboration

When helping on this project, the AI should:

1. respect the current project architecture unless a redesign is explicitly justified
2. explain where code belongs before suggesting code
3. state assumptions explicitly
4. separate facts / inferences / guesses when diagnosing issues
5. propose validation steps for any implementation change
6. prefer small, reviewable changes over broad rewrites
7. optimize for learning, correctness, and maintainability

### Preferred response mode
Choose one as default and update as needed.

- [ ] mostly guidance, minimal code
- [ ] mixed guidance and code
- [ ] full code when requested
- [ ] review/debug-first
- [ ] architecture-first

Current preference: [TO FILL]

---

## Active questions

List the most important unresolved questions.

- [TO FILL]
- [TO FILL]
- [TO FILL]

---

## Quick commands

Keep frequently used commands here.

```bash
# environment
[TO FILL]

# train
[TO FILL]

# evaluate
[TO FILL]

# tests
[TO FILL]

# lint
[TO FILL]