# TASK_TEMPLATE.md

## Task title
[TO FILL]

## Date opened
[YYYY-MM-DD]

## Status
Choose one:
- planned
- active
- blocked
- in review
- paused
- done
- dropped

Current status: [TO FILL]

---

## Why this task exists

What larger project need does this task serve?

[TO FILL]

---

## Task definition

Describe the task in precise technical terms.

[TO FILL]

---

## Desired end state

What should be true when this task is complete?

[TO FILL]

---

## Out of scope

What should not be changed in this task?

- [TO FILL]
- [TO FILL]

---

## Assumptions

State working assumptions explicitly.

- [TO FILL]
- [TO FILL]

---

## Relevant files/modules

List the files likely involved.

- `src/...` — [TO FILL]
- `scripts/...` — [TO FILL]
- `tests/...` — [TO FILL]
- `configs/...` — [TO FILL]

---

## Interfaces/contracts affected

Describe the interfaces that matter.

### Interface 1
- Owner: [TO FILL]
- Input contract: [TO FILL]
- Output contract: [TO FILL]
- Invariants: [TO FILL]

### Interface 2
- Owner: [TO FILL]
- Input contract: [TO FILL]
- Output contract: [TO FILL]
- Invariants: [TO FILL]

---

## Implementation plan

Break the task into concrete steps.

1. [TO FILL]
2. [TO FILL]
3. [TO FILL]
4. [TO FILL]

---

## Validation plan

How will correctness be checked?

### Required checks
- [ ] code runs without crashing
- [ ] lint/format passes
- [ ] tests pass
- [ ] shape/type/device checks pass
- [ ] toy example behaves correctly
- [ ] one-batch debug run passes
- [ ] visual sanity check passes
- [ ] metric sanity check passes

### Task-specific checks
- [TO FILL]
- [TO FILL]

---

## Definition of done

This task is done when all of the following are true:

- [ ] implementation matches desired end state
- [ ] required validation checks pass
- [ ] no known critical bug remains
- [ ] affected interfaces are still coherent
- [ ] task notes are updated
- [ ] follow-up work is identified if needed

Additional done criteria:
- [TO FILL]
- [TO FILL]

---

## Risks / failure modes

What is most likely to go wrong?

- [TO FILL]
- [TO FILL]
- [TO FILL]

---

## Debug notes

Use this section during implementation.

### Observed issue 1
- Observation: [TO FILL]
- Suspected cause: [TO FILL]
- Evidence: [TO FILL]
- Next check: [TO FILL]

### Observed issue 2
- Observation: [TO FILL]
- Suspected cause: [TO FILL]
- Evidence: [TO FILL]
- Next check: [TO FILL]

---

## Review request format for AI

When asking AI to review this task, use this structure:

### Context
[Brief project/task context]

### What I changed
[TO FILL]

### Files changed
- [TO FILL]
- [TO FILL]

### Expected behavior
[TO FILL]

### Actual behavior
[TO FILL]

### What I already tested
- [TO FILL]
- [TO FILL]

### What I suspect
[TO FILL]

### Review request
Choose one or more:
- inspect for correctness bugs
- inspect interface design
- inspect for hidden edge cases
- suggest minimal fixes
- suggest missing tests
- critique implementation plan
- check whether task should be split

---

## Outcome

Fill this when the task is closed.

### Final result
[TO FILL]

### What changed
[TO FILL]

### What was learned
[TO FILL]

### Follow-up tasks
- [TO FILL]
- [TO FILL]

### Date closed
[YYYY-MM-DD]