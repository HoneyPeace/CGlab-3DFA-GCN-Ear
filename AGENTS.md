# AGENTS.md

## Purpose

This repository may be modified with the help of AI coding agents such as OpenAI Codex.

The agent must prioritize:

1. Correctness
2. Reproducibility
3. Minimal code changes
4. Preservation of existing behavior
5. Clear validation
6. Safe Git/GitHub workflow
7. Readable code that follows the existing project style

This project may include research code, model training code, model evaluation code, data preprocessing code, visualization code, and report-generation code.

Because small code changes can affect experiment results, the agent must be conservative.

---

## Core Rule

Before making any code change, the agent must first understand the current code.

The agent must not modify code based only on assumptions.

For every task, follow this order:

1. Understand the user's request.
2. Identify the relevant files and functions.
3. Inspect the current implementation before editing.
4. Explain the current behavior briefly.
5. Identify what must be preserved.
6. Make the smallest possible change.
7. Validate the change.
8. Report what changed, what was preserved, and how to test it.

---

## Non-Negotiable Rules

The agent must not:

- Rewrite large parts of the codebase unless explicitly requested.
- Change existing file loading logic unless explicitly requested.
- Change existing folder paths unless explicitly requested.
- Change existing command-line arguments unless explicitly requested.
- Change existing print messages or log messages unless explicitly requested.
- Change existing output file names or result formats unless explicitly requested.
- Change model architecture unless explicitly requested.
- Change metric calculation unless explicitly requested.
- Change dataset split logic unless explicitly requested.
- Change normalization, sampling, heatmap, or regression logic unless explicitly requested.
- Remove existing functionality while adding or fixing another feature.
- Install new dependencies unless explicitly approved.
- Delete, overwrite, or rename existing datasets, checkpoints, logs, or result files.
- Use destructive commands such as `rm -rf`, `git reset --hard`, `git clean -fd`, or database deletion commands.
- Commit or push to GitHub unless explicitly requested by the user.

---

## Minimal Code Change Rule

The agent must make the smallest change that satisfies the user's request.

Preferred examples:

- Add one helper function.
- Add one optional argument.
- Add one validation script.
- Modify one conditional branch.
- Add one output-saving block without changing metric logic.
- Add a small check without changing the original data flow.

Avoid:

- Rewriting the whole file.
- Replacing the data loader.
- Renaming many variables.
- Reorganizing the project structure.
- Modifying unrelated files.
- Combining bug fix, refactor, and feature addition in one patch.
- Reformatting the entire file.
- Changing style-only details unrelated to the task.

---

## Preserve Existing Behavior

Unless explicitly requested, preserve:

- Existing file paths
- Existing relative path usage
- Existing command-line arguments
- Existing output messages
- Existing result table formats
- Existing model loading behavior
- Existing data loading behavior
- Existing training loop behavior
- Existing evaluation loop behavior
- Existing visualization behavior
- Existing saved file naming conventions
- Existing function names
- Existing class names
- Existing return values
- Existing import paths
- Existing folder structure
- Existing experiment flow

---

## Research Code Protection Rules

This repository may include machine learning, deep learning, 3D point cloud, landmark detection, heatmap regression, and thesis-related experiment code.

Do not change the following unless explicitly requested:

- Dataset split logic
- Data augmentation logic
- FPS sampling logic
- Number of input points
- Number of input channels
- Landmark count
- Landmark index order
- Coordinate system
- Coordinate normalization logic
- Coordinate denormalization logic
- Heatmap generation formula
- Sigma handling
- Top-k selection logic
- Landmark regression logic
- Weighted average logic
- NaN handling
- Mean Error calculation
- Standard Deviation calculation
- Success Rate calculation
- mIoU calculation
- Cosine similarity calculation
- Evaluation aggregation logic
- Model checkpoint loading logic
- Model architecture
- Loss function
- Batch size behavior
- Training epoch behavior
- Random seed behavior

---

## Metric Integrity

Metrics must not be changed unless the user explicitly requests it.

Protected metrics include:

- ME
- STD
- Per-landmark ME
- Per-landmark STD
- Average ME
- Average STD
- SR@5mm
- SR@10mm
- mIoU
- Cosine similarity
- Inference time

If the user asks to add a new metric, add it separately without changing existing metric calculations.

If metric logic is changed, the agent must clearly state that previous and new results may not be directly comparable.

---

## Landmark and Heatmap Rules

For landmark-related code, preserve:

- Landmark index order
- Landmark count
- Landmark naming convention
- GT landmark coordinate format
- Predicted landmark coordinate format
- NaN landmark exclusion logic
- Per-landmark reporting format
- Overall average reporting format

Do not reorder landmarks for readability.

Do not rename landmark IDs unless explicitly requested.

For heatmap-related code, preserve:

- Heatmap shape
- Heatmap generation formula
- Sigma value interpretation
- Distance calculation method
- Normalization method
- Whether heatmap values are clipped
- Whether heatmap values are normalized
- Point-to-landmark relationship

If changing heatmap logic is requested, clearly state that results may no longer be directly comparable to previous experiments.

---

## Point Cloud Rules

For point cloud code, preserve:

- Point order unless explicitly requested
- Point count
- Coordinate scale
- Coordinate unit
- Channel order
- Normal vector handling
- Curvature or additional feature channels
- FPS or sampling behavior
- Original coordinate space for metric calculation

If normalization is applied, clearly distinguish between:

- Normalized coordinates used for model input
- Original coordinates used for metric calculation

---

## Model Evaluation Rules

Evaluation code must be treated as highly sensitive.

Do not change the following unless explicitly requested:

- Model loading path
- Checkpoint selection logic
- Evaluation dataset selection
- Batch iteration behavior
- Metric aggregation
- NaN exclusion logic
- Printed summary format
- Saved result format

When adding output-saving functionality, add it after existing metric calculation unless there is a clear reason not to.

---

## Training Code Rules

Training code must preserve:

- Dataset loading
- Batch size behavior
- Shuffle behavior
- Loss calculation
- Optimizer behavior
- Learning rate schedule
- Epoch count
- Checkpoint saving
- Logging format

Do not change training hyperparameters unless explicitly requested.

---

## Logging and Print Message Rules

Existing print messages and logs must be preserved.

Do not rename, remove, or reformat existing messages unless explicitly requested.

If new messages are needed, add them separately using a clear prefix such as:

```text
[INFO]
[CHECK]
[WARNING]
[ERROR]
```

Do not remove existing Korean or English messages unless explicitly requested.

---

## Error Handling Rules

Do not hide errors with broad exception handling.

Avoid:

```python
try:
    ...
except:
    pass
```

Prefer:

```python
try:
    ...
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    raise
```

Do not suppress warnings unless the user explicitly asks.

If an error is fixed, explain:

- What caused the error
- Where it occurred
- What was changed
- How to confirm that it is fixed

---

## File Writing Rules

When generating new files, do not overwrite existing important files.

Use safe output folders such as:

```text
outputs/
results/
debug_outputs/
validation_outputs/
```

If a file may already exist, either:

- Create a timestamped file name.
- Ask before overwriting.
- Save with a new suffix such as `_new`, `_debug`, or `_validated`.

Do not overwrite the following unless explicitly requested:

- Raw datasets
- Processed datasets
- Model checkpoints
- Existing experiment logs
- Existing evaluation results
- Thesis/report result tables

---

## Dependency Rules

Do not add a new package unless:

- The task cannot reasonably be done with existing packages.
- The user explicitly approves the dependency.
- The installation command is documented.
- The impact on reproducibility is explained.

Prefer standard library modules when possible.

Examples:

```python
import os
from pathlib import Path
import json
import csv
```

Do not introduce a new dependency just for simple file saving, path handling, formatting, or logging.

---

## Destructive Operation Rules

Do not run or suggest destructive commands unless the user explicitly requests them.

Forbidden by default:

```bash
rm -rf
git reset --hard
git clean -fd
del /s
rmdir /s
drop database
truncate table
```

Before suggesting Git operations, prefer safe commands:

```bash
git status
git diff
git branch
git log --oneline -5
```

---

## GitHub Commit and Push Safety Rules

The agent must handle Git, GitHub commits, and pushes conservatively.

Git operations can affect the entire project history and collaboration flow, so the agent must prioritize safety, reviewability, and reversibility.

### Core Git Rule

The agent must not commit or push changes unless the user explicitly requests it.

Allowed only with explicit user request:

- `git commit`
- `git push`
- `git pull`
- `git merge`
- `git rebase`
- `git reset`
- `git clean`
- `git checkout` that overwrites local changes
- Branch deletion
- Force push

The agent may suggest Git commands, but must not execute risky Git commands without explicit approval.

### Before Any Commit

Before creating a commit, the agent must check the current repository state.

Required commands:

```bash
git status
git diff
git diff --staged
```

The agent must review:

- Which files were modified
- Which files were added
- Which files were deleted
- Whether unrelated files are included
- Whether generated files are accidentally included
- Whether large files, datasets, checkpoints, or logs are included
- Whether secrets or credentials are included

The agent must not commit unrelated changes.

### Files That Should Not Be Committed Without Explicit Approval

Do not commit the following unless the user explicitly requests it:

- Raw datasets
- Processed datasets
- Model checkpoints
- Large binary files
- Experiment logs
- Temporary debug outputs
- Cache folders
- Environment folders
- `.env` files
- API keys or credentials
- Personal files
- IDE-specific local settings
- Generated result files unless they are part of the requested output

Examples:

```text
*.pth
*.pt
*.t7
*.ckpt
*.pkl
*.npy
*.npz
*.mat
*.log
.env
__pycache__/
.venv/
env/
outputs/
debug_outputs/
validation_outputs/
```

If these files appear in `git status`, the agent must warn the user before committing.

### Commit Scope Rule

Each commit should contain one logical change.

Good commit scopes:

- Fix model checkpoint path handling
- Add prediction result export
- Add landmark validation script
- Fix heatmap shape mismatch
- Update README command example
- Add Codex agent rules

Bad commit scopes:

- Fix bug, refactor model, change metrics, and update slides
- Modify training code and evaluation code without clear reason
- Include unrelated formatting changes
- Include local experiment output files

### Commit Message Rule

Commit messages must be clear and specific.

Preferred format:

```text
<type>: <short summary>
```

Allowed types:

- `fix`
- `feat`
- `refactor`
- `docs`
- `test`
- `chore`
- `experiment`

Examples:

```text
fix: preserve eval metric calculation when saving predictions
feat: add landmark prediction export option
test: add heatmap shape validation script
docs: add Codex agent rules
refactor: simplify result table formatting without behavior change
```

Avoid vague commit messages:

- update
- fix
- changes
- final
- new code

### Before Push

Before pushing, the agent must confirm:

- `git status` is clean or only expected files are staged.
- The commit contains only intended changes.
- No secrets, datasets, checkpoints, logs, or temporary files are included.
- The target branch is correct.
- The remote repository is correct.
- The user explicitly requested push.

Required commands before push:

```bash
git status
git branch
git remote -v
git log --oneline -5
```

The agent must not push to `main` or `master` unless the user explicitly says to push to that branch.

### Branch Safety Rule

Prefer working on a feature branch instead of directly modifying `main` or `master`.

Recommended branch naming:

```text
fix/<short-description>
feat/<short-description>
docs/<short-description>
test/<short-description>
experiment/<short-description>
```

Examples:

```text
fix/eval-output-save
feat/pred-landmark-export
docs/agent-rules
test/heatmap-validation
```

Before creating or switching branches, check:

```bash
git status
git branch
```

Do not switch branches if there are uncommitted changes that may be overwritten.

### Pull Safety Rule

Do not run `git pull` blindly.

Before pulling:

```bash
git status
git branch
git remote -v
```

If there are local changes, the agent must not pull unless the user confirms how to handle them.

The agent must explain whether the pull may cause:

- Merge conflicts
- Overwritten local changes
- Diverged branch history
- Dependency changes
- Result changes

### Conflict Handling Rule

If a merge conflict occurs, the agent must not guess blindly.

The agent must:

- Identify conflicted files.
- Explain the conflict.
- Preserve the user's local changes unless told otherwise.
- Resolve only the relevant conflict.
- Re-run validation if code was affected.
- Show the resolved difference before commit.

### Force Push Rule

Force push is forbidden by default.

Do not run:

```bash
git push --force
git push -f
git push --force-with-lease
```

unless the user explicitly requests it and understands the risk.

If force push is requested, the agent must first explain:

- Which branch will be overwritten
- Whether other people's commits may be lost
- Why force push is needed
- Safer alternatives

### Recommended Git Workflow

For normal code changes, use this workflow:

```bash
git status
git checkout -b fix/<short-description>
# modify code
git diff
# run validation
git add <specific files only>
git diff --staged
git commit -m "fix: <clear summary>"
git status
git push origin fix/<short-description>
```

Do not use `git add .` unless the agent has confirmed that every changed file should be committed.

Prefer:

```bash
git add path/to/specific_file.py
git add AGENTS.md
```

Avoid:

```bash
git add .
git add -A
```

### Final Git Report

After suggesting or performing Git actions, the agent must report:

```text
Git status
- Current branch:
- Changed files:
- Commit performed:
- Push performed:
- Remote repository:
- Notes:
```

---

## Code Convention and Readability Rules

The agent must improve readability while preserving the existing code style and behavior.

The goal is not to make the code look like a completely new project.

The goal is to make the current code easier to understand without breaking the existing workflow.

### Existing Style First Rule

The agent must follow the style already used in the repository.

Before editing, inspect nearby code and match:

- Naming style
- Indentation style
- Function structure
- Comment style
- Print/log message style
- File organization
- Import style
- Error handling style
- CLI argument style
- Output formatting style

Do not impose a new style if the existing code uses a different but working style.

### Minimal Convention Change Rule

Do not reformat the entire file.

Do not rename variables, functions, classes, or files only for style reasons.

Do not move functions to different files unless explicitly requested.

Do not change code layout broadly unless the user requests cleanup or refactoring.

Acceptable readability improvements:

- Add a short comment explaining non-obvious logic.
- Extract repeated logic into a small helper function only when necessary.
- Rename a new variable clearly.
- Add whitespace around a newly added block.
- Add validation prints with clear prefixes.
- Split a very long new line into readable lines.
- Add docstring to a new helper function.

Avoid:

- Rewriting old functions just to make them prettier.
- Applying automatic formatting to the whole file.
- Renaming existing variables across many places.
- Changing import order throughout the file.
- Changing all quote styles.
- Changing all comments.
- Converting procedural code into class-based code.
- Converting class-based code into procedural code.

### Readability Without Behavior Change

When improving readability, the agent must preserve behavior exactly.

Do not change:

- Calculation order
- Random seed behavior
- Loop order
- Data order
- Landmark order
- Tensor shape
- Return type
- File output format
- Print output format
- Exception behavior
- Floating-point formula
- Metric aggregation logic

Even small-looking changes can affect experimental results.

### Naming Rules

For new variables, use clear and descriptive names.

Good examples:

```text
pred_landmarks
gt_landmarks
valid_mask
nan_count
num_points
num_landmarks
output_dir
save_path
```

Avoid unclear names for new code:

```text
x1
tmp2
aaa
newdata
result2
final_final
```

However, do not rename existing variables unless necessary.

If existing code uses short names such as `x`, `y`, `B`, `N`, `C`, or `L`, keep them if they are standard tensor dimensions.

Recommended tensor dimension meanings:

- `B` = batch size
- `N` = number of points
- `C` = number of channels
- `L` = number of landmarks
- `K` = top-k count

### Comment Rules

Comments should explain why, not repeat what the code already says.

Good comments:

```python
# Keep coordinates in the original point-cloud space because ME is computed in millimeters.
# Do not change landmark order because evaluation tables depend on fixed landmark IDs.
# NaN landmarks are excluded to match the existing evaluation protocol.
```

Bad comments:

```python
# Add one
x = x + 1

# Loop over list
for item in items:
    ...
```

Do not add too many comments.

Prefer a few useful comments near fragile or important logic.

### Function Rules

New functions should be small and focused.

A new function should usually do one thing.

Good examples:

```python
def save_prediction_results(...):
    ...

def count_nan_landmarks(...):
    ...

def validate_heatmap_shape(...):
    ...
```

Avoid large generic functions such as:

```python
def process_everything(...):
    ...

def do_all_tasks(...):
    ...
```

Do not change existing function signatures unless explicitly requested.

If a new behavior is optional, prefer adding an optional argument with a safe default.

Example:

```python
def evaluate_model(..., save_predictions=False):
    ...
```

The default behavior must preserve the existing behavior.

### Import Rules

Do not add unnecessary imports.

Do not reorganize all imports unless requested.

When adding a new import:

- Check whether the package is already used in the project.
- Prefer standard library modules when possible.
- Avoid new external dependencies.
- Add the import near similar existing imports.

Preferred standard libraries:

```python
import os
from pathlib import Path
import json
import csv
```

Do not introduce a new dependency just for simple file saving or formatting.

### Path Handling Rules

Preserve existing path style.

If the project currently uses relative paths, do not convert everything to absolute paths.

If adding new path handling, prefer safe and readable code:

```python
from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / "predictions.csv"
```

Do not hardcode user-specific paths unless the existing code already requires it or the user explicitly asks.

Avoid adding paths like:

```text
C:\Users\...
/home/user/...
```

unless the user specifically requested it.

### Print and Logging Style

Preserve existing print style.

If adding new print messages, use clear prefixes:

```python
print("[INFO] Saving prediction results...")
print("[CHECK] pred_landmarks shape:", pred_landmarks.shape)
print("[WARNING] Output file already exists:", save_path)
```

Do not remove existing print messages.

Do not change Korean/English message style unless requested.

### Formatting Rules

Readable formatting is allowed only for modified or newly added code.

Acceptable:

```python
error = np.linalg.norm(pred_point - gt_point)
```

Acceptable for long lines:

```python
pred_landmark = landmark_regression(
    points=points,
    heatmap=pred_heatmap,
    topk=regression_point_num,
)
```

Avoid reformatting unrelated old code.

Do not run global formatters such as `black`, `isort`, or `autopep8` unless the user explicitly requests it.

### Code Duplication Rule

Do not aggressively remove duplication unless requested.

Small duplication is acceptable in research code if it makes experiments easier to compare.

Only extract duplicated code when:

- The duplication is directly related to the requested change.
- The extraction does not change behavior.
- The new helper function is simple.
- The user can still understand the flow.

### Backward Compatibility Rule

New changes must be backward-compatible by default.

If adding a CLI argument, provide a default value that preserves current behavior.

Example:

```python
parser.add_argument("--save_predictions", action="store_true")
```

This means existing commands still behave the same unless the new flag is used.

Avoid changing required arguments.

### Beginner-Friendly Code Rule

Because the code may be reviewed by a beginner, prefer straightforward code over clever code.

Avoid:

- Overly compressed one-liners
- Complex nested comprehensions
- Unnecessary decorators
- Advanced metaprogramming
- Hidden side effects
- Ambiguous variable names

Prefer:

- Step-by-step logic
- Clear intermediate variables
- Simple conditionals
- Explicit shape comments for tensor code
- Clear validation prints

### Research Code Readability Rule

For research and ML code, readability should support experiment tracking.

When adding or modifying code, make it clear:

- What data goes in
- What shape it has
- What transformation is applied
- What output is produced
- Whether coordinates are normalized or original
- Whether metrics are affected

Example comment style:

```python
# points: [B, N, C], original point-cloud coordinates
# pred_heatmap: [B, L, N], predicted probability per landmark and point
# pred_landmarks: [B, L, 3], regressed landmark coordinates in original space
```

### Final Convention Report

After making readability or convention-related changes, the agent must report:

```text
Code convention changes
- ...

Existing style preserved
- ...

Readability additions
- ...

Behavior change
- No behavior change / Behavior changed

Notes
- ...
```

---

## Refactoring Rules

Refactoring is allowed only when:

- The user explicitly asks for refactoring.
- The requested change is impossible without a small refactor.

If refactoring is necessary, clearly separate:

- Behavior-preserving refactor
- Functional change

Do not mix large refactoring with feature implementation.

When refactoring is done, the agent must explain:

- Why the refactor was necessary
- What behavior was preserved
- How behavior preservation was validated
- Whether numerical results may change

---

## Validation Rules

Every code change must include a validation plan.

The agent must not say that code works unless it was actually executed and verified.

The agent must distinguish between:

- Code reviewed
- Code modified
- Code executed
- Code validated

If the code was not executed, say clearly:

```text
I reviewed the code statically, but did not execute it.
```

If the code requires the user's local environment, say:

```text
This validation must be run in the user's local dataset, checkpoint, and CUDA environment.
```

### Required Validation Report

After modifying code, provide:

```text
Validation:
1. Command:
2. Expected result:
3. Values to check:
4. Parts that must remain unchanged:
5. Remaining risk:
```

### Shape Validation

For tensor, point cloud, image, or model code, always track shape changes.

Use this format:

```text
Input:
- points: [B, N, C]

Model input:
- points_transposed: [B, C, N]

Model output:
- pred_heatmap: [B, L, N]

Regression output:
- pred_landmarks: [B, L, 3]
```

### Numeric Validation

For metric-related code, validate:

- Number of samples
- Number of landmarks
- Number of valid landmarks
- Number of NaN landmarks
- Per-landmark ME
- Per-landmark STD
- Average ME
- Average STD
- SR@5mm
- SR@10mm

### Before/After Comparison

If the change is supposed to preserve behavior, compare before and after.

Example:

```text
Before:
Average ME = 1.104 mm
Average STD = 0.670 mm

After:
Average ME = 1.104 mm
Average STD = 0.670 mm

Result:
Metric calculation was preserved.
```

### Output File Validation

If the change creates files, verify:

- File path
- File name
- File extension
- Whether existing files are overwritten
- Number of rows
- Number of columns
- Column names
- Sample contents

### Error Validation

If fixing an error, confirm:

- The original error no longer occurs.
- The original intended command works.
- No unrelated behavior changed.
- Any new warning is explained.

### Reproducibility Statement

When modifying research code, the agent must report whether the change can affect reproducibility.

Use one of the following:

```text
This change should not affect numerical results because it only adds output saving.
This change may affect numerical results because it modifies preprocessing.
This change will affect comparability with previous experiments because it changes the metric calculation.
```

---

## Harness Engineering Rules

When the user asks for harness engineering or when a change may silently affect data, model output, coordinate transformation, or metrics, prefer adding or suggesting a separate validation harness.

A harness can be:

- A small test script
- A debug runner
- A minimal reproducible example
- A validation wrapper
- A comparison script
- A dry-run mode
- A logging utility
- A controlled experiment runner

Harness scripts should use names such as:

```text
check_<feature>.py
validate_<feature>.py
debug_<feature>.py
compare_<before>_<after>.py
test_<feature>.py
```

A harness must:

- Not change the original behavior.
- Be easy to run.
- Produce clear output.
- Compare before/after behavior when possible.
- Not overwrite original data, checkpoints, logs, or results.
- Isolate the tested function or behavior.
- Make failures easy to understand.

For data or model pipelines, check:

- Input shape
- Output shape
- Data type
- Device
- Min and max values
- NaN count
- Inf count
- Batch size
- Number of points
- Number of channels
- Coordinate range
- Whether normalization was applied
- Whether original coordinate space is preserved
- Whether output file path is safe

For landmark or heatmap code, also check:

- Number of landmarks
- Heatmap shape
- Sigma value
- Top-k indices
- Top-k point coordinates
- Weighted average result
- GT coordinate
- Predicted coordinate
- Error distance
- Mean Error
- Standard Deviation
- Whether NaN landmarks are excluded or included

Harness output should use clear prefixes:

```text
[CHECK] Input shape:
[CHECK] Output shape:
[CHECK] Min / Max:
[CHECK] NaN count:
[CHECK] Expected:
[CHECK] Actual:
[PASS] ...
[FAIL] ...
```

---

## Final Response Format

After modifying code, respond with:

```text
변경한 내용
- ...

수정한 파일
- ...

수정한 함수/위치
- ...

보존한 부분
- ...

하네스/검증 방법
1. 실행 명령어:
2. 예상 출력:
3. 확인해야 할 값:
4. 기존과 동일해야 하는 부분:

주의할 점
- ...
```

---

## Honesty Requirement

Do not overclaim.

Do not claim that code was executed or tested unless it was actually executed.

Use clear statements:

```text
I reviewed the code statically, but did not execute it.
This part must be confirmed with real local data.
Based on the current information, I cannot determine whether this file is actually called.
```

---

## Final Principle

When uncertain, preserve the existing behavior and ask for clarification only if the task cannot be safely completed.

Small, safe, explainable changes are better than broad automatic rewrites.

The agent should help improve the code, but must not silently change the experiment, metric, model behavior, file structure, or Git history.
