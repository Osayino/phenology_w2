# Course workflow and learning plan

Status (11 November 2026): the active project remains
`C:\Study\2026-27\P2_AI\phenology_w2`. The notebook now implements all remaining
exercises and has completed a fresh ordered run. Full pixel datasets, a shared split,
matched classifier results, figures and verification are saved. The original ROI-level
work is preserved. The reflection needs student review, and full-rubric and submission
checks remain pending. See README.md and results/verification.json for current evidence.
No Git repository or remote publication has been created.

Portability update (13 November 2026): the notebook now finds its project folder
when launched from the root or `casework`, with a `PROJECT_DIR` override for other
launch locations. See START-HERE.txt for extraction, environment recreation and
kernel selection on another computer. A separate optional classifier comparison
has been added. All 35 cells passed a fresh ordered run in a relocated project
with an empty cache. Original K-NN predictions and metrics matched exactly.
The validation-selected random forest achieved 34.12% test accuracy versus
26.54% for raw K-NN; this remains an exploratory result on the fixed ROI split.
A clean environment also passed the four numerical/data checks. The 11 November
status above records the original 32-cell K-NN experiment. The runnable hand-in
archive is `dist/phenology_submission.zip`; its setup instructions are in START-HERE.txt.

Default study root: `C:\Study`.

Default current course folder: `C:\Study\2026-27\P2_AI`.

Default phenology project destination: `C:\Study\2026-27\P2_AI\phenology_w2`.

Continue editing `C:\Study\2026-27\P2_AI\phenology_w2\casework\case_notebook.ipynb`. The earlier copy at `C:\Users\User\Documents\Codex\2026-11-04\t\outputs\phenology-vscode` is retained as a backup; do not alternate between copies.

## Purpose

Build a reusable way to organize coursework, understand code, keep a history of changes, and produce work another person can inspect and reproduce. Learn the tools through real assignments, beginning with phenology.

The student wants these habits to support future computer science study, research collaboration, and building a startup. Those paths need further subject knowledge and experience; this workflow supplies a practical foundation.

## Agreed location and planned contents

Use `C:\Study` as the permanent course home, grouped by academic year and course. This supersedes the earlier proposed `Documents\University` location and `AIN30306\assignments` nesting. The phenology project below now exists with its notebook, dataset, references, README and environment. Folders marked “Add” remain suggestions rather than existing content.

```text
C:/Study/
  2026-27/
    P2_AI/
      course-info/              # Add for syllabus, assessment and submission rules
      notes/                    # Add for personal course notes
      phenology_w2/             # One self-contained project
        README.md               # Question, setup, how to run, current status
        casework/
          case_notebook.ipynb    # Code, explanations, experiments, results
        raw/                    # Supplied images and masks, kept unchanged
        reference/              # Original notebook and assignment documents
        results/                # Add when there are results worth saving
        .vscode/                # Project-specific editor settings
        .venv/                  # Local Python environment
        .gitignore
        requirements.txt
        requirements-lock.txt
```

Use the same pattern for other courses, creating folders only when they have a purpose. A writing assignment may need documents and references rather than a Python environment or Git repository.

For phenology, preserve the existing `casework`, `raw`, and `reference` names. The Section 4 configuration detects the project when the kernel starts in its root or `casework` folder and reads `raw` inside that project. For another launch location, set `PROJECT_DIR` to the extracted project folder. If relocated, copy the project as a unit, recreate its Python environment, and verify the selected kernel at the destination. See START-HERE.txt for Windows and macOS/Linux setup commands; the pinned environment has been tested on Windows only. Keep the source until the destination is verified.

## A project README

Keep one short README per substantial project. It should answer:

1. What question does this project investigate?
2. What files and data are supplied, and where did they come from?
3. Which environment and commands or notebook cells run it?
4. What has been completed and verified?
5. What is the next step, and what remains uncertain?

Put detailed explanations beside the relevant notebook cells. This plan is temporary setup documentation; an ordinary small assignment does not need a collection of planning files.

## Git: a history of understood changes

Use one local Git repository per substantial coding assignment. Git records selected file versions. GitHub is an optional hosting service to introduce later; a local repository does not require an account.

Before the first snapshot, inspect which files Git would include:

- Include original code, the working notebook, README, environment specifications, and small results needed to explain the work.
- Exclude `.venv`, caches, notebook checkpoints, bulk datasets, and large generated files. Record how to obtain or regenerate excluded files.
- Decide explicitly whether supplied course materials belong in the history. Keep access and submission rules with the course information.

The current phenology `.gitignore` excludes `.venv`, `__pycache__`, and `.ipynb_checkpoints`. It does not yet exclude `raw`; review this before initializing and staging the project.

Start with four concepts: status, diff, staging, and commit. A diff shows what changed; staging selects changes for a snapshot; a commit records that snapshot with a message. Learn history and recovery on a harmless practice change. Add branches and collaboration when an actual task needs them.

A local Git repository is not a separate backup. Choose a backup method for the full project, including data that Git excludes, after the permanent location is settled.

## The working loop

1. State the small question or change in plain language.
2. Identify the inputs and expected output.
3. Predict what will happen.
4. Make one change that the student can explain.
5. Run an appropriate check and inspect the result.
6. Explain the result in the notebook, including surprises or limitations.
7. Review the changed files and save a Git checkpoint at a meaningful milestone.

For an experiment, record the hypothesis, the one intended change, fixed conditions such as the split and seed, observed results, and interpretation. An unexpected or worse result is still evidence.

## Teaching sequence

### 1. Find and run the project

Learn folders and file paths, the VS Code Explorer, the terminal's current directory, the Python environment, and the notebook kernel. These are different concepts; explain each in the context of the existing project. Run one cell, inspect its output, write a short explanation in Markdown, and save.

Success: the student can reopen the correct project, find the notebook, select its environment, and explain what ran.

### 2. Make and inspect a checkpoint

Review ignore rules, create an initial local Git snapshot, make a small understood edit, inspect the diff, and commit it. Explain the difference between saving a file and committing it.

Success: the student can identify what changed and find an earlier snapshot without destructive commands.

### 3. Complete one small scientific calculation

Explain samples, features, labels, and array shapes. Implement chromatic coordinates, check a tiny known example, apply the function to an image, and explain the result. Learn Python syntax as it becomes necessary.

Success: the student can explain each line and reproduce the calculation independently.

### 4. Build and evaluate the first classifier

Extract one trajectory per pixel. Explain the split, establish a simple baseline, and train K-NN on raw time series. Use validation data to choose settings and reserve test data for final evaluation.

Success: the student can explain what the model receives, what it predicts, and how the score was obtained.

### 5. Compare representations and models

Explain interpolation and continuous recurrence plots with small examples before building the full representation. Complete the required K-NN comparison, then add other classifiers using the same split and evaluation rules.

Success: the student can justify each comparison and discuss accuracy, class-level errors, storage, and computation.

### 6. Prepare work another person can review

Restart the notebook kernel and run the notebook from the beginning. Check that recorded explanations match actual outputs. Provide a concise account of the question, method, results, limitations, and next experiment. Follow the actual submission instructions.

Success: another person can understand the work and reproduce its documented results.

## Phenology facts to carry forward

- After copying into the permanent project, provided setup, image/mask loading, ROI splitting and both supplied plots were rerun with its new Python environment. The classifier exercises are now implemented and verified; see `results/verification.json`. The earlier `setup-check.json` records only the original setup checks.
- The supplied noon-image sequence has 37 days. Parts of the written exercise refer to 35 days and a 140-point target. The supplied slides explicitly describe 37 daily noon images (slide 9) and interpolation from 37 to 140 points (slide 14). A reported classroom suggestion of 255 remains ambiguous between time points and image-value scaling.
- Classification requires one sample per pixel. Whole-ROI averages are useful for plots but represent a different sample unit.
- The fixed split contains one training ROI and one test ROI per species. Its selected masks share no pixel coordinates. Pixel validation within training ROIs can still be optimistic about performance on new ROIs.
- Reuse the same sample split for raw-series and recurrence-plot comparisons. Fit learned preprocessing using training data only.
- Increasing the interpolation grid adds estimated values rather than independent observations. Consider the memory cost before building all recurrence plots.
- The written specification requires K-NN on both representations and a metric-based comparison. Other classifiers extend the exercise.
- The inspected written specification and 20 supplied slides do not state a submission format or Ricardo's personal tooling preferences. Confirm the assessment instructions before packaging the final work. The slides have now been text-extracted; their image/matrix, acquisition, ROI-partition, and recurrence-plot diagrams were also visually inspected.

## What makes the work useful to a collaborator

Demonstrate an understandable question, a defensible evaluation, reproducible code, clear figures and explanations, proper attribution, and a record of limitations and failed approaches. These give a potential collaborator concrete work to assess. Coauthoring would depend on an actual research contribution and agreement with the collaborator.

For startup work, carry forward the habits of defining a problem, making a small working change, checking evidence, and revising assumptions. Learning to build for users will also require conversations with users and experience shipping software.

## Resources to use as needed

- [VS Code: source control](https://code.visualstudio.com/docs/sourcecontrol/overview)
- [VS Code: Jupyter notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [Git book: version control](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control)
- [MIT's Missing Semester](https://missing.csail.mit.edu/) covers practical computing tools. Use selected lessons as they become useful; it is not a prerequisite for continuing the assignment.

## Next study step

Review the completed notebook in order, starting with Exercise 2's preparation/split
correction. Explain what one pixel row, its species label and its saved identity mean.
Then inspect validation versus test performance and review the draft reflection.
Compare the finished work with the full paper rubric when available and confirm the
submission format. Keep `casework/case_notebook.ipynb` as the single working notebook.

The earlier setup/teaching sequence above is retained as a learning plan. A new Git
checkpoint remains a separate future task; cache and temporary directories are now
ignored, but raw-data inclusion still requires review before initializing a repository.
