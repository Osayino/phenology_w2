# Phenology — week 2

Working project: `C:\Study\2026-27\P2_AI\phenology_w2`.

The assignment compares K-nearest-neighbour classification of plant species using pixel colour time series and recurrence plot images.

## Start here

1. Extract the submission ZIP completely and open its project folder in VS Code. For a fresh environment, follow [START-HERE.txt](START-HERE.txt), which provides Windows and macOS/Linux setup commands.
2. Open `casework/case_notebook.ipynb`. This is the one working notebook to continue editing and eventually submit.
3. Select the recreated `.venv` environment as the notebook kernel. Python 3.11.15 and the pinned dependencies were tested on Windows; macOS/Linux execution has not been tested.
4. Run cells from the top in order. Saved outputs do not load variables into a new session.

The Section 4 configuration cell finds the project when the kernel starts in the project root or its `casework` folder, then checks the image and mask directories under `raw`. For another launch location, set `PROJECT_DIR` in that cell to the extracted project folder. Check the printed input locations. Save changes to the notebook using Save/Ctrl+S.

## Files

This table describes the full working folder; the Submission section lists the ZIP contents.

| Path | Purpose |
|---|---|
| `casework/case_notebook.ipynb` | Working assignment: explanations, code and verified outputs |
| `casework/phenology_data.py` | Pixel extraction, shared dataset saving and recurrence cache |
| `casework/knn_utils.py` | Exact Euclidean K-NN in bounded memory, deterministic ties |
| `results/` | Saved pixel dataset, predictions, metrics, verification and figures |
| `cache/` | Regenerable full recurrence arrays (about 1.53 GiB; excluded from Git) |
| `checks/` | Executable checks for data, nearest neighbours and experiment results |
| `experiments/` | Optional model comparison, validation selection, verification and plotting |
| `raw/images/1/` | Supplied photographs, kept unchanged |
| `raw/masks/` | Supplied region masks, kept unchanged |
| `reference/` | Original notebook, specification and slides |
| `visuals/` | Supporting learning illustrations |
| `.venv/` | Local Python environment; do not submit it |
| `requirements.txt` | Direct Python dependencies |
| `requirements-lock.txt` | Exact dependency versions used for this environment |
| `requirements-experiments-lock.txt` | Pinned optional environment, including PyTorch, torchvision and XGBoost |
| `setup-check.json` | Historical setup and data-loading checks; current experiment checks are in `results/verification.json` |
| `COURSE-WORKFLOW.md` | Learning plan and agreed course organisation |

Keep `casework` and `raw` together inside the project. Moving or extracting the whole folder preserves automatic path detection when launched from the root or `casework`; use the `PROJECT_DIR` override for other launch locations. Recreate the Python environment on the receiving computer.

## Assignment status — 11 November 2026

The remaining notebook exercises are implemented and a fresh Python kernel executed
all 32 code cells in order without errors. Saved results
were independently checked against predictions, sample identities and cache checksums.
The reflection is a draft for student review. The full assessment rubric and exact
submission requirements are still pending; completion is not a claim of a rubric score.

Exercise 2 now prepares individual-pixel RGBcc histories before creating the shared
split. This moves the preparation originally described in Exercise 3 ahead of its
consumer. Exercise 3 uses those histories; Exercise 4 generates recurrence images in
the same row order and uses the same saved indices. Run the notebook from top to bottom.

The experiment uses all 37 noon observations (DOY 242–278), despite the template's
35-day wording. It preserves the original six training and six test ROIs. The
13,745 training-ROI pixels split into 10,996 training and 2,749 validation pixels;
all 14,105 test-ROI pixels remain held out. The 22 ROI-average images from earlier
sections are kept separately as regional summaries.

| Representation | Selected K | Validation accuracy | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| Raw RGBcc (111 features) | 3 | 0.9287 | 0.2654 | 0.2298 |
| Recurrence images (58,800 features) | 1 | 0.7639 | 0.2303 | 0.2227 |

Both models use exact Euclidean distances, uniform votes, identical candidate K values
and identical pixel membership. K is selected only on validation; final fitting retains
the 10,996 training pixels as in the template. Distance ties use original row index;
vote ties use the smallest class ID; validation-score ties use the smallest K.
The recurrence implementation uses verified symmetry/zero diagonals to remove duplicate
arithmetic, restoring full-image Euclidean distances exactly. It does not sample pixels
or approximate neighbour search.

The validation/test gap and low test scores are results to interpret, not grounds for
tuning on test labels. Validation shares training ROIs and nearby pixels are correlated;
the different test ROIs are a harder generalization setting. See the notebook discussion
for limitations and unresolved instructor questions.

## Optional classifier extension — 13 November 2026

The notebook also includes an exploratory comparison using the same 111 RGBcc
features and the same training, validation and test pixels as the raw K-NN baseline.
Its fixed shortlist contains two `StandardScaler` + RBF `SVC` pipelines (`C=1` and
`C=10`, both with `gamma="scale"` and balanced class weights), plus a random forest
with 150 trees, `min_samples_leaf=2`, `max_features="sqrt"`, balanced class weights
and the same random seed. Scaling is fitted only on training pixels.

The candidate with the highest validation accuracy is selected; a tie keeps the
first declared candidate. Only the selected candidate is evaluated on the test
regions, using its original training fit. The selected random forest reached
99.42% validation accuracy and 34.12% test accuracy, compared with 26.54% test
accuracy for raw K-NN (+7.59 percentage points). Test macro F1 increased from
0.2298 to 0.2691. The baseline test outcomes were
already known when this exploratory extension was designed, so confirmation on
additional independent regions would be needed.

The saved extension outputs are `results/extension_results.json`,
`results/extension_predictions.npz` and `results/extension_comparison.png`.
These separate files preserve the required K-NN comparison. The completed 32-cell
run and scores documented above are the historical 11 November baseline run.

## Broader optional model comparison

The notebook's final comparison covers six additional approaches: SVM, a
multinomial logistic GLM, XGBoost, Bayesian classification (Gaussian naive Bayes
and a tree-augmented Bayesian network), pretrained CNN transfer features, and
pretrained vision-transformer transfer features. It includes the two original
K-NN models and the earlier random forest for reference. The image models use
frozen pretrained features with small classifiers; they are not trained from
scratch on these pixels.

Each new model's configuration is selected using the existing validation split,
then recorded before evaluating the same 14,105 test pixels. The comparison is
exploratory: earlier test outcomes were already known, and validation pixels
share training regions. The table reports ordinary accuracy, balanced accuracy
(macro recall), and macro F1 so class imbalance is visible.

By default, the notebook verifies and displays the saved comparison. This needs
only the core environment; PyTorch and XGBoost are not required to view it or
check its saved predictions. To reproduce the optional training, first run the
core notebook to prepare its dataset and recurrence cache, then create a separate
Python 3.11 environment from the project root:

```powershell
python -m venv .venv-experiments
.\.venv-experiments\Scripts\python.exe -m pip install -r requirements-experiments-lock.txt
.\.venv-experiments\Scripts\python.exe -B experiments/run_comparison.py --fit
```

On macOS/Linux, use `.venv-experiments/bin/python` for the last two commands;
these platforms have not been tested. With that environment activated, the run
command is `python -B experiments/run_comparison.py --fit`. Budget roughly one
additional CPU hour, depending on the computer, and additional disk space for
the larger environment and feature caches. Pretrained weights need internet
access on their first download. The runner saves new comparison files separately
from the original K-NN and random-forest results. Reopen or rerun the final
notebook cells to display newly generated comparison results.

The durable comparison files are `results/model_comparison.json`,
`model_comparison_predictions.npz`, `model_comparison_selection.json`,
`model_comparison_validation.json` and `model_comparison.png`. Local fitted
models and embedding caches under `experiments/artifacts/` can be regenerated
with `--fit`; they are not needed to inspect or verify the saved results.

In the fixed 13 November comparison, multinomial logistic regression on original
RGB intensities had the highest observed test accuracy: **36.39%**, compared with
the random forest's **34.12%** (+2.27 percentage points). Its balanced accuracy was
**38.91%** and macro F1 **0.3146**. The notebook reports every model, including the
weaker outcomes; its 100% validation accuracy did not carry over to new regions.

## Portability verification — 13 November 2026

The expanded notebook's **38 code cells** also completed in order in a fresh
core-only kernel in a relocated project in **708 seconds**
(about 11.8 minutes). It started without core results or
recurrence caches. The original predictions were reproduced exactly; the saved
advanced comparison was independently checked and displayed without PyTorch,
XGBoost, downloaded weights or fitted-model artifacts. The heavier training
was completed separately in the optional environment. All 60 original notebook
cells, including their outputs and metadata, remain intact in the main notebook.

Before the broader comparison was added, all 35 code cells completed in order in a fresh kernel in a relocated project,
starting without results or recurrence caches. The run took 859 seconds (about
14 minutes). Every original K-NN prediction, neighbour index and metric matched
the saved baseline. The working notebook retains its original baseline outputs;
the optional extension has the new run's outputs. Original timing figures are
labelled as historical measurements.

A separate clean Python 3.11.15 environment installed all 50 pinned packages from
the local package cache, passed dependency compatibility checks, and passed all
four numerical/data checks. These tests were on Windows. The full notebook run
used the project environment with the same pinned versions. macOS/Linux have
not been executed here.

## Saved outputs and rerunning

- `results/pixel_dataset.npz`: raw features, species labels, pixel identities, days and split indices.
- `results/predictions.npz`: aligned test predictions and neighbour indices for checking votes and K selection.
- `results/experiment_results.json`: metrics, confusion matrices, settings, cache identifiers and timings.
- `results/verification.json`: latest successful result checks and executable-source hashes.
- `results/pixel_examples.png`, `comparison.png`, `confusion_matrices.png`: inspected figures.
- `cache/rp_*.npy` and adjacent JSON: complete uint8 recurrence arrays and provenance/checksums.

The full raw feature arrays require 11.79 MiB; the recurrence arrays require 1.525 GiB,
excluding labels and working memory. Caches are reused only after content/signature and
checksum checks. They can be regenerated from the notebook and source data. Full K-NN
is recomputed on a notebook rerun; budget for its measured computation time, which is
recorded in `experiment_results.json`. Timings use two BLAS threads for raw features,
four for recurrence features, and bounded row blocks; they are implementation-specific.

Run the checks from the project root:

```powershell
.\.venv\Scripts\python.exe checks\check_transformations.py
.\.venv\Scripts\python.exe checks\check_pixel_data.py
.\.venv\Scripts\python.exe checks\check_knn.py
.\.venv\Scripts\python.exe checks\check_saved_data.py
.\.venv\Scripts\python.exe checks\verify_results.py
```

The saved-data and result checks require a completed notebook run. The first three use
small independent examples and do not regenerate the full experiment.
`checks/verify_results.py` also checks the broader comparison when its result
files are present. To verify only that comparison, run
`.\.venv\Scripts\python.exe -B experiments/verify_comparison.py`; this does not
load fitted models, require the optional environment, or change result files.

## Submission

The working submission notebook is `casework/case_notebook.ipynb`. Do not maintain another draft labelled “final” while still working.

Before handing it in, restart its kernel, run the completed notebook from the beginning, inspect results and explanations, and save it. Follow Ricardo's submission instructions for naming and any required accompanying files. The notebook reads photographs and masks from `raw`; keep the expected `casework`/`raw` layout when extracting the project.

The submission ZIP is `dist/phenology_submission.zip`. It includes
the main notebook, both helpers (`casework/phenology_data.py` and
`casework/knn_utils.py`), the 37 noon JPEG images, all 22 ROI masks, requirements,
setup instructions and saved results. The 1.53 GiB recurrence cache is regenerated
automatically and does not need to accompany the notebook.

The ZIP omits `.venv`, caches, temporary files and unrelated data. Internet access
is needed for dependency installation, not to download the included assignment
data. Allow at least 4 GB of free disk space and around 20 minutes for a full run;
runtime depends on the computer and whether a usable cache already exists.
The optional comparison scripts, dependency specifications and saved comparison
are included. Refitting the additional models is a separate, longer run described
above; the notebook displays their saved results by default.

After verifying any later notebook changes, rebuild the ZIP from the project root
with `.\.venv\Scripts\python.exe -B checks/build_submission.py`. The builder uses
an explicit file list and checks every archived file against its source.

## Recreate the Python environment if needed

See [START-HERE.txt](START-HERE.txt) for Windows and macOS/Linux instructions.
With Python 3.11 installed, the Windows commands from the project root are:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Then select the new `.venv` in the notebook. Recreate environments after relocating a project rather than assuming an old `.venv` can be copied safely.

The earlier project in `C:\Users\User\Documents\Codex\2026-11-04\t\outputs\phenology-vscode` is retained as a backup. Continue coursework in this `C:\Study` project to avoid diverging copies. This project now has a Git repository; local environments, recurrence caches and optional fitted-model artifacts are ignored.
