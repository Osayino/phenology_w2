# Whole-assignment review - 7 November 2026

**Status update — 11 November 2026:** The audit below is retained as the historical
7 November review. Its statements that Exercises 2–6 are unfinished have been superseded
by the completed implementation and fresh execution recorded in `README.md`,
`results/experiment_results.json` and `results/verification.json`. Pixel-level data and
the common split are now saved, both full-data classifiers are evaluated, and the
notebook contains a draft reflection. The interpretation caveats and open instructor
questions below remain relevant. The student reports having emailed Ricardo about the
exercise-order issue; no instructor reply is claimed here.

This is an audit of the supplied materials and the current work, not a completed student reflection or an additional assignment implementation. Existing notebook edits were saved for inspection. No new solution code, classifier training, or changes to the dataset or split were made during this review.

The governing objective is **pixel-wise plant-species classification using two representations of the same pixel histories**. The written specification asks for dataset construction, splitting, two K-NN evaluations, comparison, and critical reflection. The slides supply a worked example and emphasize representation, validation, computational cost, and generalization to different ROIs. A defensible comparison is the destination; recurrence plots are not required to outperform raw time series.

Sources reviewed:

- [Hands-on activity specification](reference/2026-11-04-hands-on-activity-specification.pdf), especially pages 1-4; all five pages were text-inspected and the relevant body pages visually inspected.
- [Hands-on activity slides](reference/2026-11-04-hands-on-activity-slides.pdf), especially slides 4, 8-14, and 16-19; all 20 pages were text-inspected, with relevant diagrams and result pages visually checked.
- [Supplied reference notebook](reference/2026-11-04-case_notebook_with_data.ipynb), all 45 cells.
- [Working notebook](casework/case_notebook.ipynb), 55 saved cells at the time of this review, including the actual implementations through Exercise 1.

**The sample must remain clearly defined.** One sample is one original camera pixel inside a selected ROI, observed across the same 37 days. Its species class comes from the ROI mask. It has either 111 chromatic-coordinate features (37 red, 37 green, 37 blue) or a 140 x 140 x 3 recurrence image, which supplies 58,800 features when flattened. The 37 days are not 37 classification samples. The 19,600 positions in a recurrence image are not new labeled samples either.

| Stage completed so far | Audit finding | Consequence |
|---|---|---|
| Image and mask loading | 37 noon images of shape 960 x 1280 x 3, 22 masks, six species | The loaded sequence and its ordering are the common foundation for both representations. |
| Section 4.3 ROI partition | Six training ROIs contain 13,745 pixels; six test ROIs contain 14,105 pixels. Their pixel coordinates do not overlap. | Preserve this spatial partition. The other ten available ROIs do not automatically become training or test data. |
| Section 5 RGB chromatic coordinates | Actual implementation passes known-value, range, nonblack sum-to-one, and input-preservation checks | Floating-point conversion avoids integer overflow. Black pixels are explicitly assigned three zeros; those are a handling convention for undefined ratios. |
| Section 6 ROI averages | Actual implementation returns three 37-value series for each of 22 ROIs and preserves species-label alignment | This correctly implements the notebook's ROI-mean instruction. Averaging removes individual pixel histories, so these summaries cannot substitute for pixel samples. |
| Exercise 1 interpolation and distance functions | Actual implementations pass endpoint/length, absolute-distance, symmetry, zero-diagonal, and constant-series checks | The functions are reusable for individual pixel series. Interpolation adds estimated positions, not additional observations. |
| Exercise 1 RGB construction | Scaling each distance matrix to 0-255, guarding constant channels, and RGB stacking produce valid uint8 images | Per-image/channel scaling removes absolute distance magnitude and integer conversion adds quantization. These are relevant representation choices to discuss later. |
| Exercise 1 current dataset and visualization | The current result is 22 ROI-average images, shape (22, 140, 140, 3), with one ROI example displayed per species | Valid region summaries have been built. The full individual-pixel dataset for the classifier comparison has not yet been built. |
| Exercise 2 in the saved working notebook | Its split cells still contain commented placeholders | There is not yet an implemented pixel split in this notebook to certify. The previously proposed preparation/split approach was checked separately, not installed here. |

Fresh execution of the actual saved code through Exercise 1 passed (19 code cells in a separate process). The original image stack was unchanged. The feature arrays for the later pixel classifiers were absent at the Exercise 2 boundary. These checks verify the implemented transformations, not classifier performance or future exercises.

**The 35/37 discrepancy is real and resolved operationally by the supplied data.** The specification and some notebook text say 35. The loaded dataset contains 37 consecutive noon observations, and slides 9 and 14 explicitly use 37. Using the actual input length and the specified target of 140 is consistent with the data and those slides. It should be documented rather than silently claiming that 35 observations were used.

**The ROI-versus-pixel distinction still needs careful wording.** The reference notebook's Section 6 asks for ROI means and its Exercise 1 dataset TODO says "for each ROI." However, the specification's objective and notebook Exercises 3-4 require pixel-wise classification. Slide 14, labelled Dataset Construction (Task 1), captions its examples "one pixel per species." Calling the 22 ROI images a preview was the assistant's interpretation, not an explicit instructor instruction. They are valid summaries, but should not be described as completion of the full pixel-level Task 1 dataset.

| Exercise | Purpose in the complete experiment | What must hold when it is finished |
|---|---|---|
| 1. Construct recurrence images | Provide the alternative representation of pixel histories | A recurrence image can be traced to the same pixel and species label as its raw-series counterpart. ROI summaries have a clearly separate role. |
| 2. Split the data | Define a fair and reproducible evaluation | Preserve the supplied training/test ROIs. Split only training-ROI pixels into training and validation. Use identical sample membership for both representations. |
| 3. K-NN on raw time series | Establish a baseline using the 111 RGBcc features | Tune K on validation data; evaluate the selected configuration on the held-out test pixels. "Raw" here means before recurrence transformation, not unnormalized JPEG intensities. |
| 4. K-NN on recurrence images | Evaluate the transformed representation under the same experimental protocol | Use the same pixels, labels, split, candidate-K protocol, and test set. Its best K may legitimately differ from the raw-series model's. |
| 5. Compare results | Determine what the representation change does to predictive performance | Compare matched test predictions using accuracy and consistently defined precision, recall, F1, and class supports. Document sampling, preprocessing, and fitting choices. |
| 6. Critical reflection | Explain the observed tradeoffs and limits using evidence | Connect results to representation, information changes, storage, runtime, validation design, and generalization. Separate observed results from proposed explanations. |

**Exercise 2's data requirements should be settled before proceeding.**

1. Each row identifies a particular camera pixel, with its ROI, coordinates, species label, and the same chronological observation axis.
2. Both representations derive from that same pixel history. Keep row order and label order aligned; the same random seed alone cannot repair different sample ordering.
3. Preserve the six training and six test ROIs from Section 4.3. For the full prescribed dataset, the 13,745 training-ROI pixels split 80/20 into 10,996 training and 2,749 validation pixels. All 14,105 test-ROI pixels remain held out.
4. Reuse one set of sample indices for both representations. If a computationally necessary subset is used, select and document a common subset for both methods rather than silently reducing only one.
5. Select K using validation results. Test results must not drive hyperparameter choice. Any learned cross-sample preprocessing must be fitted on training data; the current per-sample chromatic coordinates and per-RP scaling do not fit statistics across samples.
6. Preserve and report sample counts, species supports, shapes, feature ordering, seed, and representation settings so the later comparison can be checked.

The exact execution order remains a notebook organization choice. Exercise 2 explicitly refers to feature extraction in Exercises 3 and 4. Section 4.3 already establishes ROI membership, but does not construct the feature matrices used by the shown split calls. The original skeleton runs because its unfinished split calls are comments. Enabling only those calls without feature preparation produces missing-variable errors. Valid approaches include running the later preparation cells first, moving preparation earlier, or defining pixel-index splits before generating either representation. This is not evidence of a hidden LLM test or proof of faulty instructional intent.

**A comparison detail in the slides needs clarification.** The confusion matrices on slide 17 contain different test sample counts:

| Displayed representation | Test samples | Correct predictions | Displayed accuracy, reproduced from the matrix |
|---|---:|---:|---:|
| Raw time series | 14,105 | 3,699 | 0.2622 |
| Recurrence plots | 1,414 | 321 | 0.2270 |

The RP class counts are 146, 246, 153, 212, 301, and 356, approximately one tenth of the corresponding raw-series class counts. The inspected slides do not state how or why this sample selection was made. All four metrics on slide 18 match their respective matrices, including macro precision, recall, and F1. Therefore the different sample counts are not merely a PDF extraction artifact. They do not by themselves prove a bug, but the score difference cannot be attributed solely to representation without clarifying the sampling and training protocol. These displayed scores are not automatic targets that our implementation must reproduce exactly.

**Reflection should be anticipated now, not invented after the scores are known.**

- The transformation is deterministic. A larger image has more entries, not more independently measured evidence. Interpolating 37 observations to 140 positions does not collect more data.
- Pairwise differences remove a constant offset in a color series; per-RP scaling also removes absolute variation magnitude. Ask whether useful species information is lost alongside the temporal structure that becomes visible.
- K-NN on a flattened recurrence image does not learn spatial filters as a CNN would. A visually rich representation is not guaranteed to improve this classifier.
- Validation pixels drawn from the same training ROIs can resemble neighboring training pixels. Testing on different ROIs asks a harder generalization question. Slides 16-19 show a large validation/test gap and attribute it to spatial domain shift; that is a proposed explanation to assess, not something our future gap alone would prove.
- At the current settings, one raw-series sample contains 111 float32 values (444 bytes), while one RGB recurrence sample contains 58,800 uint8 values (58,800 bytes). That is about 530 times as many entries and 132 times the stored bytes at these particular dtypes. Across the 27,850 pixels in the selected training/test ROIs, RP storage alone is about 1.525 GiB before temporary copies or classifier conversions. Measure actual time and memory rather than assuming image conversion is free.
- Use matched accuracy, per-class errors, macro metrics, sample supports, and measured costs to support the final argument. Poorer performance can still be an informative and correctly executed result.

The useful instructor questions are now specific: what role should the ROI-average examples play relative to the per-pixel Task 1 dataset; what execution organization is preferred for the final notebook; and what sampling protocol produced the RP results in the slides, especially if full per-pixel RP K-NN is too costly. No question or comment was sent to the instructor during this review.
