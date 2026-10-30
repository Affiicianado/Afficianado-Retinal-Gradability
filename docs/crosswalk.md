# Label crosswalk

The five datasets define image quality differently. This is how each was mapped onto
EyeQ's three levels: **0 Good** (fully gradable), **1 Usable** (gradable with
difficulty), **2 Reject** (not gradable). The raw label is always kept in
`quality_raw`; the mapping is applied in `src/data/load.py`.

| Dataset | Raw labels | Mapping | What is lost | How it is scored |
|---|---|---|---|---|
| EyeQ | 0 / 1 / 2 | reference scheme | nothing | 3-class |
| DRIMDB | good / bad / outlier | good → 0, bad → 2, outlier dropped (22 images that are not fundus photographs) | no Usable class | gradable vs ungradable |
| DeepDRiD | Overall quality 1 / 0, plus Clarity, Field definition, Artifact | 1 → 0, 0 → 2; sub-scores kept as real defect labels | no Usable class | gradable vs ungradable; sub-scores for transfer |
| FQS | qualityLevel 0 / 1 / 2, plus MOS 0–100 | used directly (same encoding as EyeQ) | nothing for classification | 3-class |
| IDRiD | none (all images treated as good quality) | all → 0, role = control | not applicable | predicted class rates only |

When a binary dataset is scored, a predicted Usable counts as gradable. A Usable image
can be graded, so calling it ungradable would punish the model for a clinically correct
answer.

## How each direction was checked rather than assumed

- **DeepDRiD.** The label direction is not stated where the files are. Images with
  quality 1 have higher mean Clarity (8.66 vs 7.11) and far lower mean Artifact (1.27 vs
  4.11), so 1 = good.
- **FQS.** Mean opinion score by mapped class: Good 82.0, Usable 69.1, Reject 40.9,
  rising in the right order. The paper's bands are Good 80–100, Usable 60–79,
  Reject 0–59. A few images sit outside their band, as the authors' labelling
  procedure allows. Class counts 514 / 794 / 938 against the paper's 516 / 793 / 937:
  the released split files differ by one or two images.
- **All datasets.** Three Good and three Reject images per dataset were plotted after
  preprocessing (acquire-external-datasets) to confirm the top row looks better than the
  bottom.

## Known gaps

- **IDRiD's quality label is our assumption**, not an IDRiD annotation. It is a
  disease-grading dataset whose images were captured carefully; we treat it as good
  quality so that a Reject rate rising with disease grade can be detected.
- **FQS's Usable boundary is not EyeQ's.** The FQS authors report a higher error rate
  on Usable when they applied their own model to EyeQ, which they attribute to the two
  datasets' different quality criteria (Gong et al. 2025).
- **FQS was resized twice** (by the authors, 1942 → 1024; by us, 1024 → 512). Applied to
  every image alike, so it is systematic rather than tied to the label.
