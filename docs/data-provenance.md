# Data provenance

Where every dataset came from, how it was checked, and what it holds. No image is
stored in this repository. The preprocessed copies are private Kaggle datasets.

## EyePACS images and EyeQ labels

- **Images:** Kaggle competition *Diabetic Retinopathy Detection* (EyePACS, 2015).
  `train.zip` in five parts (~35 GB, 35,126 images, 17,563 patients) and `test.zip` in
  seven (~53 GB). Read in place with `src/data/archive.py`, because the notebook disk
  is capped at 20 GB.
- **Quality labels:** EyeQ (Fu et al. 2019), github.com/HzFu/EyeQ, cloned at commit
  `1dcfabb08cd31045de1a4a3ee82e48f6833697aa`. `Label_EyeQ_train.csv` (12,543) and
  `Label_EyeQ_test.csv` (16,249): 28,792 images, every one matched to an archive member.
- **Patients:** 20,662. 12,532 contribute one image and 8,130 contribute two, so 56% of
  images sit in a correlated pair. EyeQ's own train/test partition shares no patient
  (checked: 0 overlaps).
- **Class balance:** training partition 8,347 Good / 1,876 Usable / 2,320 Reject;
  test partition 8,471 / 4,558 / 3,220.
- **Preprocessed:** 28,792 PNGs at 512 x 512, 6.8 GB, no conversion failures.
  Published as `eyeq-512-full`.

## DRIMDB

- Kaggle mirror `subhajournal/drimdb-diabetic-retinopathy-images-database`.
- Verified 216 images at class-folder level (Good 125 / Bad 69 / Outlier 22), matching
  Şevik et al. (2014). Outliers are not fundus images and were dropped, leaving 194.
- The upload nests iWeb gallery exports inside each class folder. They are excluded by
  folder depth.
- Native 760 x 570. No patient identifiers. Easy: good and bad differ grossly, so a high
  score here proves less than it looks.

## IDRiD

- Kaggle mirror `mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid`
  (licence shown there: CC BY-SA 4.0; the original release is CC BY 4.0).
- The upload merges two sources: 516 images at IDRiD's native 4288 x 2848, matching the
  published count (Porwal et al. 2018), and 1,032 at 4880 x 3862 of unknown provenance.
  Only the 516 native images are used.
- Train and Test both number from 0001, so IDs carry the split name.
- Disease grades 0–4: 168 / 25 / 168 / 93 / 62. Role: pathology control only.

## DeepDRiD

- Kaggle mirror `nancyhisham/deepdrid`, a copy of github.com/deepdrdoc/DeepDRiD (it
  ships that repository's LICENSE and README). Liu et al. (2022).
- Regular fundus images only (ultra-widefield is a different modality): 1,600 labelled
  images from 400 patients, 758 good / 842 not.
- Kept: Clarity, Field definition and Artifact sub-scores, the only human per-defect
  labels available to the project.

## FQS (Fundus Quality Score)

- Gong et al. (2025), released on figshare (~9 GB). Uploaded to Kaggle privately as
  `fqs-fundus-quality-scores`: `images/` (2,246 at 1024 x 1024) and `divisions/` only.
- Labels from `divisions/cross_validation-00-{train,val,test}.xlsx`, which together
  cover all 2,246 images once (1,572 + 225 + 449).
- 514 Good / 794 Usable / 938 Reject; MOS 1.0–95.3. The class balance is the reverse of
  EyeQ's, by the authors' design.
- No patient identifiers (EXIF stripped by the authors).

## External set, combined

4,556 images through the same preprocessing as EyeQ, no failures, 1.1 GB. Published as
`retina-external-512`.
