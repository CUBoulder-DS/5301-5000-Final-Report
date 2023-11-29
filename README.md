# DTSC 5301 and STAT 5000 Final Report

The final project for Data Science as a Field and STAT 5000.

<span style="background-color: Green">Website link: [https://cuboulder-ds.github.io/5301-5000-Final-Report/](https://cuboulder-ds.github.io/5301-5000-Final-Report/)</span>

## How to generally do work for Quarto STAT 5000 website and PDF

1. Install Quarto and `install.packages("rmarkdown")` if you haven't already.
2. Whenever you do git pull, always rebase: `git pull --rebase`.
3. Make a branch for any major work that will affect the actual report files (aka the `.qmd` files), and start your branch name with initials, e.g. `BJ_work1`.
4. ALWAYS keep your git updated, do `git pull` always before any commiting, pushing, branches, etc.
5. When pushing, do a force push.
6. cd into the `5000-final` folder
7. Run `quarto render` on the CLI/bash
8. Do a git pull, commit and push (do force push if needed)
9. Github will automatically turn the `docs/` folder file changes into the website

## How the website generation works

As Github workflows (`.github/workflows/`), there are 2 separate scripts:

1. A script for auto-generating the static site file from the `5301_Final_Report.ipynb` DTSC 5301 file.
2. Workflow for deploying static content to GitHub Pages; Run every time and only after the `docs/` directory is updated.

## Directory Structure

- **docs/**: Where the github site is being hosted from, and what gets generated when we export the .ipynb notebook / Quarto into a webpage HTML. DO NOT manually change or touch, will auto-update upon changes to `5301_Final_Report.ipynb`/`5000-final/`.
- **images/**: Where images should go.
- **data/**: Where data we are using should go.
- **5301_Final_Report.ipynb**: The actual file that gets turned into the webpage. Use this naming scheme only, when exporting from DeepNote export only as this file name and replace the existing file.
- **5000-final**: Where the Quarto code for the STATS 5000 report should go.
- **requirements.txt**: Update this with any python packages you use in `5301_Final_Report.ipynb`.

## Deepnote to github/website instructions:

1. Right-click on the `5301_Final_Report` file in the Deepnote sidebar, click on "Export as .ipynb"
2. Re-upload the file downloaded to Deepnote under the github repo folder in the sidebar, make sure you are overwriting the previous `5301_Final_Report.ipynb`.
3. Click on `Bhav's terminal #1` in the sidebar under the github repo (or create a new terminal, either way)
4. In that terminal execute:
   1. `cd DTSC-5301-Final-Report/`
   2. `git commit -am "YOUR MESSAGE HERE"`
   3. `git push`
5. OR. Repeat steps 2-4 locally if you have the repo checked out and know how to, same result of commiting+pushing from deepnote itself.


## Info on Files in /data
<div class="cell-output-display">
<table class="table table-sm table-striped small">
<colgroup>
<col style="width: 17%">
<col style="width: 63%">
<col style="width: 18%">
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">filenames</th>
<th style="text-align: left;">purpose</th>
<th style="text-align: left;">recommendations</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">Croped_ff_np.csv</td>
<td style="text-align: left;">Permutation evaluation (older version) for fairface, no preprocessing on cropped images. Updated this file to look at the same files as the uncropped dataset.</td>
<td style="text-align: left;">Remove from github data folder.</td>
</tr>
<tr class="even">
<td style="text-align: left;">MasterDataFrame.csv</td>
<td style="text-align: left;">Final master data file containing all input and output files</td>
<td style="text-align: left;">Keep as-is with no changes</td>
</tr>
<tr class="odd">
<td style="text-align: left;">crop_df_np.csv</td>
<td style="text-align: left;">Permutation evaluation for DeepFace, cropped images, no pre-processing</td>
<td style="text-align: left;">Retain; rename to PERM_DF_c_np.csv</td>
</tr>
<tr class="even">
<td style="text-align: left;">crop_df_p_mtcnn.csv</td>
<td style="text-align: left;">Permutation evaluation for DeepFace, cropped images, preprocessed with MTCNN backend.</td>
<td style="text-align: left;">Retain; rename to PERM_DF_c_p_mtcnn.csv</td>
</tr>
<tr class="odd">
<td style="text-align: left;">crop_df_p_opencv.csv</td>
<td style="text-align: left;">Permutation evaluation for DeepFace, cropped images, preprocessed with OpenCV backend.</td>
<td style="text-align: left;">Retain; rename to PERM_DF_c_p_opencv.csv</td>
</tr>
<tr class="even">
<td style="text-align: left;">cropped_UTK.csv</td>
<td style="text-align: left;">Permutation evaluation (older version), list of cropped files to perform evaluation.</td>
<td style="text-align: left;">Remove from github data folder</td>
</tr>
<tr class="odd">
<td style="text-align: left;">cropped_UTK_dataset.csv</td>
<td style="text-align: left;">Permutation evaluation (newest version), list of cropped files to perform evaluation.</td>
<td style="text-align: left;">Retain with no changes</td>
</tr>
<tr class="even">
<td style="text-align: left;">cropped_ff_p.csv</td>
<td style="text-align: left;">Permutation evaluation (older version), used older version of cropped images dataset.</td>
<td style="text-align: left;">Remove from github data folder.</td>
</tr>
<tr class="odd">
<td style="text-align: left;">joined_permutations.csv</td>
<td style="text-align: left;">Permutation evaluation (newest version), joined all permutation outputs from DeepFace and FairFace to a single file</td>
<td style="text-align: left;">Retain with no changes</td>
</tr>
<tr class="even">
<td style="text-align: left;">new_ff_c_np.csv</td>
<td style="text-align: left;">Permutation evaluation (newest version), FairFaice outputs for cropped images with no preprocessing</td>
<td style="text-align: left;">Retain; rename to PERM_FF_c_np.csv</td>
</tr>
<tr class="odd">
<td style="text-align: left;">new_ff_c_p.csv</td>
<td style="text-align: left;">Permutation evaluation (newest version), FairFaice outputs for cropped images with dlib preprocessing</td>
<td style="text-align: left;">Retain; rename to PERM_FF_c_p.csv</td>
</tr>
<tr class="even">
<td style="text-align: left;">new_ff_uc_np.csv</td>
<td style="text-align: left;">Permutation evaluation (newest version), FairFaice outputs for uncropped images with no preprocessing</td>
<td style="text-align: left;">Retain; rename to PERM_FF_uc_np.csv</td>
</tr>
<tr class="odd">
<td style="text-align: left;">new_ff_uc_p.csv</td>
<td style="text-align: left;">Permutation evaluation (newest version), FairFaice outputs for uncropped images with dlib preprocessing.</td>
<td style="text-align: left;">Retain; rename to PERM_FF_uc_p.csv</td>
</tr>
<tr class="even">
<td style="text-align: left;">non_normalized_DeepFace_uncropped_DF_all.csv</td>
<td style="text-align: left;">Final dataset of DeepFace Outputs (non-normalized)</td>
<td style="text-align: left;">Retain; rename to Master_DF_non_normalized.csv</td>
</tr>
<tr class="odd">
<td style="text-align: left;">non_normalized_FairFace_uncropped_FF_all.csv</td>
<td style="text-align: left;">Final dataset of FairFace Outputs (non-normalized)</td>
<td style="text-align: left;">Retain; rename to Master_FF_non_normalized.csv</td>
</tr>
<tr class="even">
<td style="text-align: left;">uncropped_DF_all.csv</td>
<td style="text-align: left;">Final normalized output for DeepFace - used to build MasterDataFrame.csv</td>
<td style="text-align: left;">Retain with no changes</td>
</tr>
<tr class="odd">
<td style="text-align: left;">uncropped_FF_all.csv</td>
<td style="text-align: left;">Final normalized output for FairFace - used to build MasterDataFrame.csv</td>
<td style="text-align: left;">Retain with no changes</td>
</tr>
<tr class="even">
<td style="text-align: left;">uncropped_UTK.csv</td>
<td style="text-align: left;">Permutation evaluation (older version) - source data file for iteration script</td>
<td style="text-align: left;">Remove from github data folder.</td>
</tr>
<tr class="odd">
<td style="text-align: left;">uncropped_UTK_dataset.csv</td>
<td style="text-align: left;">Permutation evaluation (newest version) - source data file for uncropped images in iteration script</td>
<td style="text-align: left;">Retain with no changes</td>
</tr>
<tr class="even">
<td style="text-align: left;">uncropped_df_np.csv</td>
<td style="text-align: left;">Permutation evaluation (newest version) - DeepFace uncropped images with no preprocessing</td>
<td style="text-align: left;">Retain; rename to PERM_DF_uc_np.csv</td>
</tr>
<tr class="odd">
<td style="text-align: left;">uncropped_df_p_mtcnn.csv</td>
<td style="text-align: left;">Permutation Evaluation (newest version) - DeepFace uncropped images with mtcnn preprocessing</td>
<td style="text-align: left;">Retain; rename to PERM_DF_uc_p_mtcnn.csv</td>
</tr>
<tr class="even">
<td style="text-align: left;">uncropped_df_p_opencv.csv</td>
<td style="text-align: left;">Permutation Evaluation (newest version) - DeepFace uncropped images with opencv preprocessing</td>
<td style="text-align: left;">Retain; rename to PERM_DF_uc_p_opencv.csv</td>
</tr>
<tr class="odd">
<td style="text-align: left;">uncropped_ff_np.csv</td>
<td style="text-align: left;">Permutation Evaluation (older version) - FairFace uncropped images with no preprocessing</td>
<td style="text-align: left;">Remove from github data folder.</td>
</tr>
<tr class="even">
<td style="text-align: left;">uncropped_ff_p.csv</td>
<td style="text-align: left;">Permutation Evaluation (older version) - FairFace uncropped images with dlib preprocessing.</td>
<td style="text-align: left;">Remove from github data folder.</td>
</tr>
</tbody>
</table>
</div>
