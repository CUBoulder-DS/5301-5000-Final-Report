# DTSC 5301 and STAT 5000 Final Report

The final project for Data Science as a Field and STAT 5000.

<span style="background-color: Green">Website link: [https://cuboulder-ds.github.io/5301-5000-Final-Report/](https://cuboulder-ds.github.io/5301-5000-Final-Report/)</span>

## How the website generation works

As Github workflows (`.github/workflows/`), there are 2 separate scripts:

1. A script for auto-generating the static site files from the `5301_Final_Report.ipynb` DTSC 5301 file and the `5000-final` Quarto project directory.
2. Workflow for deploying static content to GitHub Pages; Run every time and only after the static site build runs.

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
