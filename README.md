# OCR

## Install

Clone repo and install [requirements.txt](requirements.txt) in a Python>=3.8.0 environment.

```
git clone https://github.com/LapTQ/handwritten_text_recognition.git
cd handwritten_text_recognition
pip install -r requirements.txt
```

## Usage

**Note**: This repository should be run on local machine only because during execution we prompt some user interactive gestures, which might not be supported by other environments like Colab.

The script accept single image as input.

```
python3 run.py --input path/to/image --output path/to/folder
```

Arguments:
* `--input`: path to an image.
* `--output`: path to a directory where the output is stored (default to [`output`](output) if user does not specify). 

The output includes a `.json` file and a demo image (for visualization purpose). In addition, we save intermediate result in the [cache](cache) folder for convenience.

## Workflow

As for the flow of execution, see picture below.

![Flow](imgs/flow.jpg)

For fast pre-processing, in our project we use classical image processing techniques, leaving the text detection and recognition stages for deep learning.
* Text detection: CRAFT
* Text recognition: Transformer OCR by VietOCR.

For the details of techniques used in this project, please see [report](report.pdf).

## Contributors

This repository is following the course project in class IT4343E - Computer Vision. We would like to express our sincere gratitude to PhD. Dinh Viet Sang for guiding this course project.

Group members:
* Pham Thanh Hung - 20194437
* Hoang Nguyen Minh Nhat - 20194445
* Pham Thanh Truong - 20194460
* Tran Quoc Lap - 20194443

