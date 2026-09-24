# Deepfake Thesis

Code and experimental pipeline developed for a Master's thesis on deepfake detection in static facial images, with particular attention to detector reliability, robustness to image degradation, confidence calibration, selective classification (abstention), and human-in-the-loop support.

The experimental pipeline compares Xception and EfficientNet-B4 on FaceForensics++ C23 and includes preprocessing, baseline training, robustness analysis, robust retraining, confidence calibration, selective classification, Grad-CAM visualizations, technical/forensic indicators, and a final multi-face Streamlit application.

## Repository structure

```text
deepfake-thesis/
├── face_detector/
│   └── face_detection_yunet_2023mar.onnx
├── metadata/
│   ├── official_splits/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   ├── dataset_splits.csv
│   ├── faces_metadata.csv
│   └── frames_metadata.csv
├── notebooks/
│   └── colab/
│       ├── calibration_analysis.ipynb
│       ├── selective_classification_analysis.ipynb
│       ├── selective_classification_degraded_analysis.ipynb
│       ├── train_efficientnet.ipynb
│       ├── train_xception.ipynb
│       └── transformation.ipynb
├── src/
├── src_multiface/
├── .gitignore
├── app_multiface.py
├── requirements.txt
├── requirements_colab.txt
└── README.md
```

`src/` contains the main experimental pipeline.  
`src_multiface/` contains the multi-face extension used by the final technical-report application.  
`notebooks/colab/` documents the Google Colab workflows used for training and the main experimental analyses.

## Experimental scope

The task is binary classification:

- `0` = REAL
- `1` = FAKE

The fake class combines the four FaceForensics++ manipulation methods used in this experiment:

- Deepfakes
- Face2Face
- FaceSwap
- NeuralTextures

The experiments use the C23 compression setting.

Ten frames are extracted from each video. Face detection and cropping are performed with YuNet. The official FaceForensics++ train/validation/test split definitions are preserved at video level, so frames derived from the same source sequence are not assigned to different splits.

## Dataset

The FaceForensics++ videos are not included in this repository.

For independent reproduction, FaceForensics++ should be obtained through the official project and used according to its access conditions:

https://github.com/ondyari/FaceForensics

The preprocessing script expects the following local directory structure:

```text
data_raw/
└── archive/
    └── FaceForensics++_C23/
        ├── original/
        ├── Deepfakes/
        ├── Face2Face/
        ├── FaceSwap/
        └── NeuralTextures/
```

The official split files used by the project are stored in:

```text
metadata/official_splits/
```

The generated frames and facial crops are not stored in the GitHub repository because of their size.

## Environment

Two environments were used during the thesis.

### Local environment

The local development environment was based on:

- Windows 11 Pro
- Python 3.12.10
- PyTorch 2.13.0+cpu
- torchvision 0.28.0+cpu
- timm 1.0.28
- OpenCV 5.0.0
- NumPy 2.5.2
- pandas 3.0.5
- scikit-learn 1.9.0
- Matplotlib 3.11.1
- Pillow 12.3.0

The project virtual environment also contains:

- Streamlit 1.64.0
- tqdm 4.70.0
- ReportLab 5.0.1

Install the local dependencies with:

```bash
pip install -r requirements.txt
```

Streamlit is used by the final interface, `tqdm` by evaluation scripts for progress reporting, and ReportLab by the PDF report exporter.

### Google Colab environment

Training and computationally intensive evaluation were performed on Google Colab using an NVIDIA Tesla T4 GPU with CUDA 12.8.

The thesis records the following main Colab environment:

- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- timm 1.0.28
- OpenCV 5.0.0
- NumPy 2.1.3
- pandas 2.2.3
- scikit-learn 1.6.1
- Matplotlib 3.10.0
- Pillow 11.3.0

The corresponding package requirements are documented in:

```text
requirements_colab.txt
```

The notebooks under `notebooks/colab/` document the execution workflow used during the experiments.

### Colab path conventions

Some evaluation scripts and notebooks use the project paths adopted during the thesis, including:

```text
/content/drive/MyDrive/deepfake-thesis
/content/deepfake-thesis
```

and, for some robustness evaluations:

```text
/content/deepfake-thesis/faces
```

If a different Google Drive or Colab directory layout is used, these paths must be adapted accordingly.

## Preprocessing

The preprocessing pipeline is executed in the following order.

### 1. Frame extraction

```bash
python src/extract_frames.py
```

The script:

- reads the original and manipulated FaceForensics++ C23 videos;
- extracts 10 temporally distributed frames from each video;
- stores them as PNG files;
- creates `metadata/frames_metadata.csv`.

Generated frames are saved under:

```text
data_processed/frames/
```

### 2. Face detection and cropping

```bash
python src/crop_faces.py
```

The script uses YuNet to detect the main facial region and produces:

```text
data_processed/faces/
metadata/faces_metadata.csv
```

The detector model used by the project is:

```text
face_detector/face_detection_yunet_2023mar.onnx
```

### 3. Dataset splits

```bash
python src/create_splits.py
```

This script combines the frame metadata, face-detection metadata and official FaceForensics++ split files to generate:

```text
metadata/dataset_splits.csv
```

The split assignment is performed at video level.

## Baseline training

The two baseline classifiers are Xception and EfficientNet-B4.

A smoke test can be executed before complete training:

```bash
python src/train_xception.py --smoke-test
python src/train_efficientnet.py --smoke-test
```

The complete baseline training used batch size 32 and two DataLoader workers in Colab:

```bash
python src/train_xception.py --batch-size 32 --num-workers 2
python src/train_efficientnet.py --batch-size 32 --num-workers 2
```

The best checkpoints are selected according to the minimum validation loss. The resulting checkpoint filenames are:

```text
xception_baseline_best.pth
efficientnet_b4_baseline_best.pth
```

The Colab execution used for the thesis is documented in:

```text
notebooks/colab/train_xception.ipynb
notebooks/colab/train_efficientnet.ipynb
```

## Robustness evaluation

The baseline models are evaluated under controlled benign degradations:

- JPEG compression: quality 90, 70 and 50;
- isotropic resize: 75%, 50% and 25%, followed by restoration to the original size;
- Gaussian blur: 0.5, 1.0 and 2.0.

Relevant source files are located in `src/`, including the robustness transforms, robustness datasets, evaluation scripts and prediction-stability analysis.

The workflow used for these experiments is documented in:

```text
notebooks/colab/transformation.ipynb
```

Robustness in this project refers to stability under these controlled transformations. It should **not** be interpreted as evidence of generalization to unseen datasets, manipulation methods or generators.

## Robust retraining

Xception and EfficientNet-B4 are retrained using degradation-based data augmentation.

The corresponding scripts are:

```text
src/train_xception_robust.py
src/train_efficientnet_robust.py
```

The selected robust checkpoints are named:

```text
xception_robust_best.pth
efficientnet_b4_robust_best.pth
```

The checkpoint files are not included in this repository. They can be regenerated by running the robust-training pipeline.

`src/evaluate_robust_model.py` contains a smoke-test mode intended only to verify the evaluation pipeline on a small number of batches. For the complete robust-model evaluation, set:

```python
SMOKE_TEST = False
```

before running the full experiment.

## Calibration

Confidence calibration is performed using Temperature Scaling.

The calibration workflow is available in:

```text
notebooks/colab/calibration_analysis.ipynb
```

Calibration is evaluated with:

- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)
- Brier Score

Calibration results are specific to the experimental domain and should not be interpreted as universal probabilities of image authenticity.

## Selective classification

The project implements selective classification, allowing a detector to abstain when its calibrated confidence is below a threshold selected on the validation set.

The main workflow is available in:

```text
notebooks/colab/selective_classification_analysis.ipynb
```

The additional analysis under degraded conditions is available in:

```text
notebooks/colab/selective_classification_degraded_analysis.ipynb
```

The mechanism is evaluated through coverage and selective risk.

An `ABSTAIN` output means that the detector output does not satisfy the selected confidence criterion and that human review is recommended.

## Final multi-face application

The final interface is implemented with Streamlit:

```bash
streamlit run app_multiface.py
```

The application:

- accepts an uploaded image;
- detects multiple facial regions;
- analyses each detected face separately;
- runs Xception and EfficientNet-B4;
- applies calibrated confidence and selective classification;
- reports model agreement/disagreement;
- generates Grad-CAM visualizations;
- reports additional image/file indicators;
- displays EXIF and JPEG-related information when available;
- uses SHA-256 for integrity/traceability information;
- exports a technical PDF report.

The multi-face orchestration code is stored in:

```text
src_multiface/
```

while the stable single-face detector pipeline remains in:

```text
src/
```

### Checkpoints required by the application

The final application requires the trained robust Xception and EfficientNet-B4 checkpoints:

```text
xception_robust_best.pth
efficientnet_b4_robust_best.pth
```

These checkpoint files are not included in the repository. They can be regenerated through the robust-training scripts.

The report/model-loading code must point to the location in which the regenerated checkpoints are stored.

## Important limitations

This repository documents and enables reconstruction of the experimental system developed for the thesis, but its outputs must be interpreted within the limits of the experimental protocol.

In particular:

- the models were trained and evaluated on FaceForensics++ C23;
- reported results are based on facial frame/crop samples rather than aggregated video-level decisions;
- robustness experiments cover specific controlled JPEG, resize and blur transformations;
- robustness does not imply cross-dataset or cross-generator generalization;
- calibrated confidence is not a universal probability that an image is real or fake;
- Grad-CAM highlights regions associated with the model decision but does not prove or localize a manipulation;
- SHA-256 supports integrity and traceability of a file, not authenticity;
- the Streamlit application is intended as support for human assessment and does not provide an automatic authenticity certification.

## Third-party components

### YuNet

The repository includes the YuNet face detector model used in the thesis:

```text
face_detector/face_detection_yunet_2023mar.onnx
```

OpenCV Zoo YuNet directory:

https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

Exact model file:

https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

The files in the OpenCV Zoo YuNet directory are distributed under the MIT License.

### FaceForensics++

The experimental dataset is based on:

> Rössler, A. et al. (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images*. ICCV 2019.

Official project and dataset access information:

https://github.com/ondyari/FaceForensics

The FaceForensics++ data is distributed under the project's Terms of Use; the repository code is released under the MIT License.

## Thesis

This repository accompanies a Master's Degree thesis developed at the University of Bari, Degree Programme in Computer Science, Security Engineering curriculum.

The experimental objective is not to provide an automatic forensic verdict, but to study the reliability limits of deepfake detectors and to integrate robustness analysis, calibration, abstention and human review into a cautious technical-support workflow.
