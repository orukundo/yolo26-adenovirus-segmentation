# YOLO26 Adenovirus Segmentation
YOLO26 semantic-segmentation benchmark for adenovirus in transmission electron microscopy (TEM) images.

## Overview

This trains YOLO26n/s/m/l/x semantic-segmentation model variants using a selected augmentation profile, preparing the segmentation dataset from image folders and binary masks, evaluating each model on validation and test splits, saving the best-performing weights, and exporting predictions and comparison metrics.

## Requirements

- Python 3.x
- PyTorch
- Ultralytics YOLO26
- OpenCV
- NumPy
- pandas
- scikit-learn

## Author

Olivier Rukundo, Ph.D.
University of Eastern Finland
