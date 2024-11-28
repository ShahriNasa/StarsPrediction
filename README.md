# StarsPrediction

This repository provides tools and scripts for efficiently processing Kepler Light Curve data and training machine learning models to predict star properties using neural networks. The structure is designed to streamline the data retrieval, preparation, and model training processes.

## Repository Structure

- **LightCurveData/**  
  This directory contains Python scripts optimized for efficient data retrieval from the Kepler Light Curve dataset using parallel processing techniques. It also includes modules for data cleaning and preprocessing, specifically tailored to different star characteristics.

- **Star&Beyond/**  
  This folder houses the tools required to split and prepare the data for use in neural network models. Key features include:
  
  - **1D CNN and RCNN Models**: Scripts for training models using 1D Convolutional Neural Networks (CNNs) and Recurrent Convolutional Neural Networks (RCNNs).
  - **Transfer Learning**: Pre-trained models are available to facilitate transfer learning for enhanced performance.
  - **Automated Grid Search**: Simplifies hyperparameter tuning by utilizing Grid Search across various time periods (e.g., 27-day or 97-day data windows).
  - **Sample Data**: A small sample dataset is provided for quick model training and testing.

## Getting Started

To get started, clone this repository:

```bash
git clone https://github.com/Shahri70/StarsPrediction.git
```
## Contact

For any questions, feedback, or collaboration inquiries, please feel free to reach out via email:

[shahriyarnasa@gmail.com](mailto:shahriyarnasa@gmail.com)
