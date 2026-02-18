# ⭐ StarsPrediction

### Deep Learning Pipeline for Stellar Property Inference from Photometric Time Series

StarsPrediction is a full-stack scientific framework for large-scale stellar light-curve preprocessing, deep learning–based inference, and production-grade deployment via a modern web application.

The repository integrates:

* ⚡ High-performance light curve preprocessing
* 🧠 Deep neural network training & optimization
* 🌐 Production-ready React + Python application
* ☁️ Cloud-based preprocessing and model selection (GCP)
* 🗄 Persistent database storage of user detections

The system is designed for both **research reproducibility** and **real-world deployment**.

---

# 📂 Repository Structure

```
StarsPrediction/
│
├── LightCurveData/
├── StarBeyond/
├── stellarscopeApp/
├── run_stellarscope.sh
└── README.md
```

---

## 🔹 LightCurveData/

High-performance preprocessing pipeline for photometric time-series data.

**Core features:**

* Parallelized Kepler/TESS light-curve retrieval
* Data cleaning & normalization
* FFT-based spectral feature extraction
* Window segmentation (e.g., 27-day, 97-day baselines)
* Signal conditioning and SNR-aware filtering

This module transforms raw astrophysical time-series data into structured tensors ready for deep learning.

---

## 🔹 StarBeyond/

Deep learning modeling, training, and evaluation framework.

**Includes:**

* 1D CNN architectures
* RCNN (Recurrent Convolutional Neural Networks)
* Transfer learning pipelines
* Automated hyperparameter search (Grid Search)
* Training optimization & evaluation tools
* Model export utilities

Designed for:

* Stellar parameter regression
* Efficient short-baseline inference
* Scientifically reproducible training workflows

---

## 🔹 stellarscopeApp/

Production-grade web application for deploying trained models.

**Architecture:**

* ⚛ React Frontend
* 🐍 Python Backend (API layer)
* ☁️ Google Cloud (GCP) for preprocessing & model selection
* 🗄 Persistent database for storing user predictions

**Capabilities:**

* Upload light curves
* Run model inference
* Select different trained models
* Store and retrieve user detection history
* Production-ready containerized deployment

---

# 🚀 Running StellarScope (Recommended)

The easiest way to run the full application stack is via Docker using the provided wrapper script.

---

## 🐳 Requirements

* Docker installed

  * Docker Desktop (Mac / Windows)
  * Docker Engine (Linux)

Verify installation:

```bash
docker info
```

If Docker requires sudo (common on Linux), the script automatically detects and handles it.

---

## ▶️ Quick Start

Make the script executable:

```bash
chmod +x run_stellarscope.sh
```

Start the application:

```bash
./run_stellarscope.sh start
```

After startup:

```
http://localhost:8080
```

---

## 🛑 Stop the Application

```bash
./run_stellarscope.sh stop
```

---

## 🔄 Restart

```bash
./run_stellarscope.sh restart
```

---

## 📜 View Logs

```bash
./run_stellarscope.sh logs
```

---

# 🔎 What the Script Actually Does

The `run_stellarscope.sh` script:

1. ✅ Verifies Docker is installed
2. ✅ Checks whether Docker works without sudo
3. ✅ Falls back to `sudo docker` if required
4. ✅ Downloads the correct production `docker-compose.yml`
5. ✅ Creates required directories
6. ✅ Launches containers in detached mode
7. ✅ Handles logs, restart, and shutdown

This makes deployment reproducible and OS-agnostic.

---

# 🧪 Running Model Training Manually

If you want to train models locally:

```bash
cd StarBeyond
python train.py
```

For preprocessing:

```bash
cd LightCurveData
python preprocess_pipeline.py
```

(Refer to module-specific scripts for advanced configurations.)

---

# 🔬 Scientific Design Philosophy

This repository is built around three core principles:

1. **Reproducibility**
   Deterministic preprocessing + version-controlled training pipelines.

2. **Scalability**
   Parallel data ingestion and containerized deployment.

3. **Deployment-Readiness**
   Research models integrated into a production web interface.

---

# ☁️ Cloud Integration

StellarScope integrates:

* Google Cloud Platform (GCP)
* Remote preprocessing
* Model selection workflows
* Database persistence for user predictions

The architecture supports scaling from research experiments to public-facing applications.

---

# 📊 Typical Workflow

1. Retrieve and preprocess light curves (`LightCurveData/`)
2. Train and evaluate models (`StarBeyond/`)
3. Deploy selected model (`stellarscopeApp/`)
4. Run production via Docker wrapper
5. Store and analyze user predictions

---

# 🛠 Development Setup (Optional)

For local development without Docker:

### Backend

```bash
cd stellarscopeApp/backend
pip install -r requirements.txt
python app.py
```

### Frontend

```bash
cd stellarscopeApp/frontend
npm install
npm run dev
```

---

# 📦 Production Deployment

The repository is container-first.

The provided Docker workflow ensures:

* Environment consistency
* Version-controlled deployment
* OS-independent execution
* Simplified scaling

---

## 📜 License

This project is licensed under the Apache License 2.0.
See the LICENSE file for details.


---

# 📧 Contact

For collaboration, research inquiries, or technical questions:

**Shahriyar Nasa**
📩 [shahriyarnasa@gmail.com](mailto:shahriyarnasa@gmail.com)

---

# 🌌 Final Note

StarsPrediction is not just a model repository.
It is an end-to-end scientific inference pipeline — from raw astrophysical signals to deployable AI systems.

If you're serious about reproducible astrophysical ML, this is built for you.

---
