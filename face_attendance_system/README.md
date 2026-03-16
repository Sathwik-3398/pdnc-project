# Face Attendance System using Deep Neural Networks

A complete real-time Face Attendance System built using Deep Neural Networks via the `face_recognition` and OpenCV Python libraries. 

It handles automated capturing of student faces via a live webcam, computes and extracts the deep facial embeddings, recognizes identities in real-time, and sequentially marks attendance preventing redundant submissions.

## Features
- **Dataset Collection Module**: Effortlessly capture images using a webcam and Haar cascades for fast face alignment logic. Automatic folder structuring saves ~30 images smoothly.
- **Deep Neural Network Embedding**: Extracts powerful 128-dimensional face embeddings out of image datasets to guarantee extremely high-accuracy similarity metrics robust to brightness, poses, etc.
- **Real-Time Recognition System**: Maps captured live facial geometries mapping smoothly back into our memory encodings using highly performant cosine-distance algorithms overlaying Live confidence metrics directly on-feed.
- **Automated Attendance Log System**: Employs robust `Pandas` based CSV interactions to insert uniquely indexed (Name, Date, Time) logs guaranteeing NO Duplicate attendance counts! Logs natively to Excel compatible `.csv` formats dynamically per-day.

## Folder Architecture
```text
face_attendance_system/
│
├── dataset/             # Stores identity sub-directories containing dataset faces
├── models/              # Saved model parameters storing `encodings.pickle` deep embeddings
├── attendance/          # Final produced spreadsheets exported automatically locally labeled logically by individual dates
├── collect_faces.py     # Setup script to capture and prepare new unique identity profiles directly via UI webcam interaction
├── train_model.py       # Iterates datasets extracting Deep 128D Face Embeddings compiling final mathematical profile weights
├── recognize_faces.py   # Final Application loop reading webcam iteratively matching known face architectures drawing predictions natively
├── attendance_system.py # Core modular framework backend API isolating logic for safe and duplicate safe log-writing implementations using Pandas
├── requirements.txt     # Complete Python requirements list easily exportable via PIP implementations
└── README.md            # You are reading this! Detailed instructional breakdown document.
```

## System Setup & Guides

### 1. Pre-requirements Setup
Ensure that you have installed Python versions properly locally above `v3.7` ensuring pip environment integrations.
Also depending on OS please install `C++ Build tools`/`CMake` locally before using face recognition library.
`pip install CMake` often provides a good first step before PIP installing the `face_recognition` deep architectures depending on local environments natively.

### 2. Environment Dependencies
Setup PIP modules natively easily navigating terminal location exactly into codebase directory `face_attendance_system/` and execution of single unified PIP setup:
```bash
pip install -r requirements.txt
```

### 3. Execution Implementations

#### Step 1: Data Profiles Collections
To successfully introduce users into the tracking dataset please run command line instruction below swapping placeholder name with target User identifying naming nomenclature. Process requires webcam natively running for smooth interactions. Default counts are ~30 images per person ensuring healthy datasets without severe over-fitting instances natively.
```bash
python collect_faces.py --name "John_Doe" --count 30
```
*(Please hold attention to camera allowing minor diverse slight rotational movements!)*

#### Step 2: Extracting Face Mathematical Profiles & Export Training Models
Iterate seamlessly via dataset paths preparing numerical models for live deployments. Script fully manages iterations internally!
```bash
python train_model.py
```

#### Step 3: Recognizing Embeddings and Executing Logs Natively (Main Module)
Start the primary infinite live feedback processing rendering bounds & calculated metric confident percentages visually natively! Press Keyboard `q` intentionally allowing graceful cleanup routines! 
```bash
python recognize_faces.py
```

#### Step 4: Access Tracking Result Analytics logs cleanly!
Browse visually directly manually local into inner module `attendance/` where CSV outputs persist gracefully generated formatted labeled by Date formats Native.
Open inside visual handlers easily e.g., Spreadsheets ensuring Data Analytics natively.
