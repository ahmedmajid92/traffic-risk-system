# 🚦 Traffic Risk Prediction System

A **Spatio-Temporal Graph Neural Network (ST-GNN)** for predicting intersection crash risk in Dallas, TX. This project is part of an Intelligent Transportation Systems (ITS) Master's thesis.

---

## 📋 Project Overview

This system uses deep learning to predict traffic accident risk by analyzing:

- **Spatial patterns**: Geographic relationships between intersections
- **Temporal patterns**: Time-based accident trends
- **Environmental factors**: Weather, visibility, and lighting conditions
- **Behavioral indicators**: Driver distraction annotations via NLP

---

## 🏗️ Project Structure

```
traffic-risk-system/
├── data/
│   ├── raw/                    # Original accident data (not tracked)
│   ├── processed/              # Cleaned & annotated data (not tracked)
│   └── external/               # External data sources
├── src/
│   └── data_loader.py          # Phase 1: Data engineering pipeline
├── notebooks/                  # Jupyter notebooks for exploration
├── environment.yml             # Conda environment specification
├── requirements.txt            # Pip dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended: RTX 3080 or better)
- ~4GB disk space for dependencies
- ~500MB for processed Dallas data

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/traffic-risk-system.git
   cd traffic-risk-system
   ```

2. **Create conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate traffic-risk
   ```

3. **Install PyTorch with CUDA** (if not using environment.yml)

   ```bash
   pip install torch==2.5.0+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Download the dataset**

   Download [US Accidents (March 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) from Kaggle and place it in:

   ```
   data/raw/US_Accidents_March23.csv
   ```

---

## 📊 Phase 1: Data Engineering & NLP Annotation

### Running the Pipeline

```bash
python src/data_loader.py
```

### What it does:

1. **Chunked Processing**: Reads the ~3GB dataset in 100k-row chunks to manage memory
2. **Geographic Filtering**: Extracts only Dallas, TX records
3. **Data Cleaning**: Drops missing coordinates/timestamps, rounds coordinates
4. **NLP Annotation**: Uses `all-MiniLM-L6-v2` sentence transformer to detect distraction-related crashes via semantic similarity

### Configuration Options

| Argument           | Default                                       | Description                          |
| ------------------ | --------------------------------------------- | ------------------------------------ |
| `--input, -i`      | `data/raw/US_Accidents_March23.csv`           | Input file path                      |
| `--output, -o`     | `data/processed/dallas_crashes_annotated.csv` | Output file path                     |
| `--chunk-size, -c` | `100000`                                      | Rows per processing chunk            |
| `--threshold, -t`  | `0.35`                                        | Similarity threshold for distraction |
| `--no-gpu`         | -                                             | Disable CUDA acceleration            |

### Output Statistics (Dallas, TX)

| Metric                        | Value           |
| ----------------------------- | --------------- |
| Total Records                 | 115,545         |
| Distracted Crashes            | 28,571 (24.73%) |
| Avg Severity (Distracted)     | 2.530           |
| Avg Severity (Non-Distracted) | 2.218           |

---

## 🔧 Technical Details

### NLP Distraction Detection

Instead of regex-based keyword matching, we use **semantic similarity** with anchor phrases:

```python
ANCHOR_PHRASES = [
    "driver texting while driving",
    "using cell phone",
    "driver inattentive",
    "distracted by mobile device",
    ...
]
```

The model encodes accident descriptions and computes cosine similarity against these anchors. If `max(similarity) >= 0.35`, the crash is labeled as distraction-related.

### Memory Management

- Chunked CSV reading (100k rows/chunk)
- Early filtering before processing
- Explicit garbage collection after each chunk
- GPU memory monitoring via `torch.cuda.memory_allocated()`

---

## 📁 Data Dictionary

Output columns in `dallas_crashes_annotated.csv`:

| Column              | Type      | Description                     |
| ------------------- | --------- | ------------------------------- |
| `ID`                | string    | Unique accident identifier      |
| `Severity`          | int (1-4) | Accident severity level         |
| `Start_Time`        | datetime  | Accident timestamp              |
| `Start_Lat`         | float     | Latitude (5 decimal precision)  |
| `Start_Lng`         | float     | Longitude (5 decimal precision) |
| `Description`       | string    | Accident description text       |
| `Weather_Condition` | string    | Weather at time of accident     |
| `Visibility(mi)`    | float     | Visibility in miles             |
| `Traffic_Signal`    | bool      | Near traffic signal             |
| `Junction`          | bool      | At junction/intersection        |
| `Sunrise_Sunset`    | string    | Day/Night indicator             |
| `Is_Distracted`     | int (0/1) | NLP-derived distraction label   |

---

## 📈 Roadmap

- [x] **Phase 1**: Data Engineering & NLP Annotation
- [ ] **Phase 2**: Graph Construction (intersection nodes, spatial edges)
- [ ] **Phase 3**: Temporal Feature Engineering
- [ ] **Phase 4**: ST-GNN Model Development
- [ ] **Phase 5**: Training & Evaluation
- [ ] **Phase 6**: Deployment & Visualization

---

## 📚 References

- **Dataset**: [US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Model**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## 📄 License

This project is for academic research purposes as part of a Master's thesis in Intelligent Transportation Systems.

---

## 👥 Authors

- Ahmed - Lead Developer & Researcher
