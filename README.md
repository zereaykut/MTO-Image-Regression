# MTO-Image-Regression
Image Regression Models on Meteorological Data

## 📂 Directory Structure

```text
MTO-Image-Regression/
├── app.py
├── README.md
└── src
    ├── __init__.py
    ├── models
    │   ├── conv.py
    │   ├── conv_transformer.py
    │   └── __init__.py
    └── utils.py
```

## 🚀 Getting Started
Prerequisites

Ensure you have Python 3.8+ installed. You will need the following libraries:
* tensorflow (2.x)
* numpy
* pandas
* scikit-learn
* xarray (if handling raw NetCDF/Grib in prep stages)

Installation

Clone the repository:
```Bash
git clone https://github.com/your-username/meteo-power-prediction.git
cd meteo-power-prediction
```
Install dependencies:
```Bash
pip install -r requirements.txt
```

## 📊 Data Input (Both Models)

Models receive the exact same input tensor from your DataLoader.

* **Raw Source:** Individual .npz files (e.g., u1000hPa, t950hPa).
* **Stacking:** These are stacked together to create "Channels".
* **Final Input Shape:** (Batch_Size, Height, Width, Channels)
    * **Example:** (32, 73, 145, 14) means a batch of 32 samples, a 73x145 weather grid, and 14 different weather parameters per grid point.


## 🧠 Model Architectures

### ConvModel (Standard CNN)

This model treats the data as an image, looking for local spatial patterns (like pressure gradients or wind fronts).
* **Step 1:** Convolution: Conv2D slides filters over the Height/Width.
    * Transformation: (H, W, C) → (H, W, Filters)
* **Step 2:** Pooling: MaxPooling2D shrinks the grid size to summarize features.
    * Transformation: (H, W, F) → (H/2, W/2, F)
* **Step 3:** Flatten: Collapses the remaining grid into a single 1D vector.
    * Transformation: (H', W', F) → (Features_Vector)
* **Step 4:** Output: A Dense layer maps the vector to a single number (MW).

**Core Logic:** "Analyze neighboring pixels to find local shapes."

### ConvTransformer (Hybrid)

This model treats the grid as a sequence of patches, allowing it to learn relationships between distant parts of the map (Global Context).
* **Step 1:** Feature Extraction (CNN): Uses Conv2D initially to reduce noise and grid size.
    * Shape: (H, W, C) → (H', W', F)
* **Step 2:** Reshape (The Critical Step): The grid is flattened into a sequence.
    * Logic: Every grid cell (or patch) becomes a "token" (like a word in a sentence).
    * Transformation: (H', W', F) → (Sequence_Length, Embedding_Dim)
    * Where: Sequence_Length = H' * W'
* **Step 3:** Attention: The Transformer mechanism looks at how every patch relates to every other patch, regardless of distance.
* **Step 4:** Aggregation: The sequence is averaged (Global Average Pooling) or a specific token is used to predict the final output.

**Core Logic:** "Turn the map into a sentence of features, then see how every part of the map influences every other part."