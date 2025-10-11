# Handwritten Digit Classifier (NumPy + Pandas + Matplotlib)

A minimal, from-scratch (no deep‑learning framework) **multilayer perceptron** for classifying handwritten digits (MNIST‑style CSVs).  
It trains on `train.csv`, evaluates on a dev split, and lets you interactively test indices from `test.csv`.

---

## 📦 Contents

- `main.py` (your script)
- `train.csv` — training data (first column = label, next 784 columns = pixel values 0–255)
- `test.csv` — test data (784 pixel columns, **no labels**)
- (generated) `README.md` — this file

> The code expects Kaggle‑style MNIST CSVs (28×28 grayscale, row‑major).

---

## 🧠 Model Overview

- **Architecture:** `784 → 64 → 32 → 10`
- **Activations:** ReLU (hidden layers), Softmax (output)
- **Loss:** cross‑entropy via softmax with one‑hot targets (implicit in gradient derivation)
- **Optimizer:** SGD with mini‑batches (shuffle each epoch)
- **Regularization:** L2 (`lambd`)
- **Normalization:** inputs scaled to `[0, 1]` by dividing by 255
- **Dev split:** first 1000 rows of shuffled `train.csv` used as `data_dev`

---

## 🗂️ Data Format

**`train.csv` (N rows × 785 cols):**
- Column `0`: integer label `0..9`
- Columns `1..784`: pixel intensities `0..255` (flattened 28×28)

**`test.csv` (M rows × 784 cols):**
- Columns `0..783`: pixel intensities `0..255` (no labels)

---

## ⚙️ Key Script Entry Points

- **Data loading / preprocessing**
  - `pd.read_csv('train.csv')`, `pd.read_csv('test.csv')`
  - `X = pixels.T / 255.0` → shapes `(784, m)`
  - `Y = labels` → shape `(m,)`

- **Leakage check**
  - Builds two DataFrames (`X_train.T` and `X_test.T`) and performs an **inner merge** to detect rows that are identical across splits.

- **Model functions**
  - `init_params()` — Kaiming‑style init for ReLU layers
  - `forward_prop(...)` — Z/A for each layer; softmax at output
  - `backward_prop(...)` — analytical gradients + L2
  - `update_params(...)` — SGD step
  - `gradient_descent(...)` — shuffling, mini‑batches, dev accuracy every 10 iters

- **Inference**
  - `make_predictions(X, ...)` → `argmax` over softmax
  - `test_prediction(index, ...)` → show `matplotlib` image (28×28) and predicted class

---

## 🔧 Tunable Hyperparameters

Change these in the `__main__` call (or expose argparse):
```python
iterations=10       # training epochs
alpha=0.1           # learning rate
batch_size=64       # mini-batch size
lambd=0.01          # L2 regularization strength
```

---

## 🧪 Expected Shapes

- `X_train`: `(784, m_train)`
- `Y_train`: `(m_train,)`
- `X_dev`: `(784, m_dev)`
- `Y_dev`: `(m_dev,)`
- Hidden activations: `(64, m)`, `(32, m)`
- Output activations: `(10, m)`

---

## 🩺 Troubleshooting

- **`FileNotFoundError: 'train.csv' / 'test.csv'`**  
  Ensure both CSVs are in the working directory. If using an IDE, check the run configuration’s working dir.

- **Unicode / console issues on Windows**  
  The script sets:
  ```python
  sys.stdout.reconfigure(encoding='utf-8')
  sys.stdin.reconfigure(encoding='utf-8')
  ```
  If Python < 3.7 or reconfigure is unsupported, remove those two lines.

- **Very low accuracy**  
  - Verify that `train.csv` is the expected MNIST format (labels in the first column).  
  - Increase `iterations` to 50–100.  
  - Try a lower `alpha` (e.g., `0.05`), or raise `batch_size` to 128.  
  - Ensure scaling (`/255.0`) is applied.

- **Data leakage warning**  
  If the console prints overlap rows between train/dev: your `test.csv` duplicates some training samples. Replace the test set or remove overlaps.

- **Matplotlib on headless servers**  
  Use a non-interactive backend:  
  ```python
  import matplotlib
  matplotlib.use('Agg')
  ```

---

## 📈 Notes & Extensions

- Add **argparse** for CLI: number of iterations, LR, batch size, regularization, paths.  
- Log losses/accuracy per epoch to CSV for plotting learning curves.  
- Replace manual loops with **vectorized** ops where possible for speed.  
- Try different widths: `128 → 64 → 10`, add dropout (manual), or switch to Adam.  
- Save/load learned weights to `.npz` with `np.savez` / `np.load`.

---

## ▶️ How to Run

### 1) Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```text
numpy
pandas
matplotlib
```

### 3) Place data files
Put `train.csv` and `test.csv` in the project root (same folder as the script).

### 4) Train + interactive test
```bash
python main.py
```
The script will:
1. Shuffle and split `train.csv` into `data_train` / `data_dev` (1000 rows for dev).
2. Train for the specified number of iterations/epochs.
3. Every 10 iterations, print dev accuracy.
4. Enter an **interactive loop**: type an index from `test.csv` to visualize the image and print the predicted digit.

To exit, enter `-1` when prompted.

---

## 🧾 License

MIT — do what you want, no warranty.
