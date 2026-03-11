# CarValue: ML Predictive Pricing Models

A professional, end-to-end Machine Learning web application designed to predict the resale valuation of used cars. Utilizing advanced regression algorithms and an elegant, minimalist user interface, this tool accurately estimates vehicle depreciation based on critical market anchors and physical specifications.

## 🚀 Features

- **Algorithm Comparison Matrix**: Integrates `RandomForest`, `GradientBoosting`, and `XGBoost` regression pipelines to systematically benchmark the highest performing model via `RandomizedSearchCV` hyperparameter tuning.
- **Data-Driven Architecture**: Achieves up to ~93% validation R² scores through rigorous 5-Fold Cross Validation.
- **Premium User Interface**: Built with an elegant, monochromatic frontend (via Streamlit) optimized for an intuitive user experience.
- **Market Anchor Logic**: Emphasizes actual ex-showroom listing prices to accurately compute value retention and depreciation metrics.

## ⚙️ Project Structure

```bash
.
├── app/
│   └── streamlit_app.py        # Streamlit frontend UI
├── archive/
│   └── car data.csv            # Original training dataset
├── models/
│   ├── car_price_model.pkl     # Serialized artifact of the winning ML model
│   └── evaluation_plots.png    # Matplotlib visualizations of RMSE/MAE variance
├── src/
│   ├── preprocess.py           # Pipeline logic for cleaning & feature-engineering
│   ├── train_model.py          # Benchmark algorithm script (RF, GB, XGB)
│   └── predict.py              # Single-point inference API logic
├── requirements.txt            # Python dependencies
└── README.md
```

## 🛠️ Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/carvalue.git
   cd carvalue
   ```

2. **Create a virtual environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # source venv/bin/activate    # On Mac/Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the ML model locally:**

   ```bash
   python src/train_model.py
   ```

5. **Launch the web application:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 📊 Technical Flow

- **Data Preprocessing Phase**: Drops legacy string columns and extracts raw engineering parameters. Implements One-Hot encoding for categorical data (Fuel Type, Seller Type, Transmission) using `pandas.get_dummies(drop_first=True)` to avoid multicollinearity.
- **Benchmark & Hyperparameter Tuning Phase**: Operates exhaustive randomized grid searches to optimize algorithm hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `min_samples_leaf`). Saves the leading mathematical model + features into an artifact.
- **Inference Phase**: Loads the `.pkl` artifact into the lightweight Streamlit service. Reconstructs identical zero-padded input DataFrames at runtime dynamically to return sub-second price inferences securely to the end-user.


