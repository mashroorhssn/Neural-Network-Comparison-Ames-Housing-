# Abstract

This project serves as an advanced extension of the "Ames Housing Price Prediction" assignment. The primary objective was to compare the performance of the established ensemble model (**Random Forest**) against various **Neural Network architectures** to determine if Deep Learning techniques could offer superior predictive power for this regression task.

While Neural Networks are state-of-the-art for unstructured data (images, text), this study aimed to test their efficacy on structured, tabular housing data. The analysis involved optimizing the baseline Random Forest via hyperparameter tuning and subsequently designing, training, and evaluating four distinct Neural Network architectures (Simple, Deep, Regularized, and Robust).

**Key Finding:** The analysis conclusively demonstrated that the **Optimized Random Forest** significantly outperformed all Neural Network variants. While advanced techniques like Batch Normalization stabilized the Neural Networks, the ensemble method remained superior for this dataset size and structure.

---

# 1. Project Scope & Goal

**Business Problem:**
Accurately predicting real estate values is critical for buyers, sellers, and investors. The challenge is to build a model that minimizes error (RMSE) while maximizing the variance explained ($R^2$).

**Goal:**
To rigorously evaluate whether increasing model complexity (via Deep Learning) translates to better performance on the Ames Housing dataset, or if traditional Machine Learning methods remain the most practical choice.

* **Dataset:** Ames Housing (1460 samples, 81 features).
* **Task:** Regression (Predicting `SalePrice`).
* **Metric:** Root Mean Squared Error (RMSE) and $R^2$ Score.

---

# 2. Methodology

The project followed a strict experimental pipeline to ensure fair comparison.

### A. Data Preprocessing
To accommodate Neural Networks, the preprocessing pipeline was upgraded from the initial submission:
* **Target Normalization:** The `SalePrice` was log-transformed (`np.log1p`) to correct skewness.
* **Feature Scaling:** Unlike tree-based models, Neural Networks require scaled inputs. `StandardScaler` was applied to all numerical features.
* **Encoding:** Categorical variables were transformed using `OneHotEncoder`.
* **Final Input:** The processed dataset contained **269 features** after encoding.

### B. Model Architectures Evaluated

**1. Baseline: Optimized Random Forest**
* **Strategy:** We moved beyond default parameters by performing `RandomizedSearchCV`.
* **Optimization:** We tuned `n_estimators`, `max_depth`, and `min_samples_split`.
* **Result:** The search confirmed that a robust configuration (300 trees) provided optimal stability ($R^2 \approx 0.90$).

**2. Neural Networks (TensorFlow/Keras)**
We tested four progressively complex architectures to investigate performance dynamics:
* **Model 1 (Simple NN):** A shallow network (1 hidden layer) to establish a deep learning baseline.
* **Model 2 (Deep NN):** A 3-layer network designed to capture hierarchical feature interactions.
* **Model 3 (Regularized NN):** Introduced `Dropout` and `EarlyStopping` to combat the overfitting observed in Model 2.
* **Model 4 (Robust NN):** Addressed gradient instability by implementing **Batch Normalization** between layers.

---

# 3. Results & Comparative Analysis

The table below summarizes the performance of all models on the unseen Test Set.

| Model | Architecture Type | Test R² Score | Test RMSE ($) | Status |
|:---|:---|---:|---:|:---|
| **Optimized Random Forest** | **Ensemble** | **0.9000** | **$30,646** | **Best** |
| Robust NN | Deep Learning (Batch Norm) | 0.7239 | $55,457 | Good |
| Simple NN | Shallow Network | 0.6698 | $127,143 | Underfit |
| Deep NN | Deep Network | 0.2383 | $945,035 | Unstable |
| Regularized NN | Deep + Dropout | < 0.00 | (Failed) | Failed |

### Detailed Observations

**1. The "Tabular Data" Reality**
The **Optimized Random Forest** achieved an $R^2$ of **0.90**, dominating the leaderboard. This reinforces the machine learning consensus: for structured tabular data with limited sample size (<10,000 rows), ensemble tree methods often outperform deep neural networks. They are better at handling discrete features and require less data to generalize.

**2. The Instability of Depth**
Moving from "Simple" to "Deep" architectures initially *hurt* performance (R² dropped from 0.67 to 0.24). This counter-intuitive result highlights the difficulty of training deep networks on small datasets—the model struggled to find a stable minimum and likely suffered from exploding gradients or severe overfitting.

**3. The Power of Batch Normalization**
Our "Robust NN" (Model 4) was the most successful neural network. By applying **Batch Normalization**, we stabilized the learning process, allowing the deep network to actually converge. This improved the score from near-zero to **0.72**. While it didn't beat the Random Forest, it proved that architectural choices (normalization) are critical for deep learning success.

---

# 4. Final Conclusion & Takeaway

> **"Complexity does not guarantee Performance."**

This project demonstrated that while Neural Networks are powerful, they are not a "silver bullet" for every problem. For the Ames Housing dataset:
1.  **Random Forest is the superior choice**, offering higher accuracy ($30k error vs $55k error) and greater stability with less tuning.
2.  **Deep Learning requires careful architecture:** Simply adding layers failed. Success was only possible when we specifically addressed stability using Batch Normalization.

This analysis compares models and explores the "black box" nature of hyperparameter tuning and neural architecture design.
