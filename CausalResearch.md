# Causal Inference and Machine Learning for Treatment Effect Estimation

---

## **Introduction**
This document summarizes key concepts, formulas, and methodologies in causal inference and machine learning for estimating treatment effects. It includes your notes, additional important formulas, and explanations to ensure a comprehensive understanding of the topic.

---

## **Key Concepts and Formulas**

### 1. **Potential Outcomes Framework**
   - **Potential Outcomes**: For a binary treatment \( A \), the potential outcomes are:
     \[
     Y^1 = \text{Outcome if treated}, \quad Y^0 = \text{Outcome if not treated}
     \]
   - **Individual Causal Effect**:
     \[
     \tau_i = Y^1_i - Y^0_i
     \]
   - **Average Causal Effect (ACE)**:
     \[
     \tau = E(Y^1 - Y^0)
     \]
   - **Fundamental Problem of Causal Inference**: We can only observe one potential outcome for each individual:
     \[
     Y_i = A_i Y^1_i + (1 - A_i) Y^0_i
     \]

---

### 2. **Assumptions for Causal Inference**
   - **Stable Unit Treatment Value Assumption (SUTVA)**:
     - No interference: \( Y_i \) does not depend on \( A_j \) for \( j \neq i \).
     - Single version of treatment.
   - **Ignorability (Unconfoundedness)**:
     \[
     (Y^1, Y^0) \perp A \mid X
     \]
   - **Positivity**:
     \[
     0 < P(A = 1 \mid X) < 1 \quad \text{for all } X
     \]

---

### 3. **Matching Methods**
   - **Mahalanobis Distance**:
     \[
     D(X_i, X_j) = \sqrt{(X_i - X_j)^T S^{-1} (X_i - X_j)}
     \]
     where \( S \) is the covariance matrix of \( X \).
   - **Propensity Score Matching**:
     \[
     e(X) = P(A = 1 \mid X)
     \]
   - **Caliper Matching**: Matches are only made if:
     \[
     |e(X_i) - e(X_j)| < \epsilon
     \]
   - **Outcome Analysis After Matching**:
     \[
     \hat{\tau} = \frac{1}{N} \sum_{i=1}^N (Y_{i, \text{treated}} - Y_{i, \text{control}})
     \]

---

### 4. **Propensity Scores and IPW**
   - **Propensity Score**:
     \[
     e(X) = P(A = 1 \mid X)
     \]
   - **Inverse Probability Weighting (IPW)**:
     \[
     \hat{\tau}_{\text{IPW}} = \frac{1}{N} \sum_{i=1}^N \left( \frac{A_i Y_i}{e(X_i)} - \frac{(1 - A_i) Y_i}{1 - e(X_i)} \right)
     \]
   - **Doubly Robust Estimator (AIPW)**:
     \[
     \hat{\tau}_{\text{AIPW}} = \frac{1}{N} \sum_{i=1}^N \left( \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{A_i (Y_i - \hat{\mu}_1(X_i))}{e(X_i)} - \frac{(1 - A_i) (Y_i - \hat{\mu}_0(X_i))}{1 - e(X_i)} \right)
     \]

---

### 5. **Instrumental Variables (IV)**
   - **Instrumental Variable (IV)**: A variable \( Z \) that satisfies:
     - **Relevance**: \( Z \) is correlated with \( A \).
     - **Exclusion Restriction**: \( Z \) affects \( Y \) only through \( A \).
   - **Two-Stage Least Squares (2SLS)**:
     - **First Stage**: Regress \( A \) on \( Z \):
       \[
       A_i = \alpha_0 + \alpha_1 Z_i + \epsilon_i
       \]
     - **Second Stage**: Regress \( Y \) on the predicted \( \hat{A} \):
       \[
       Y_i = \beta_0 + \beta_1 \hat{A}_i + \eta_i
       \]
   - **Complier Average Causal Effect (CACE)**:
     \[
     \text{CACE} = \frac{E(Y \mid Z = 1) - E(Y \mid Z = 0)}{E(A \mid Z = 1) - E(A \mid Z = 0)}
     \]

---

### 6. **Machine Learning for Causal Inference**
   - **Causal Trees and Forests**:
     - Partition the covariate space to estimate heterogeneous treatment effects.
   - **T-Learner**:
     - Fit separate models for treated and control groups:
       \[
       \hat{\tau}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)
       \]
   - **S-Learner**:
     - Fit a single model including treatment as a covariate:
       \[
       \hat{\tau}(X) = \hat{\mu}(X, 1) - \hat{\mu}(X, 0)
       \]
   - **AIPW with Machine Learning**:
     - Use ML models to estimate \( \hat{\mu}_1(X) \), \( \hat{\mu}_0(X) \), and \( e(X) \).

---

### 7. **Policy Learning**
   - **Policy Value Function**:
     \[
     V(\pi) = E[Y(\pi(X))]
     \]
   - **IPW Estimator for Policy Value**:
     \[
     \hat{V}(\pi) = \frac{1}{N} \sum_{i=1}^N \frac{1(W_i = \pi(X_i))}{P(W_i = \pi(X_i) \mid X_i)} Y_i
     \]
   - **AIPW Estimator for Policy Value**:
     \[
     \hat{V}(\pi) = \frac{1}{N} \sum_{i=1}^N \left( \hat{\mu}_{\pi(X_i)}(X_i) + \frac{1(W_i = \pi(X_i))}{P(W_i = \pi(X_i) \mid X_i)} (Y_i - \hat{\mu}_{\pi(X_i)}(X_i)) \right)
     \]

---

## **Additional Important Data**

### 1. **Sensitivity Analysis**
   - Assess the robustness of results to unmeasured confounding:
     \[
     \Gamma = \frac{P(A = 1 \mid X, U)}{P(A = 0 \mid X, U)} \cdot \frac{P(A = 0 \mid X)}{P(A = 1 \mid X)}
     \]
     where \( U \) is an unmeasured confounder.

### 2. **High-Dimensional Data**
   - **LASSO Regression**:
     \[
     \min_\beta \frac{1}{N} \sum_{i=1}^N (Y_i - X_i \beta)^2 + \lambda \sum_{j=1}^p |\beta_j|
     \]
   - **Ridge Regression**:
     \[
     \min_\beta \frac{1}{N} \sum_{i=1}^N (Y_i - X_i \beta)^2 + \lambda \sum_{j=1}^p \beta_j^2
     \]

---

## **Conclusion**
This document provides a comprehensive overview of causal inference methods, including matching, propensity scores, instrumental variables, and machine learning techniques. By combining these methods, researchers can estimate treatment effects more robustly and interpretably.

---

## **Recommendations**
1. **Use Doubly Robust Estimators**: Combine regression adjustment with IPW for robustness.
2. **Check for Overlap**: Ensure positivity by examining propensity score distributions.
3. **Conduct Sensitivity Analysis**: Assess the impact of unmeasured confounding.
4. **Leverage Machine Learning**: Use ML methods like causal forests and policy learning for heterogeneous treatment effects and optimal treatment assignment.

---

## **References**
- Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies.
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies.
- Imbens, G. W., & Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences.
- Athey, S., & Imbens, G. W. (2019). Machine Learning Methods Economists Should Know About.

---

**Download this file as `causal_inference_report.md` for your reference.**
