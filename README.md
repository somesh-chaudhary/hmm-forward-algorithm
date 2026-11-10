# Observation-Sequence Likelihood Calculation Using Hidden Markov Model (HMM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-green.svg)](https://numpy.org/)
[![Topic](https://img.shields.io/badge/Module%205-HMM%20Forward%20Algorithm-orange.svg)](#)

A complete, from-scratch implementation in Python and NumPy of a **Hidden Markov Model (HMM)** designed to compute the observation-sequence likelihood using the **Forward Algorithm** and verify it with the **Backward Algorithm**.

---

## 👥 Project Information & Team Members

- **Course Syllabus**: Module 5 — HMM Forward Algorithm
- **Topic**: Observation-Sequence Likelihood Calculation Using Hidden Markov Model (HMM)
- **Team**:
  1. `1AT23AI050` — **SHIVASHANKAR**
  2. `1AT23AI051` — **SOMOIPTA NANDY**
  3. `1AT23AI052` — **SOMESH CHAUDHARY**

---

## 📂 Project Structure

```text
.
├── demo.py                      # Standalone classroom demonstration script
├── hmm.py                       # Core HiddenMarkovModel class (Forward, Backward, Viterbi, Baum-Welch)
├── Hidden Markov Model.ipynb   # Interactive notebook with step-by-step outputs
└── README.md                    # Project documentation
```

---

## 📖 Theoretical Overview

### 1. The Likelihood Problem
Given a Hidden Markov Model $\lambda = (A, B, \pi)$ and an observation sequence $O = (O_0, O_1, \dots, O_{T-1})$, determine the likelihood $P(O \mid \lambda)$.

### 2. Forward Algorithm (Dynamic Programming)
Instead of enumerating all $N^T$ paths (exponential complexity), the **Forward Algorithm** computes cumulative probabilities in $O(N^2 T)$ time:

- **Initialization ($t = 0$):**
  $$\alpha_0(i) = \pi_i \cdot b_i(O_0)$$

- **Induction / Recursion ($t = 1 \dots T-1$):**
  $$\alpha_t(j) = \left[ \sum_{i=1}^N \alpha_{t-1}(i) \cdot a_{ij} \right] \cdot b_j(O_t)$$

- **Termination ($t = T-1$):**
  $$P(O \mid \lambda) = \sum_{i=1}^N \alpha_{T-1}(i)$$

---

## 🍦 The Weather & Ice-Cream Example

- **Hidden States ($N=2$):** `["hot", "cold"]` (outdoor weather)
- **Observations ($M=3$):** `[1, 2, 3]` (number of ice creams eaten daily)
- **Observation Sequence ($T=7$):** `[1, 2, 3, 2, 2, 1, 2]`

### Model Parameters:
- **Initial Probabilities ($\pi$):** Hot = 0.8, Cold = 0.2
- **Transition Matrix ($A$):**
  $$\begin{pmatrix} P(\text{hot} \to \text{hot}) & P(\text{hot} \to \text{cold}) \\ P(\text{cold} \to \text{hot}) & P(\text{cold} \to \text{cold}) \end{pmatrix} = \begin{pmatrix} 0.6 & 0.4 \\ 0.5 & 0.5 \end{pmatrix}$$
- **Emission Matrix ($B$):**
  $$\begin{pmatrix} P(1 \mid \text{hot}) & P(2 \mid \text{hot}) & P(3 \mid \text{hot}) \\ P(1 \mid \text{cold}) & P(2 \mid \text{cold}) & P(3 \mid \text{cold}) \end{pmatrix} = \begin{pmatrix} 0.2 & 0.4 & 0.4 \\ 0.5 & 0.4 & 0.1 \end{pmatrix}$$

---

## 📊 Forward Probability ($\alpha$) Table

| State | $t=0$ ($O_0=1$) | $t=1$ ($O_1=2$) | $t=2$ ($O_2=3$) | $t=3$ ($O_3=2$) | $t=4$ ($O_4=2$) | $t=5$ ($O_5=1$) | $t=6$ ($O_6=2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hot`** | 0.16000000 | 0.05840000 | 0.02313600 | 0.00647584 | 0.00247919 | 0.00049362 | 0.00031569 |
| **`cold`** | 0.10000000 | 0.04560000 | 0.00461600 | 0.00462496 | 0.00196113 | 0.00098612 | 0.00027620 |
| **Sum** | **0.26000000** | **0.10400000** | **0.02775200** | **0.01110080** | **0.00444032** | **0.00147974** | **0.00059189** |

---

## 🔍 Verification (Forward vs. Backward Algorithm)

| Metric | Forward Algorithm | Backward Algorithm | Absolute Difference | Status |
| :--- | :---: | :---: | :---: | :---: |
| **$P(O \mid \lambda)$** | **0.0005918945** | **0.0005918945** | **0.00e+00** | ✅ **PASSED (Exact Match)** |

$$\text{Final Likelihood: } P(O \mid \lambda) \approx 5.92 \times 10^{-4}$$

---

## 🚀 How to Run

### 1. Terminal Demonstration
Clone the repository and run:
```bash
python demo.py
```
*(Or run `python hmm.py`)*

### 2. Interactive Notebook
Open `Hidden Markov Model.ipynb` in VS Code or JupyterLab and execute all cells.

---

## 📚 References
- Daniel Jurafsky & James H. Martin, *Speech and Language Processing*, Chapter: Hidden Markov Models.
- Lawrence R. Rabiner, *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition* (IEEE Proceedings).
