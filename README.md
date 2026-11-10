# Observation-Sequence Likelihood Calculation Using Hidden Markov Model (HMM)

A complete, from-scratch implementation in Python and NumPy of a **Hidden Markov Model (HMM)** to compute observation-sequence likelihood using the **Forward Algorithm** and independently verify it using the **Backward Algorithm**.

---



## 🎯 Project Overview

In sequential and time-series data analysis, standard probabilistic models often fail because real-world observations depend on unobserved underlying factors. A **Hidden Markov Model (HMM)** resolves this by modeling an unobservable (hidden) state sequence that emits observable tokens according to statistical distributions.

This project addresses **HMM Problem 1: Evaluation / Likelihood**:
> Given a model $\lambda = (A, B, \pi)$ and an observation sequence $O = (O_0, O_1, \dots, O_{T-1})$, calculate the total probability that the sequence was produced by the model: $P(O \mid \lambda)$.

---

## ⚡ The Problem: Observation Likelihood

### Why Brute Force Calculation Fails
For an observation sequence of length $T$ and a model with $N$ hidden states:
1. To calculate $P(O \mid \lambda)$ naively, every possible hidden state sequence $Q = (q_0, q_1, \dots, q_{T-1})$ must be evaluated.
2. For each sequence $Q$, the joint probability is:
   $$P(O, Q \mid \lambda) = \pi_{q_0} b_{q_0}(O_0) \prod_{t=1}^{T-1} a_{q_{t-1}, q_t} b_{q_t}(O_t)$$
3. The total likelihood is the sum over all possible state paths:
   $$P(O \mid \lambda) = \sum_{\text{all } Q} P(O, Q \mid \lambda)$$
4. **Computational Complexity**: There are $N^T$ possible state sequences.
   - For $N=2$ and $T=7$: $2^7 = 128$ paths.
   - For $N=10$ and $T=50$: $10^{50}$ paths (computationally impossible).

### Dynamic Programming Solution
The **Forward Algorithm** uses dynamic programming to collapse identical state histories into a cumulative probability matrix called the **Alpha ($\alpha$) table**. This reduces time complexity from exponential $O(N^T)$ to polynomial **$O(N^2 T)$**.

---

## 📐 HMM Formal Mathematical Definition

An HMM is completely specified by the parameter triplet $\lambda = (A, B, \pi)$:

1. **Hidden States ($S$):**
   $$S = \{s_1, s_2, \dots, s_N\}$$
   A discrete set of $N$ underlying hidden states.

2. **Observations ($V$):**
   $$V = \{v_1, v_2, \dots, v_M\}$$
   A discrete vocabulary of $M$ observable symbols.

3. **Initial State Distribution ($\pi$):**
   $$\pi_i = P(q_0 = s_i), \quad \sum_{i=1}^N \pi_i = 1$$

4. **State Transition Probability Matrix ($A$):**
   $$a_{ij} = P(q_{t+1} = s_j \mid q_t = s_i), \quad \sum_{j=1}^N a_{ij} = 1$$

5. **Emission Probability Matrix ($B$):**
   $$b_j(k) = P(O_t = v_k \mid q_t = s_j), \quad \sum_{k=1}^M b_j(k) = 1$$

### Fundamental Markov Assumptions
- **First-Order Markov Property:** State transitions depend solely on the immediate predecessor:
  $$P(q_{t+1} \mid q_t, q_{t-1}, \dots, q_0) = P(q_{t+1} \mid q_t)$$
- **Output Independence:** The emission at time $t$ depends solely on the current state $q_t$:
  $$P(O_t \mid q_t, q_{t-1}, \dots, O_{t-1}, \dots) = P(O_t \mid q_t)$$

---

## 🔄 The Forward Algorithm

The forward variable $\alpha_t(i)$ represents the joint probability of observing the partial sequence up to time $t$ and being in state $s_i$ at time $t$:
$$\alpha_t(i) = P(O_0, O_1, \dots, O_t, q_t = s_i \mid \lambda)$$

The algorithm computes $\alpha_t(i)$ in three steps:

### Step 1: Initialization ($t = 0$)
Multiply the initial state probability by the emission probability of the first observation $O_0$:
$$\alpha_0(i) = \pi_i \cdot b_i(O_0), \quad 1 \le i \le N$$

### Step 2: Induction / Recursion ($t = 1, 2, \dots, T-1$)
Sum the probabilities of reaching state $s_j$ from all possible previous states $s_i$, then multiply by the emission probability of $O_t$:
$$\alpha_t(j) = \left[ \sum_{i=1}^N \alpha_{t-1}(i) \cdot a_{ij} \right] \cdot b_j(O_t), \quad 1 \le j \le N$$

*Vectorized Form (NumPy):*
$$\boldsymbol{\alpha}_t = (\boldsymbol{\alpha}_{t-1} \cdot A) \odot B_{*, O_t}$$

### Step 3: Termination ($t = T-1$)
Sum all forward probabilities at the final time step across all states:
$$P(O \mid \lambda) = \sum_{i=1}^N \alpha_{T-1}(i)$$

---

## 🔁 The Backward Algorithm (Verification)

To verify the Forward Algorithm, the **Backward Algorithm** calculates the same observation likelihood by working in reverse from $t = T-1$ down to $t = 0$.

### Definition of Backward Variable $\beta_t(i)$
$$\beta_t(i) = P(O_{t+1}, O_{t+2}, \dots, O_{T-1} \mid q_t = s_i, \lambda)$$

### Steps:
1. **Initialization ($t = T-1$):**
   $$\beta_{T-1}(i) = 1.0, \quad 1 \le i \le N$$
2. **Recursion ($t = T-2, T-3, \dots, 0$):**
   $$\beta_t(i) = \sum_{j=1}^N a_{ij} \cdot b_j(O_{t+1}) \cdot \beta_{t+1}(j), \quad 1 \le i \le N$$
3. **Termination ($t = 0$):**
   $$P(O \mid \lambda) = \sum_{i=1}^N \pi_i \cdot b_i(O_0) \cdot \beta_0(i)$$

Because both algorithms evaluate the total probability over the same probability space, they yield identical results:
$$P(O \mid \lambda)_{\text{forward}} = P(O \mid \lambda)_{\text{backward}}$$

---

## 🍦 Step-by-Step Example: Weather & Ice Cream

### Model Configuration
- **Hidden States ($N=2$):** `["hot", "cold"]` (outdoor weather)
- **Observation Space ($M=3$):** `[1, 2, 3]` (number of ice creams eaten per day)
- **Observation Sequence ($T=7$):** `[1, 2, 3, 2, 2, 1, 2]`

### Model Parameters:
- **Initial Probabilities ($\pi$):**
  $$\pi = [\pi_{\text{hot}}, \pi_{\text{cold}}] = [0.8, 0.2]$$

- **Transition Probability Matrix ($A$):**
  $$A = \begin{pmatrix} P(\text{hot} \to \text{hot}) & P(\text{hot} \to \text{cold}) \\ P(\text{cold} \to \text{hot}) & P(\text{cold} \to \text{cold}) \end{pmatrix} = \begin{pmatrix} 0.6 & 0.4 \\ 0.5 & 0.5 \end{pmatrix}$$

- **Emission Probability Matrix ($B$):**
  $$B = \begin{pmatrix} P(1 \mid \text{hot}) & P(2 \mid \text{hot}) & P(3 \mid \text{hot}) \\ P(1 \mid \text{cold}) & P(2 \mid \text{cold}) & P(3 \mid \text{cold}) \end{pmatrix} = \begin{pmatrix} 0.2 & 0.4 & 0.4 \\ 0.5 & 0.4 & 0.1 \end{pmatrix}$$

---

### Step-by-Step Arithmetic Trace

- **$t = 0$ (Observation = 1):**
  - $\alpha_0(\text{hot}) = 0.8 \times 0.2 = \mathbf{0.16000000}$
  - $\alpha_0(\text{cold}) = 0.2 \times 0.5 = \mathbf{0.10000000}$

- **$t = 1$ (Observation = 2):**
  - $\alpha_1(\text{hot}) = (0.16 \times 0.6 + 0.10 \times 0.5) \times 0.4 = 0.146 \times 0.4 = \mathbf{0.05840000}$
  - $\alpha_1(\text{cold}) = (0.16 \times 0.4 + 0.10 \times 0.5) \times 0.4 = 0.114 \times 0.4 = \mathbf{0.04560000}$

- **$t = 2$ (Observation = 3):**
  - $\alpha_2(\text{hot}) = (0.0584 \times 0.6 + 0.0456 \times 0.5) \times 0.4 = 0.05784 \times 0.4 = \mathbf{0.02313600}$
  - $\alpha_2(\text{cold}) = (0.0584 \times 0.4 + 0.0456 \times 0.5) \times 0.1 = 0.04616 \times 0.1 = \mathbf{0.00461600}$

- **$t = 3$ (Observation = 2):**
  - $\alpha_3(\text{hot}) = (0.023136 \times 0.6 + 0.004616 \times 0.5) \times 0.4 = 0.0161896 \times 0.4 = \mathbf{0.00647584}$
  - $\alpha_3(\text{cold}) = (0.023136 \times 0.4 + 0.004616 \times 0.5) \times 0.4 = 0.0115624 \times 0.4 = \mathbf{0.00462496}$

- **$t = 4$ (Observation = 2):**
  - $\alpha_4(\text{hot}) = (0.00647584 \times 0.6 + 0.00462496 \times 0.5) \times 0.4 = \mathbf{0.00247919}$
  - $\alpha_4(\text{cold}) = (0.00647584 \times 0.4 + 0.00462496 \times 0.5) \times 0.4 = \mathbf{0.00196113}$

- **$t = 5$ (Observation = 1):**
  - $\alpha_5(\text{hot}) = (0.00247919 \times 0.6 + 0.00196113 \times 0.5) \times 0.2 = \mathbf{0.00049362}$
  - $\alpha_5(\text{cold}) = (0.00247919 \times 0.4 + 0.00196113 \times 0.5) \times 0.5 = \mathbf{0.00098612}$
  *(Cold probability overtakes Hot probability due to low consumption emission evidence)*

- **$t = 6$ (Observation = 2):**
  - $\alpha_6(\text{hot}) = (0.00049362 \times 0.6 + 0.00098612 \times 0.5) \times 0.4 = \mathbf{0.00031569}$
  - $\alpha_6(\text{cold}) = (0.00049362 \times 0.4 + 0.00098612 \times 0.5) \times 0.4 = \mathbf{0.00027620}$

- **Final Termination ($t = 6$):**
  $$P(O \mid \lambda) = \alpha_6(\text{hot}) + \alpha_6(\text{cold}) = 0.00031569 + 0.00027620 = \mathbf{0.0005918945}$$
  $$P(O \mid \lambda) \approx \mathbf{5.92 \times 10^{-4}}$$

---

## 📊 The Forward Probability (Alpha) Table

| Hidden State | $t=0$ ($O_0=1$) | $t=1$ ($O_1=2$) | $t=2$ ($O_2=3$) | $t=3$ ($O_3=2$) | $t=4$ ($O_4=2$) | $t=5$ ($O_5=1$) | $t=6$ ($O_6=2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hot`** | 0.16000000 | 0.05840000 | 0.02313600 | 0.00647584 | 0.00247919 | 0.00049362 | 0.00031569 |
| **`cold`** | 0.10000000 | 0.04560000 | 0.00461600 | 0.00462496 | 0.00196113 | 0.00098612 | 0.00027620 |
| **Column Sum** | **0.26000000** | **0.10400000** | **0.02775200** | **0.01110080** | **0.00444032** | **0.00147974** | **0.00059189** |

---

## 🔍 Verification & Experimental Results

| Metric | Forward Algorithm | Backward Algorithm | Absolute Difference | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Likelihood $P(O \mid \lambda)$** | **0.0005918945** | **0.0005918945** | **0.00e+00** | **PASSED (Exact Match)** |

### Scientific Notation
$$P(O \mid \lambda) = 5.9189 \times 10^{-4}$$

---

## 📁 Project Architecture & Code Breakdown

```text
.
├── demo.py                      # Standalone demonstration script with formatted ASCII outputs
├── hmm.py                       # Core class containing vectorized NumPy implementation
├── Hidden Markov Model.ipynb   # Interactive notebook with pre-executed cells and markdown
└── README.md                    # Technical documentation
```

### Module Responsibilities:
- **`hmm.py`**:
  - `likelihood_forward(obs)`: Computes forward matrix $\alpha$ and total likelihood.
  - `likelihood_backward(obs)`: Computes backward matrix $\beta$ for cross-verification.
  - `decode(obs)`: Implements Viterbi algorithm for most likely hidden state sequence.
  - `learn(obs, iterations)`: Implements Baum-Welch (Expectation-Maximization) parameter learning.
- **`demo.py`**:
  - Defines the ice cream model parameters.
  - Executes forward and backward passes.
  - Formats and prints the Alpha table and verification metrics directly to the console.
- **`Hidden Markov Model.ipynb`**:
  - Interactive visualization of all three canonical HMM problems (Likelihood, Decoding, Learning).

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.8+
- NumPy

### Run via Command Line / Terminal:
```bash
python demo.py
```
*(Alternatively, running `python hmm.py` also runs the exact demonstration).*

### Run via Jupyter Notebook:
Open `Hidden Markov Model.ipynb` in VS Code or JupyterLab and run all cells sequentially.

---

## 🖥️ Actual Execution Output

Running `python demo.py` produces the following terminal output:

```text
==============================================================================
 HIDDEN MARKOV MODEL: OBSERVATION LIKELIHOOD CALCULATION
 Module 5 - Forward Algorithm Demonstration
==============================================================================

[1] MODEL SPECIFICATION:
  * Hidden States (Weather)     : ['hot', 'cold']
  * Observation Space (Icecreams): [1, 2, 3]
  * Initial Probabilities (pi)  : Hot=0.8, Cold=0.2
  * Transition Matrix (A):
      From\To    hot     cold   
      hot       0.60    0.40   
      cold      0.50    0.50   
  * Emission Matrix (B):
      State\Obs  1 (low) 2 (med) 3 (high)
      hot       0.20    0.40    0.40   
      cold      0.50    0.40    0.10   

[2] GIVEN OBSERVATION SEQUENCE:
  * Observations (Ice Creams Eaten): [1, 2, 3, 2, 2, 1, 2]
  * Sequence Length (T)             : 7 days

[3] FORWARD PROBABILITY (ALPHA) TABLE:
  --------------------------------------------------------------------------
  State         t=0 (obs=1)   t=1 (obs=2)   t=2 (obs=3)   t=3 (obs=2)   t=4 (obs=2)   t=5 (obs=1)   t=6 (obs=2)
  --------------------------------------------------------------------------
  hot            0.16000000    0.05840000    0.02313600    0.00647584    0.00247919    0.00049362    0.00031569
  cold           0.10000000    0.04560000    0.00461600    0.00462496    0.00196113    0.00098612    0.00027620
  --------------------------------------------------------------------------
  Sum            0.26000000    0.10400000    0.02775200    0.01110080    0.00444032    0.00147974    0.00059189
  --------------------------------------------------------------------------

[4] FINAL OBSERVATION-SEQUENCE LIKELIHOOD:
  * P(O | lambda) [Decimal]             : 0.0005918945
  * P(O | lambda) [Scientific Notation] : 5.9189e-04

[5] ALGORITHM VERIFICATION (FORWARD vs BACKWARD):
  * Forward Algorithm Likelihood  : 0.0005918945
  * Backward Algorithm Likelihood : 0.0005918945
  * Absolute Difference           : 0.00e+00
  * Verification Status           : PASSED (Exact Match)

==============================================================================
 DEMONSTRATION COMPLETE - RESULTS READY FOR PRESENTATION
==============================================================================
```

---

## ⚠️ Limitations & Practical Considerations

1. **Floating-Point Numerical Underflow:**
   Multiplying probabilities ($< 1.0$) over long sequences ($T > 100$) can cause values to underflow to zero in standard floating-point representation. Production implementations mitigate this by operating in the **log-probability domain** or by using per-step scaling factors.
2. **First-Order Markov Limitation:**
   The assumption that state transitions depend only on the single preceding state restricts the model from capturing long-range contextual relationships.
3. **Discrete Observation Constraint:**
   This implementation processes categorical discrete symbols. Continuous real-valued data (e.g., sensor signals) requires Gaussian Mixture Model emissions (GMM-HMM).
