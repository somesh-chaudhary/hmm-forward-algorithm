"""
Hidden Markov Model (HMM) - Forward Algorithm Classroom Demonstration
College Topic : Observation-Sequence Likelihood Calculation Using HMM
Syllabus      : Module 5 - HMM Forward Algorithm

Team Members:
  1. 1AT23AI050 - SHIVASHANKAR
  2. 1AT23AI051 - SOMOIPTA NANDY
  3. 1AT23AI052 - SOMESH CHAUDHARY
"""

import numpy as np
from hmm import HiddenMarkovModel


def main():
    print("=" * 78)
    print(" HIDDEN MARKOV MODEL: OBSERVATION LIKELIHOOD CALCULATION")
    print(" Module 5 - Forward Algorithm Demonstration")
    print("=" * 78)

    # 1. Define Model Parameters (Hot/Cold Weather & Ice Cream Example)
    states = ["hot", "cold"]
    observations = [1, 2, 3]

    initial_probs = [0.8, 0.2]
    transition_probs = [
        [0.6, 0.4],  # hot  -> hot: 0.6,  hot  -> cold: 0.4
        [0.5, 0.5],  # cold -> hot: 0.5,  cold -> cold: 0.5
    ]
    emission_probs = [
        [0.2, 0.4, 0.4],  # hot  -> 1: 0.2, 2: 0.4, 3: 0.4
        [0.5, 0.4, 0.1],  # cold -> 1: 0.5, 2: 0.4, 3: 0.1
    ]

    hmm = HiddenMarkovModel(
        states, observations, transition_probs, emission_probs, initial_probs
    )

    # 2. Define Observation Sequence
    obs_seq = [1, 2, 3, 2, 2, 1, 2]
    T = len(obs_seq)

    print("\n[1] MODEL SPECIFICATION:")
    print(f"  * Hidden States (Weather)     : {states}")
    print(f"  * Observation Space (Icecreams): {observations}")
    print(f"  * Initial Probabilities (pi)  : Hot={initial_probs[0]}, Cold={initial_probs[1]}")
    print("  * Transition Matrix (A):")
    print("      From\\To    hot     cold")
    print(f"      hot       {transition_probs[0][0]:<7.2f} {transition_probs[0][1]:<7.2f}")
    print(f"      cold      {transition_probs[1][0]:<7.2f} {transition_probs[1][1]:<7.2f}")
    print("  * Emission Matrix (B):")
    print("      State\\Obs  1 (low) 2 (med) 3 (high)")
    print(f"      hot       {emission_probs[0][0]:<7.2f} {emission_probs[0][1]:<7.2f} {emission_probs[0][2]:<7.2f}")
    print(f"      cold      {emission_probs[1][0]:<7.2f} {emission_probs[1][1]:<7.2f} {emission_probs[1][2]:<7.2f}")

    print(f"\n[2] GIVEN OBSERVATION SEQUENCE:")
    print(f"  * Observations (Ice Creams Eaten): {obs_seq}")
    print(f"  * Sequence Length (T)             : {T} days")

    # 3. Calculate Forward Probabilities (Alpha Table)
    prob_forward, alpha = hmm.likelihood_forward(obs_seq)

    print("\n[3] FORWARD PROBABILITY (ALPHA) TABLE:")
    print("  " + "-" * 74)
    # Header
    col_headers = [f"t={t} (obs={o})" for t, o in enumerate(obs_seq)]
    header_str = f"  {'State':<8}" + "".join([f"{h:>14}" for h in col_headers])
    print(header_str)
    print("  " + "-" * 74)

    # State rows
    for i, state in enumerate(states):
        row_str = f"  {state:<8}" + "".join([f"{alpha[i, t]:>14.8f}" for t in range(T)])
        print(row_str)

    # Column sum (Total partial likelihood up to time t)
    print("  " + "-" * 74)
    col_sums = alpha.sum(axis=0)
    sum_str = f"  {'Sum':<8}" + "".join([f"{col_sums[t]:>14.8f}" for t in range(T)])
    print(sum_str)
    print("  " + "-" * 74)

    # 4. Display Final Likelihood
    print("\n[4] FINAL OBSERVATION-SEQUENCE LIKELIHOOD:")
    print(f"  * P(O | lambda) [Decimal]             : {prob_forward:.10f}")
    print(f"  * P(O | lambda) [Scientific Notation] : {prob_forward:.4e}")

    # 5. Verification Using Backward Algorithm
    prob_backward, beta = hmm.likelihood_backward(obs_seq)
    diff = abs(prob_forward - prob_backward)
    is_verified = np.isclose(prob_forward, prob_backward)

    print("\n[5] ALGORITHM VERIFICATION (FORWARD vs BACKWARD):")
    print(f"  * Forward Algorithm Likelihood  : {prob_forward:.10f}")
    print(f"  * Backward Algorithm Likelihood : {prob_backward:.10f}")
    print(f"  * Absolute Difference           : {diff:.2e}")
    print(f"  * Verification Status           : {'PASSED (Exact Match)' if is_verified else 'FAILED'}")

    print("\n" + "=" * 78)
    print(" DEMONSTRATION COMPLETE - RESULTS READY FOR PRESENTATION")
    print("=" * 78)


if __name__ == "__main__":
    main()

