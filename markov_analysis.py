"""
Discrete-Time Markov Chain Analysis of Daily Stock Returns (AAPL)

Models discretized daily percentage returns (R1–R7) as a first-order Markov chain,
verifies chain ergodicity, computes the stationary distribution, mean recurrence times,
convergence rates, and n-step first-return probabilities, and generates diagnostic plots.

Usage:
    python markov_analysis.py --csv data/AAPL_states.csv --output outputs
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
STATE_LABELS = {
    "R1": "R1 (< -2%)",
    "R2": "R2 [-2%, -1%)",
    "R3": "R3 [-1%, 0%)",
    "R4": "R4 (= 0%)",
    "R5": "R5 (0%, 1%]",
    "R6": "R6 (1%, 2%]",
    "R7": "R7 (> 2%)",
}


def load_states(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "State" not in df.columns:
        raise ValueError("Input CSV must contain a 'State' column.")
    df["State"] = df["State"].astype(str).str.strip()
    invalid = ~df["State"].isin(STATES)
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} rows with unrecognized states.")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_transition_matrices(state_seq: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.DataFrame(0, index=STATES, columns=STATES, dtype=int)
    for s_curr, s_next in zip(state_seq[:-1], state_seq[1:]):
        counts.loc[s_curr, s_next] += 1

    row_sums = counts.sum(axis=1)
    if (row_sums == 0).any():
        empty_states = row_sums[row_sums == 0].index.tolist()
        raise ValueError(f"No outgoing transitions observed for states: {empty_states}")

    probs = counts.div(row_sums, axis=0)
    return counts, probs


def analyze_chain_properties(P: np.ndarray) -> dict:
    # Check absorbing states (P_ii == 1)
    absorbing_states = [STATES[i] for i in range(len(STATES)) if np.isclose(P[i, i], 1.0)]

    # Check regularity (smallest power k such that all entries of P^k > 0)
    regular_power = None
    Pk = P.copy()
    for k in range(1, 51):
        if np.all(Pk > 0):
            regular_power = k
            break
        Pk = Pk @ P

    # Eigenvalues of P^T
    eigvals, eigvecs = np.linalg.eig(P.T)
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]

    # Stationary distribution corresponding to lambda = 1
    pi_stat = np.real(eigvecs_sorted[:, 0])
    pi_stat = pi_stat / pi_stat.sum()
    stationary_residual = float(np.max(np.abs(pi_stat @ P - pi_stat)))

    second_eig_modulus = float(np.abs(eigvals_sorted[1]))

    return {
        "absorbing_states": absorbing_states,
        "is_regular": regular_power is not None,
        "regular_power": regular_power,
        "second_eigenvalue_modulus": second_eig_modulus,
        "stationary_distribution": pi_stat,
        "stationary_residual": stationary_residual,
    }


def compute_first_return_probabilities(P: np.ndarray, max_n: int = 20) -> pd.DataFrame:
    """
    Compute first-return probabilities f_{ii}^{(n)} using the recurrence:
        f_{ii}^{(n)} = P^{(n)}_{ii} - sum_{m=1}^{n-1} f_{ii}^{(m)} * P^{(n-m)}_{ii}
    """
    k = P.shape[0]
    f = np.zeros((k, max_n), dtype=float)
    p_powers = [np.linalg.matrix_power(P, n) for n in range(1, max_n + 1)]

    for n in range(1, max_n + 1):
        for i in range(k):
            val = p_powers[n - 1][i, i]
            for m in range(1, n):
                val -= f[i, m - 1] * p_powers[n - m - 1][i, i]
            f[i, n - 1] = max(val, 0.0)

    return pd.DataFrame(f, index=STATES, columns=[f"n={n}" for n in range(1, max_n + 1)])


def compute_convergence(
    P: np.ndarray,
    pi_emp: np.ndarray,
    pi_stat: np.ndarray,
    max_steps: int = 15,
) -> pd.DataFrame:
    """Track L_inf distance to the stationary distribution from multiple initial distributions."""
    e_r1 = np.zeros(len(STATES))
    e_r1[0] = 1.0  # Starting with certainty in R1 (sharp drop < -2%)

    e_r4 = np.zeros(len(STATES))
    e_r4[3] = 1.0  # Starting with certainty in R4 (zero return, rarest state)

    records = []
    for n in range(max_steps + 1):
        Pn = np.linalg.matrix_power(P, n)
        d_emp = float(np.max(np.abs(pi_emp @ Pn - pi_stat)))
        d_r1 = float(np.max(np.abs(e_r1 @ Pn - pi_stat)))
        d_r4 = float(np.max(np.abs(e_r4 @ Pn - pi_stat)))
        records.append(
            {
                "step": n,
                "linf_from_empirical_pi0": d_emp,
                "linf_from_R1": d_r1,
                "linf_from_R4": d_r4,
            }
        )
    return pd.DataFrame(records)


def plot_transition_heatmap(P_df: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(P_df.values, cmap="Blues", vmin=0.0, vmax=max(0.45, float(P_df.values.max())))

    ax.set_xticks(range(len(STATES)))
    ax.set_yticks(range(len(STATES)))
    ax.set_xticklabels(STATES)
    ax.set_yticklabels(STATES)
    ax.set_xlabel("Next-Day State")
    ax.set_ylabel("Current-Day State")
    ax.set_title("AAPL Daily Return Transition Probability Matrix (P)")

    thresh = P_df.values.max() * 0.55
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            val = P_df.values[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Transition Probability")
    plt.tight_layout()
    plt.savefig(plots_dir / "transition_heatmap.png", dpi=200)
    plt.close()


def plot_state_distributions(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    x = np.arange(len(STATES))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, summary_df["empirical_prob"], width, label=r"Empirical Distribution ($\pi^{(0)}$)")
    ax.bar(x + width / 2, summary_df["stationary_prob"], width, label=r"Stationary Distribution ($\pi$)")

    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABELS[s] for s in STATES], rotation=20, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Empirical vs. Stationary State Distribution (AAPL)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "state_distribution_comparison.png", dpi=200)
    plt.close()


def plot_convergence(conv_df: pd.DataFrame, second_eig: float, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    eps = 1e-16
    ax.semilogy(
        conv_df["step"],
        np.maximum(conv_df["linf_from_R4"], eps),
        marker="o",
        label=r"Start at $R_4$ (rarest state)",
    )
    ax.semilogy(
        conv_df["step"],
        np.maximum(conv_df["linf_from_R1"], eps),
        marker="s",
        label=r"Start at $R_1$ (return < -2%)",
    )
    ax.semilogy(
        conv_df["step"],
        np.maximum(conv_df["linf_from_empirical_pi0"], eps),
        marker="^",
        label=r"Start at empirical $\pi^{(0)}$",
    )

    ax.set_xlabel("Time Step ($n$ trading days)")
    ax.set_ylabel(r"Max Deviation $\|\pi^{(n)} - \pi\|_\infty$ (log scale)")
    ax.set_title(f"Convergence to Stationary Distribution ($|\\lambda_2| = {second_eig:.4f}$)")
    ax.set_xticks(conv_df["step"])
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "convergence_curve.png", dpi=200)
    plt.close()


def plot_first_return(frp_df: pd.DataFrame, plots_dir: Path, max_plot_n: int = 15) -> None:
    steps = list(range(1, min(max_plot_n, frp_df.shape[1]) + 1))
    cols = [f"n={n}" for n in steps]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for state in STATES:
        ax.plot(
            steps,
            frp_df.loc[state, cols].values,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=STATE_LABELS[state],
        )

    ax.set_xlabel("Step ($n$ trading days)")
    ax.set_ylabel(r"First-Return Probability $f_{ii}^{(n)}$")
    ax.set_title("First-Return Probabilities Across Return States")
    ax.set_xticks(steps)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "first_return_probabilities.png", dpi=200)
    plt.close()


def save_summary_report(
    df: pd.DataFrame,
    props: dict,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    report_path = output_dir / "summary_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("AAPL Daily Return Markov Chain Analysis - Summary Report\n")
        f.write("=" * 58 + "\n\n")
        if "date" in df.columns:
            f.write(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}\n")
        f.write(f"Total trading days: {len(df):,}\n")
        f.write(f"Total transitions:  {len(df) - 1:,}\n\n")

        f.write("Chain Structural Properties\n")
        f.write("-" * 30 + "\n")
        f.write(f"Regular (ergodic):         {props['is_regular']} (P^{props['regular_power']} > 0)\n")
        f.write(f"Absorbing states:          {props['absorbing_states'] or 'None'}\n")
        f.write(f"Second eigenvalue |lam_2|: {props['second_eigenvalue_modulus']:.6f}\n")
        f.write(f"Stationary residual:       {props['stationary_residual']:.2e}\n\n")

        f.write("Empirical vs. Stationary Distribution & Mean Recurrence Time\n")
        f.write("-" * 60 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze discretized daily stock returns as a 7-state discrete-time Markov chain."
    )
    parser.add_argument("--csv", default="data/AAPL_states.csv", help="Path to discretized states CSV")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--steps", type=int, default=20, help="Max steps for first-return probabilities")
    args = parser.parse_args()

    output_dir = Path(args.output)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_states(Path(args.csv))
    counts_df, P_df = compute_transition_matrices(df["State"].tolist())
    P = P_df.values

    state_counts = df["State"].value_counts().reindex(STATES).fillna(0).astype(int)
    pi_emp = (state_counts / len(df)).values

    props = analyze_chain_properties(P)
    pi_stat = props["stationary_distribution"]

    summary_df = pd.DataFrame(
        {
            "state": STATES,
            "count": state_counts.values,
            "empirical_prob": pi_emp,
            "stationary_prob": pi_stat,
            "mean_recurrence_days": 1.0 / pi_stat,
        }
    )

    frp_df = compute_first_return_probabilities(P, max_n=args.steps)
    conv_df = compute_convergence(P, pi_emp, pi_stat, max_steps=15)

    # Save tabular outputs
    counts_df.to_csv(output_dir / "transition_counts.csv")
    P_df.to_csv(output_dir / "transition_matrix.csv")
    summary_df.to_csv(output_dir / "stationary_distribution.csv", index=False)
    frp_df.to_csv(output_dir / "first_return_probabilities.csv")
    conv_df.to_csv(output_dir / "convergence_metrics.csv", index=False)
    save_summary_report(df, props, summary_df, output_dir)

    # Generate plots
    plot_transition_heatmap(P_df, plots_dir)
    plot_state_distributions(summary_df, plots_dir)
    plot_convergence(conv_df, props["second_eigenvalue_modulus"], plots_dir)
    plot_first_return(frp_df, plots_dir)

    print("Analysis complete.")
    print(f"Trading days: {len(df):,} | Transitions: {len(df) - 1:,}")
    print(
        f"Regular chain (P^{props['regular_power']} > 0): {props['is_regular']} | "
        f"|lambda_2| = {props['second_eigenvalue_modulus']:.4f}"
    )
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
