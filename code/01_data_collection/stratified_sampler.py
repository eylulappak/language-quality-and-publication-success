#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# Configuration
# =========================
INPUT_CSV = "data\\full_arxiv_before_sampling.csv"
OUT_SAMPLE = "data\\arxiv_metadata_stratified_90k.csv"

RANDOM_SEED = 42

TOP10_CATEGORIES = [
    "Physics",
    "Computer Science",
    "Mathematics",
    "Engineering",
    "Materials Science",
    "Medicine",
    "Environmental Science",
    "Chemistry",
    "Economics",
    "Biology",
]

SAMPLE_SIZE = 90_000
SAMPLE_PER_DOI = SAMPLE_SIZE // 2

# Requested number of quantile bins before duplicate-edge cleanup
N_YEAR_BINS = 5
N_CIT_BINS = 5

# Outlier handling for citation binning
# Upper cap = Q3 + multiplier * IQR, computed on positive citation counts only
CIT_IQR_MULTIPLIER = 3.0


# =========================
# Helpers
# =========================
def find_column(df, candidates):
    lower_to_real = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_real:
            return lower_to_real[cand.lower()]
    raise ValueError(f"Could not find any of these columns: {candidates}")


def normalize_category(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s_lower = s.lower()

    mapping = {
        "physics": "Physics",
        "computer science": "Computer Science",
        "mathematics": "Mathematics",
        "engineering": "Engineering",
        "materials science": "Materials Science",
        "medicine": "Medicine",
        "environmental science": "Environmental Science",
        "chemistry": "Chemistry",
        "economics": "Economics",
        "biology": "Biology",
    }
    return mapping.get(s_lower, s)


def compute_positive_iqr_cap(series, multiplier=3.0):
    """Compute an upper cap for positive values using Q3 + multiplier * IQR."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]

    if s.empty:
        raise ValueError("No positive values available to compute citation cap.")

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    cap = q3 if iqr == 0 else q3 + multiplier * iqr

    # Make sure cap is at least the 95th percentile if IQR is tiny.
    p95 = s.quantile(0.95)
    return float(max(cap, p95))


def make_quantile_bins(series, n_bins, is_integer=False, zero_special=False):
    """
    Create quantile-based bin edges from actual data.

    If zero_special=True, value 0 gets its own bin and quantiles are computed
    only over positive values.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()

    if zero_special:
        positive = s[s > 0]
        if positive.empty:
            raise ValueError("No positive values available to form citation bins.")

        qs = np.linspace(0, 1, n_bins)
        qvals = positive.quantile(qs).to_numpy()
        qvals = np.unique(qvals)

        if len(qvals) < 2:
            raise ValueError("Not enough unique citation values to form quantile bins.")

        if is_integer:
            qvals = np.unique(np.round(qvals).astype(float))

        positive_edges = [x for x in qvals if x > 0]
        if not positive_edges:
            raise ValueError("Quantile computation produced no positive citation edges.")

        edges = [-0.1, 0.0] + positive_edges[1:] if len(positive_edges) > 1 else [-0.1, 0.0, np.inf]

        cleaned = [edges[0]]
        for e in edges[1:]:
            if e > cleaned[-1]:
                cleaned.append(e)
        edges = cleaned

        labels = []
        for i in range(len(edges) - 1):
            left, right = edges[i], edges[i + 1]

            if i == 0 and left < 0 and right == 0:
                labels.append("0")
            else:
                if float(left).is_integer():
                    ltxt = str(int(left + 1))
                else:
                    ltxt = f"{left:.2f}"

                if i == len(edges) - 2:
                    labels.append(f"{ltxt}+")
                else:
                    r_display = int(right) if float(right).is_integer() else round(float(right), 2)
                    labels.append(f"{ltxt}-{r_display}")

        return edges, labels

    qs = np.linspace(0, 1, n_bins + 1)
    qvals = s.quantile(qs).to_numpy()
    qvals = np.unique(qvals)

    if len(qvals) < 2:
        raise ValueError("Not enough unique values to form quantile bins.")

    if is_integer:
        qvals = np.unique(np.round(qvals).astype(float))
        if len(qvals) < 2:
            raise ValueError("Not enough unique integer-rounded values to form bins.")

    edges = qvals.tolist()

    cleaned = [edges[0]]
    for e in edges[1:]:
        if e > cleaned[-1]:
            cleaned.append(e)
    edges = cleaned

    labels = []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        ltxt = str(int(left)) if float(left).is_integer() else f"{left:.2f}"
        rtxt = str(int(right)) if float(right).is_integer() else f"{right:.2f}"
        labels.append(f"{ltxt}-{rtxt}")

    return edges, labels


def proportional_allocation(capacities, target):
    capacities = capacities.astype(int).copy()

    if target < 0:
        raise ValueError("target must be >= 0")
    if capacities.sum() < target:
        raise ValueError(
            f"Not enough rows to allocate target={target}; only {capacities.sum()} available."
        )
    if target == 0:
        return pd.Series(0, index=capacities.index, dtype=int)

    alloc = pd.Series(0, index=capacities.index, dtype=int)
    remaining_target = int(target)
    remaining_caps = capacities.copy()

    while remaining_target > 0:
        positive = remaining_caps[remaining_caps > 0]
        if positive.empty:
            raise RuntimeError("Ran out of capacity during allocation.")

        weights = positive / positive.sum()
        raw = weights * remaining_target
        base = np.floor(raw).astype(int)
        base = np.minimum(base, positive)

        allocated_now = int(base.sum())
        alloc.loc[base.index] += base
        remaining_caps.loc[base.index] -= base
        remaining_target -= allocated_now

        if remaining_target == 0:
            break

        positive = remaining_caps[remaining_caps > 0]
        weights = positive / positive.sum()
        raw = weights * remaining_target
        frac = raw - np.floor(raw)

        order = frac.sort_values(ascending=False).index.tolist()
        i = 0
        while remaining_target > 0:
            k = order[i % len(order)]
            if remaining_caps.loc[k] > 0:
                alloc.loc[k] += 1
                remaining_caps.loc[k] -= 1
                remaining_target -= 1
            i += 1

    return alloc.astype(int)


def sample_from_counts(df_group, counts, rng):
    pieces = []
    for stratum, n in counts.items():
        if n <= 0:
            continue
        sub = df_group[df_group["stratum"] == stratum]
        if len(sub) < n:
            raise ValueError(
                f"Stratum {stratum} has only {len(sub)} rows but asked for {n}."
            )
        idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        pieces.append(df_group.loc[idx])

    if not pieces:
        return df_group.iloc[0:0].copy()

    return pd.concat(pieces, axis=0)


def distribution_table(df, name):
    print(f"\n=== {name} ===")
    print(f"Rows: {len(df):,}")

    print("\nDOI counts:")
    print(df["doi_bin"].value_counts(dropna=False).sort_index().to_string())

    print("\nYear-bin distribution (%):")
    print((df["year_bin"].value_counts(normalize=True).sort_index() * 100).round(2).to_string())

    print("\nCitation-bin distribution (%):")
    print((df["citation_bin"].value_counts(normalize=True).sort_index() * 100).round(2).to_string())

    print("\nCategory distribution (%):")
    print((df["category1_norm"].value_counts(normalize=True).sort_index() * 100).round(2).to_string())


# =========================
# Main
# =========================
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, dtype=str)
    print(f"Loaded {len(df):,} rows")

    year_col = find_column(df, ["year"])
    cit_col = find_column(df, ["citationcount", "citation_count", "citations", "citation"])
    cat_col = find_column(df, ["category1"])
    doi_col = find_column(df, ["has_doi", "doi", "doi_present", "hasdoi"])

    print("\nUsing columns:")
    print(f"  year       -> {year_col}")
    print(f"  citations  -> {cit_col}")
    print(f"  category1  -> {cat_col}")
    print(f"  DOI        -> {doi_col}")

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[cit_col] = pd.to_numeric(df[cit_col], errors="coerce")
    df["category1_norm"] = df[cat_col].map(normalize_category)
    df["doi_bin"] = pd.to_numeric(df[doi_col], errors="coerce")

    df = df[df["doi_bin"].isin([0, 1])].copy()
    df["doi_bin"] = df["doi_bin"].astype(int)

    # Keep years <= 2021, >= 2005
    df = df[
        df[year_col].notna()
        & (df[year_col] <= 2021)
        & (df[year_col] >= 2005)
        & df[cit_col].notna()
        & df["category1_norm"].isin(TOP10_CATEGORIES)
    ].copy()

    print(f"\nAfter filtering: {len(df):,} rows")

    # -------------------------
    # Quantile-based year bins
    # -------------------------
    year_edges, year_labels = make_quantile_bins(
        df[year_col],
        n_bins=N_YEAR_BINS,
        is_integer=True,
        zero_special=False,
    )

    # -------------------------
    # Citation binning with outlier cap
    # -------------------------
    citation_cap = compute_positive_iqr_cap(df[cit_col], multiplier=CIT_IQR_MULTIPLIER)
    print(
        f"\nCitation cap for binning "
        f"(Q3 + {CIT_IQR_MULTIPLIER}*IQR on positive citations): {citation_cap:.2f}"
    )

    # Clip only for bin construction, not for output values.
    df["citation_for_binning"] = df[cit_col].clip(upper=citation_cap)

    cit_edges, cit_labels = make_quantile_bins(
        df["citation_for_binning"],
        n_bins=N_CIT_BINS,
        is_integer=True,
        zero_special=True,
    )

    print("\nDerived year bin edges:")
    print(year_edges)
    print("Derived year labels:")
    print(year_labels)

    print("\nDerived citation bin edges after clipping extreme outliers for binning:")
    print(cit_edges)
    print("Derived citation labels:")
    print(cit_labels)

    df["year_bin"] = pd.cut(
        df[year_col],
        bins=year_edges,
        labels=year_labels,
        include_lowest=True,
        right=True,
    )

    df["citation_bin"] = pd.cut(
        df["citation_for_binning"],
        bins=cit_edges,
        labels=cit_labels,
        include_lowest=True,
        right=True,
    )

    df = df[df["year_bin"].notna() & df["citation_bin"].notna()].copy()

    df["stratum"] = (
        df["year_bin"].astype(str)
        + " || " + df["citation_bin"].astype(str)
        + " || " + df["category1_norm"].astype(str)
    )

    doi_counts = df["doi_bin"].value_counts()
    n0 = int(doi_counts.get(0, 0))
    n1 = int(doi_counts.get(1, 0))

    print("\nAvailable rows after filtering:")
    print(f"  DOI=0: {n0:,}")
    print(f"  DOI=1: {n1:,}")

    if n0 < SAMPLE_PER_DOI or n1 < SAMPLE_PER_DOI:
        raise ValueError(
            f"Not enough rows after filtering to create a balanced 100K sample.\n"
            f"Need at least {SAMPLE_PER_DOI:,} rows for each DOI value, "
            f"but got DOI=0: {n0:,}, DOI=1: {n1:,}."
        )

    distribution_table(df, "Filtered full pool")

    sample_parts = []

    for doi_value in [0, 1]:
        print(f"\nSampling DOI={doi_value} ...")
        sub = df[df["doi_bin"] == doi_value].copy()

        caps = sub["stratum"].value_counts().sort_index()
        sample_counts = proportional_allocation(caps, SAMPLE_PER_DOI)
        sample_sub = sample_from_counts(sub, sample_counts, rng)

        sample_parts.append(sample_sub)
        print(f"  Rows sampled: {len(sample_sub):,}")

    sample_df = pd.concat(sample_parts, axis=0).copy()
    sample_df["__source_index"] = sample_df.index

    sample_df = sample_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    assert len(sample_df) == SAMPLE_SIZE
    assert sample_df["doi_bin"].value_counts().to_dict().get(0, 0) == SAMPLE_PER_DOI
    assert sample_df["doi_bin"].value_counts().to_dict().get(1, 0) == SAMPLE_PER_DOI
    assert sample_df["__source_index"].is_unique

    Path(OUT_SAMPLE).parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(OUT_SAMPLE, index=False)

    distribution_table(sample_df, "Stratified 100K sample")

    print("\nDone.")
    print(f"Sample written to: {OUT_SAMPLE}")


if __name__ == "__main__":
    main()
