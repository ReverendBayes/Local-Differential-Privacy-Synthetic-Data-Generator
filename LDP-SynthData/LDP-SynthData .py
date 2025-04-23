#!/usr/bin/env python3
"""
dpgen.py: Local Differential Privacy Synthetic Data Generator

Usage:
    python dpgen.py --input data.csv --epsilon 1.0 --mechanism all --output synthetic.csv

Features:
    - Numeric columns: Laplace mechanism
    - Categorical columns: Randomized response
    - Combined mode: apply both to respective columns
    - Minimal dependencies: pandas, numpy
"""
import argparse
import pandas as pd
import numpy as np
import math

def laplace_perturb(column, epsilon):
    """
    Apply Laplace mechanism to a numeric pandas Series.
    Noise scale = (max - min) / epsilon
    """
    col = column.astype(float)
    sensitivity = col.max() - col.min()
    scale = sensitivity / epsilon if epsilon > 0 else 0
    noise = np.random.laplace(loc=0.0, scale=scale, size=col.shape)
    return col + noise

def randomized_response(column, epsilon):
    """
    Apply randomized response to a categorical pandas Series.
    For k categories, probability of truth p = exp(eps) / (exp(eps) + k - 1)
    """
    cat = column.astype(str)
    categories = cat.unique()
    k = len(categories)
    if k <= 1:
        return cat
    p = math.exp(epsilon) / (math.exp(epsilon) + k - 1)
    synthetic = []
    for val in cat:
        if np.random.rand() < p:
            synthetic.append(val)
        else:
            # sample a different category uniformly
            others = [c for c in categories if c != val]
            synthetic.append(np.random.choice(others))
    return pd.Series(synthetic, index=column.index)

def generate_synthetic(df, epsilon, mechanism):
    synth = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if mechanism in ('laplace', 'all'):
                synth[col] = laplace_perturb(df[col], epsilon)
        else:
            if mechanism in ('rr', 'all'):
                synth[col] = randomized_response(df[col], epsilon)
    return synth

def main():
    parser = argparse.ArgumentParser(description='Local DP Synthetic Data Generator')
    parser.add_argument('--input', '-i', required=True, help='Path to input CSV')
    parser.add_argument('--epsilon', '-e', type=float, required=True, help='Privacy budget ε')
    parser.add_argument('--mechanism', '-m', choices=['laplace','rr','all'], default='all',
                        help='Mechanism: laplace (numeric), rr (categorical), all (both)')
    parser.add_argument('--output', '-o', required=True, help='Path to output synthetic CSV')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    synthetic_df = generate_synthetic(df, args.epsilon, args.mechanism)
    synthetic_df.to_csv(args.output, index=False)
    print(f"Synthetic data written to {args.output}")

if __name__ == '__main__':
    main()
