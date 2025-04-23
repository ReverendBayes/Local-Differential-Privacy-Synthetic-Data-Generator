### LDP-SynthData 
# Local Differential Privacy Synthetic-Data Generator 

A single-file CLI that generates privacy-preserving synthetic CSVs via local differential privacy (Laplace &amp; randomized response).

## What it is 
A CLI that takes your sensitive tabular CSV and produces a privacy-preserving synthetic twin via local-DP (e.g. randomized response + local noise).

## Why you need it
* Privacy regulations (GDPR, CCPA) are forcing every team to explore synthetic data.
* Local DP (each user perturbs on-device) is hotter research than “central DP.”
* Nobody’s distilled it to a one-file script—existing toolchains are multi-repo or Jupyter mashups.

## Key features
python dpgen.py \
  --input data.csv \
  --epsilon 1.0 \
  --mechanism laplace \
  --output synthetic.csv
* Supports categorical & numeric columns
* Pluggable noise mechanisms (Laplace, Gaussian, randomized-response)
* Minimal deps: numpy, pandas, scikit-learn


## Usage example
python dpgen.py \
  --input your_data.csv \
  --epsilon 1.0 \
  --mechanism all \
  --output synthetic.csv

  ## Dependencies
  pip install pandas numpy
