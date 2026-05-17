# EV/Tesla Battery Sentiment Analysis

This project analyzes multilingual sentiment data about EV/Tesla battery recall concerns. It uses the provided dataset in `data/final_project_output.csv`, summarizes the sentiment labels, and creates output files that can be used in a report or presentation.

## Purpose

The project helps understand public feedback about EV battery safety issues. Companies can use this kind of analysis to identify customer concerns, compare reactions across languages, and decide where product quality, safety communication, or customer support should improve.

## Dataset

The input CSV contains:

- `text`: customer/public discussion text
- `language`: language code such as `en`, `hi`, `es`, `fr`, or `zh`
- `sentiment`: numeric model output
- `sentiment_label`: readable sentiment label

## Run

```bash
python3 ev_sentiment_project.py
```

## Outputs

The script writes these files to `outputs/`:

- `sentiment_summary.csv`: language-wise sentiment counts
- `project_summary.txt`: readable project summary
- `sentiment_distribution.svg`: overall sentiment chart
- `language_sentiment_distribution.svg`: language-wise sentiment chart

## Test

```bash
python3 -m unittest tests/test_ev_sentiment_project.py
```
