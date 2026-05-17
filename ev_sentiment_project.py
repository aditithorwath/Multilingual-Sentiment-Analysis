import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


REQUIRED_COLUMNS = ("text", "language", "sentiment", "sentiment_label")
SENTIMENT_LABELS = ("Negative", "Neutral", "Positive")


@dataclass
class SentimentAnalysis:
    total_rows: int
    languages: List[str]
    sentiment_counts: Counter
    language_sentiment_counts: Dict[str, Counter]


def load_dataset(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        return list(reader)


def analyze_rows(rows: Iterable[dict]) -> SentimentAnalysis:
    row_list = list(rows)
    sentiment_counts = Counter(row["sentiment_label"] for row in row_list)
    language_counts = defaultdict(Counter)

    for row in row_list:
        language_counts[row["language"]][row["sentiment_label"]] += 1

    languages = sorted(language_counts.keys())
    return SentimentAnalysis(
        total_rows=len(row_list),
        languages=languages,
        sentiment_counts=sentiment_counts,
        language_sentiment_counts=dict(language_counts),
    )


def write_language_summary(analysis: SentimentAnalysis, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["language", *SENTIMENT_LABELS, "total"])
        writer.writeheader()

        for language in analysis.languages:
            counts = analysis.language_sentiment_counts[language]
            row = {"language": language}
            row.update({label: counts[label] for label in SENTIMENT_LABELS})
            row["total"] = sum(counts.values())
            writer.writerow(row)


def write_project_summary(analysis: SentimentAnalysis, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    most_common = analysis.sentiment_counts.most_common(1)
    dominant_sentiment = most_common[0][0] if most_common else "None"

    lines = [
        "EV/Tesla Battery Sentiment Analysis Summary",
        "",
        f"Total records analyzed: {analysis.total_rows}",
        f"Languages included: {', '.join(analysis.languages)}",
        f"Dominant sentiment: {dominant_sentiment}",
        "",
        "Overall sentiment counts:",
    ]
    for label in SENTIMENT_LABELS:
        lines.append(f"- {label}: {analysis.sentiment_counts[label]}")

    lines.extend(["", "Language-wise sentiment counts:"])
    for language in analysis.languages:
        counts = analysis.language_sentiment_counts[language]
        label_counts = ", ".join(f"{label}: {counts[label]}" for label in SENTIMENT_LABELS)
        lines.append(f"- {language}: {label_counts}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bar_chart(title: str, counts: Dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    max_value = max(values) if values else 1
    width = 760
    height = 420
    margin_left = 90
    margin_bottom = 80
    chart_width = width - margin_left - 40
    chart_height = height - 90 - margin_bottom
    bar_gap = 24
    bar_width = max(36, (chart_width - bar_gap * max(0, len(labels) - 1)) / max(1, len(labels)))
    colors = {
        "Negative": "#d94c4c",
        "Neutral": "#68707a",
        "Positive": "#2f9e66",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 40}" y2="{height - margin_bottom}" stroke="#2f3437"/>',
        f'<line x1="{margin_left}" y1="70" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#2f3437"/>',
    ]

    for index, label in enumerate(labels):
        value = counts[label]
        bar_height = (value / max_value) * chart_height if max_value else 0
        x = margin_left + index * (bar_width + bar_gap)
        y = height - margin_bottom - bar_height
        color = colors.get(label, "#4f81bd")
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="14">{value}</text>',
                f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 28}" text-anchor="middle" font-family="Arial" font-size="14">{label}</text>',
            ]
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_language_chart(analysis: SentimentAnalysis, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 860
    row_height = 44
    height = 120 + row_height * len(analysis.languages)
    label_x = 70
    bar_x = 140
    bar_width = 560
    colors = {
        "Negative": "#d94c4c",
        "Neutral": "#68707a",
        "Positive": "#2f9e66",
    }
    max_total = max(
        (sum(analysis.language_sentiment_counts[language].values()) for language in analysis.languages),
        default=1,
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Language-wise Sentiment Analysis</text>',
    ]

    for row_index, language in enumerate(analysis.languages):
        y = 78 + row_index * row_height
        counts = analysis.language_sentiment_counts[language]
        total = sum(counts.values())
        x = bar_x
        parts.append(f'<text x="{label_x}" y="{y + 22}" text-anchor="middle" font-family="Arial" font-size="15">{language}</text>')
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="26" fill="#f1f3f5"/>')

        for label in SENTIMENT_LABELS:
            segment_width = (counts[label] / max_total) * bar_width if max_total else 0
            if segment_width:
                parts.append(
                    f'<rect x="{x:.1f}" y="{y}" width="{segment_width:.1f}" height="26" fill="{colors[label]}"/>'
                )
            x += segment_width
        parts.append(f'<text x="{bar_x + bar_width + 20}" y="{y + 19}" font-family="Arial" font-size="14">{total}</text>')

    legend_y = height - 28
    legend_x = 220
    for index, label in enumerate(SENTIMENT_LABELS):
        x = legend_x + index * 145
        parts.append(f'<rect x="{x}" y="{legend_y - 14}" width="16" height="16" fill="{colors[label]}"/>')
        parts.append(f'<text x="{x + 22}" y="{legend_y}" font-family="Arial" font-size="14">{label}</text>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def run_project(input_path: Path, output_dir: Path) -> SentimentAnalysis:
    rows = load_dataset(input_path)
    analysis = analyze_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_language_summary(analysis, output_dir / "sentiment_summary.csv")
    write_project_summary(analysis, output_dir / "project_summary.txt")
    write_bar_chart(
        "Sentiment Distribution",
        {label: analysis.sentiment_counts[label] for label in SENTIMENT_LABELS},
        output_dir / "sentiment_distribution.svg",
    )
    write_language_chart(analysis, output_dir / "language_sentiment_distribution.svg")

    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multilingual EV battery sentiment results.")
    parser.add_argument("--input", type=Path, default=Path("data/final_project_output.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = run_project(args.input, args.output_dir)
    print(f"Analyzed {analysis.total_rows} rows across {len(analysis.languages)} languages.")
    print(f"Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
