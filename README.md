# Clinical Trial Figure & Graph Data Extractor

Extract all data from figures and graphs (including individual data points) in
clinical trial research papers and output the results as a structured JSON
document for easy downstream analysis.

---

## How It Works

1. **PDF parsing** – [PyMuPDF](https://pymupdf.readthedocs.io) opens the paper,
   renders each page, and extracts embedded raster images.
2. **Caption detection** – page text is scanned for patterns such as
   *"Figure 1."* / *"Fig. 2A –"* and matched to the extracted images.
3. **AI-powered data extraction** – each figure image (plus any matching
   caption) is sent to an OpenAI vision model (default `gpt-4o`) with a
   detailed prompt that instructs the model to return structured JSON covering:
   - Figure type (Kaplan–Meier, bar chart, forest plot, scatter plot, …)
   - Axis labels, units, ranges, and scale
   - All data series and individual data points (x, y, confidence intervals)
   - Statistical annotations (p-value, hazard ratio, odds ratio, …)
   - At-risk tables for survival curves
   - Forest-plot row-level data
   - CONSORT / flow-diagram node–edge structure
   - Heatmap matrices and embedded tables
4. **JSON assembly** – all per-figure results are collected into a single
   top-level JSON document.

---

## Requirements

- Python ≥ 3.10
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to a
  vision-capable model (`gpt-4o` recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your API key (choose one method):

```bash
# Option A – environment variable
export OPENAI_API_KEY="sk-..."

# Option B – .env file in the project directory
echo 'OPENAI_API_KEY=sk-...' > .env
```

---

## Usage

### Command-line interface

```bash
python extractor.py path/to/paper.pdf
```

This creates `paper.json` in the same directory as the PDF.

**All options:**

```
usage: extractor.py [-h] [--output OUTPUT] [--model MODEL] [--api-key API_KEY]
                    [--dpi DPI] [--include-page-images] [--min-image-size N]
                    [--indent N] [--verbose]
                    pdf

positional arguments:
  pdf                   Path to the input PDF file.

optional arguments:
  --output, -o          Output JSON path (default: same stem as PDF + .json).
  --model               OpenAI vision model (default: gpt-4o).
  --api-key             OpenAI API key (overrides env variable).
  --dpi                 Page render resolution in DPI (default: 150).
  --include-page-images Embed base-64 page images in the JSON output.
  --min-image-size N    Minimum image dimension in pixels to consider as a
                        figure (default: 100; filters icons/logos).
  --indent N            JSON indentation level (default: 2).
  --verbose, -v         Enable DEBUG logging.
```

### Python API

```python
from extractor import ClinicalTrialFigureExtractor

extractor = ClinicalTrialFigureExtractor(api_key="sk-...")

# Returns a dict
result = extractor.extract_from_pdf("paper.pdf")

# Write directly to a JSON file
extractor.save_json("paper.pdf", "results.json")
```

---

## Output JSON Schema

```jsonc
{
  "source_file": "paper.pdf",
  "extraction_timestamp": "2024-06-01T12:00:00+00:00",
  "figures": [
    {
      "figure_id": "fig_1",
      "page_number": 3,
      "caption": "Figure 1. Kaplan–Meier overall survival curves…",

      // ── Fields returned by the vision model ───────────────────────────
      "figure_type": "kaplan_meier",   // see list below
      "title": "Overall Survival",
      "axes": {
        "x_axis": { "label": "Time (months)", "unit": "months",
                    "range": [0, 60], "scale": "linear" },
        "y_axis": { "label": "Probability of Survival", "unit": null,
                    "range": [0, 1], "scale": "linear" }
      },
      "data_series": [
        {
          "name": "Treatment A",
          "color": "blue",
          "line_style": "solid",
          "data_points": [
            { "x": 0,  "y": 1.0,  "ci_lower": null, "ci_upper": null },
            { "x": 12, "y": 0.84, "ci_lower": 0.77, "ci_upper": 0.91 },
            { "x": 24, "y": 0.68, "ci_lower": 0.59, "ci_upper": 0.77 }
          ],
          "summary_statistics": {
            "median": 38, "hazard_ratio": 0.70
          }
        }
      ],
      "statistical_annotations": {
        "p_value": "0.002",
        "hazard_ratio": "0.70 (95% CI 0.55–0.89)",
        "log_rank": null,
        "sample_size": "240"
      },
      "at_risk_table": {
        "time_points": [0, 12, 24, 36, 48],
        "series": {
          "Treatment A": [120, 98, 72, 45, 18],
          "Control":     [120, 88, 58, 32, 11]
        }
      },

      // Forest-plot specific
      "forest_plot_rows": [
        {
          "subgroup": "Age < 65",
          "n_treatment": 60, "n_control": 58,
          "effect_size": 0.65, "ci_lower": 0.45, "ci_upper": 0.94,
          "weight": 32.1
        }
      ],

      // CONSORT / flow-diagram specific
      "consort_flow": {
        "nodes": [{ "id": "n1", "label": "Enrolled", "count": 500 }],
        "edges": [{ "from": "n1", "to": "n2", "label": "Randomised" }]
      },

      // Heatmap specific
      "heatmap_data": {
        "row_labels": ["Gene A", "Gene B"],
        "col_labels": ["Sample 1", "Sample 2"],
        "values": [[1.2, -0.5], [0.8, 2.1]]
      },

      // Embedded table specific
      "table_data": {
        "headers": ["Variable", "Treatment", "Control"],
        "rows": [["Age (median)", "58", "60"]]
      },

      "extraction_confidence": "high",  // high | medium | low
      "notes": "At-risk numbers read from table below figure."
    }
  ],
  "extraction_summary": {
    "total_figures": 5,
    "figure_types": {
      "kaplan_meier": 2,
      "forest_plot": 1,
      "bar_chart": 1,
      "consort_diagram": 1
    }
  }
}
```

### Supported figure types

| Value | Description |
|---|---|
| `kaplan_meier` | Kaplan–Meier survival / event-free survival curves |
| `bar_chart` | Vertical or horizontal bar charts |
| `line_graph` | Line graphs over a continuous axis |
| `scatter_plot` | X–Y scatter / bubble plots |
| `forest_plot` | Forest plots (meta-analysis / subgroup analyses) |
| `consort_diagram` | CONSORT / patient-flow diagrams |
| `heatmap` | Heatmaps and correlation matrices |
| `pie_chart` | Pie / donut charts |
| `box_plot` | Box-and-whisker plots |
| `waterfall_plot` | Waterfall / swimmer plots |
| `table` | Figures that are primarily tabular data |
| `other` | Any other figure type |

### Confidence levels

| Value | Meaning |
|---|---|
| `high` | Exact numeric labels were present in the figure |
| `medium` | Values were read from visible grid lines |
| `low` | Values were estimated from approximate visual position |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

The test suite does **not** require a real PDF or an OpenAI API key — all API
calls are mocked.

---

## Project Structure

```
.
├── extractor.py          # Core extraction module + CLI entry point
├── requirements.txt      # Python dependencies
├── README.md
└── tests/
    └── test_extractor.py # Unit & integration tests (mocked API)
```
