"""
Clinical Trial Figure & Graph Data Extractor
=============================================
Extracts figures, graphs, and their underlying data points from clinical trial
research paper PDFs, and returns the results as a structured JSON document.

Usage (CLI):
    python extractor.py path/to/paper.pdf [--output results.json] [--model gpt-4o]

Usage (Python API):
    from extractor import ClinicalTrialFigureExtractor

    extractor = ClinicalTrialFigureExtractor(api_key="sk-...")
    result = extractor.extract_from_pdf("paper.pdf")
    print(result)           # dict
    extractor.save_json("paper.pdf", "results.json")
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# JSON output schema
# ---------------------------------------------------------------------------
#
# {
#   "source_file": "paper.pdf",
#   "extraction_timestamp": "2024-01-01T00:00:00Z",
#   "figures": [
#     {
#       "figure_id": "fig_1",
#       "page_number": 1,
#       "figure_type": "kaplan_meier | bar_chart | line_graph | scatter_plot |
#                        forest_plot | consort_diagram | heatmap | table | other",
#       "caption": "Figure 1. ...",
#       "title": "...",
#       "axes": {
#         "x_axis": {"label": "...", "unit": "...", "range": [min, max], "scale": "linear|log"},
#         "y_axis": {"label": "...", "unit": "...", "range": [min, max], "scale": "linear|log"}
#       },
#       "data_series": [
#         {
#           "name": "...",
#           "color": "...",
#           "line_style": "solid|dashed|dotted",
#           "data_points": [{"x": ..., "y": ..., "ci_lower": ..., "ci_upper": ...}],
#           "summary_statistics": {...}
#         }
#       ],
#       "statistical_annotations": {"p_value": "...", "hazard_ratio": "...", ...},
#       "at_risk_table": {"time_points": [...], "series": {...}},
#       "extraction_confidence": "high|medium|low",
#       "notes": "..."
#     }
#   ],
#   "extraction_summary": {
#     "total_figures": 5,
#     "figure_types": {"kaplan_meier": 2, "bar_chart": 1}
#   }
# }

# ---------------------------------------------------------------------------
# Prompt template sent to the vision model for each figure
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are an expert biomedical data analyst specialising in clinical trial publications.
Analyse the provided figure / graph image and extract ALL data it contains.
Return ONLY a valid JSON object (no markdown fences, no extra text) conforming to the schema below.

Schema:
{
  "figure_type": "<one of: kaplan_meier, bar_chart, line_graph, scatter_plot, forest_plot, consort_diagram, heatmap, pie_chart, box_plot, waterfall_plot, table, other>",
  "title": "<figure title if visible, else null>",
  "axes": {
    "x_axis": {"label": "<label or null>", "unit": "<unit or null>", "range": [<min>, <max>], "scale": "<linear|log>"},
    "y_axis": {"label": "<label or null>", "unit": "<unit or null>", "range": [<min>, <max>], "scale": "<linear|log>"}
  },
  "data_series": [
    {
      "name": "<series name or null>",
      "color": "<color description or null>",
      "line_style": "<solid|dashed|dotted|null>",
      "data_points": [
        {
          "x": <numeric or string value>,
          "y": <numeric value>,
          "ci_lower": <lower confidence bound or null>,
          "ci_upper": <upper confidence bound or null>,
          "error_bar": <error bar value or null>,
          "label": "<point label or null>"
        }
      ],
      "summary_statistics": {
        "mean": <value or null>,
        "median": <value or null>,
        "standard_deviation": <value or null>,
        "hazard_ratio": <value or null>,
        "odds_ratio": <value or null>,
        "relative_risk": <value or null>
      }
    }
  ],
  "statistical_annotations": {
    "p_value": "<value or null>",
    "hazard_ratio": "<value with CI or null>",
    "odds_ratio": "<value with CI or null>",
    "confidence_interval": "<value or null>",
    "log_rank": "<value or null>",
    "sample_size": "<value or null>",
    "other": {}
  },
  "at_risk_table": {
    "time_points": [],
    "series": {}
  },
  "forest_plot_rows": [
    {
      "subgroup": "<name>",
      "n_treatment": <value or null>,
      "n_control": <value or null>,
      "effect_size": <value or null>,
      "ci_lower": <value or null>,
      "ci_upper": <value or null>,
      "weight": <value or null>
    }
  ],
  "consort_flow": {
    "nodes": [{"id": "<id>", "label": "<text>", "count": <value or null>}],
    "edges": [{"from": "<id>", "to": "<id>", "label": "<text or null>"}]
  },
  "heatmap_data": {
    "row_labels": [],
    "col_labels": [],
    "values": []
  },
  "table_data": {
    "headers": [],
    "rows": []
  },
  "extraction_confidence": "<high|medium|low>",
  "notes": "<any additional observations about the figure or caveats about the extraction>"
}

Rules:
- Populate only the fields relevant to the figure type; set irrelevant fields to null or empty.
- For Kaplan–Meier curves, populate axes, data_series (one per arm), statistical_annotations, and at_risk_table.
- For forest plots, populate axes, data_series (overall effect), and forest_plot_rows.
- For CONSORT / flow diagrams, populate consort_flow.
- For heatmaps, populate heatmap_data.
- For tables embedded as images, populate table_data.
- Extract every visible data point; estimate values from the chart when exact values are not labelled.
- Use null (not the string "null") for unknown numeric values.
- Confidence: high = exact values labelled; medium = read from gridlines; low = estimated.
{caption_section}
"""


def _caption_section(caption: str | None) -> str:
    if caption:
        return f"\nThe figure caption (may aid interpretation): {caption}"
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_to_b64(image_bytes: bytes) -> str:
    """Return base-64 encoded PNG string from raw image bytes."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _bytes_to_png(image_bytes: bytes) -> bytes:
    """Ensure image bytes are in PNG format (convert if necessary)."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()


def _find_captions(text: str) -> dict[str, str]:
    """
    Scan page text for figure captions like 'Figure 1.', 'Fig. 2 –', etc.
    Returns a mapping of lowercased figure number string to the full caption sentence.
    """
    pattern = re.compile(
        r"(Fig(?:ure)?\.?\s*\d+[a-zA-Z]?)[\.\-–—\s]+(.*?)(?=Fig(?:ure)?\.?\s*\d|$)",
        re.IGNORECASE | re.DOTALL,
    )
    captions: dict[str, str] = {}
    for match in pattern.finditer(text):
        key = re.sub(r"\s+", "", match.group(1).lower())  # e.g. "figure1", "fig2a"
        caption_text = " ".join(match.group(2).split())
        captions[key] = f"{match.group(1).strip()}. {caption_text}"
    return captions


def _render_page_as_png(page: fitz.Page, dpi: int = 150) -> bytes:
    """Render a PDF page to a PNG image at the specified DPI."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------


class ClinicalTrialFigureExtractor:
    """
    Extract figures / graphs from a clinical trial PDF and return structured JSON.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key.  If *None* the value of the ``OPENAI_API_KEY``
        environment variable (or ``.env`` file) is used.
    model : str
        OpenAI vision-capable model to use (default ``"gpt-4o"``).
    render_dpi : int
        Resolution used when rendering PDF pages to images (default ``150``).
    include_page_images : bool
        When *True*, base-64 encoded page images are included in the output
        under ``"page_image_b64"`` for each figure.  This increases output
        size significantly.
    min_image_size : int
        Minimum width *or* height (pixels) for an extracted image to be
        considered a figure (filters out logos, icons, etc.).  Default 100.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        render_dpi: int = 150,
        include_page_images: bool = False,
        min_image_size: int = 100,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An OpenAI API key is required.  Pass api_key= or set the "
                "OPENAI_API_KEY environment variable (or add it to a .env file)."
            )
        self._client = OpenAI(api_key=resolved_key)
        self._model = model
        self._render_dpi = render_dpi
        self._include_page_images = include_page_images
        self._min_image_size = min_image_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_pdf(self, pdf_path: str | Path) -> dict[str, Any]:
        """
        Extract all figures and their data from *pdf_path*.

        Returns a dictionary conforming to the documented JSON schema.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Opening %s", pdf_path.name)
        doc = fitz.open(str(pdf_path))

        figures: list[dict[str, Any]] = []
        figure_counter = 0

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            page_text = page.get_text("text")
            captions = _find_captions(page_text)

            logger.info("Processing page %d/%d", page_number, len(doc))

            # Extract embedded raster images from the page
            page_images = self._extract_page_images(doc, page, page_number)

            # If no embedded images found, fall back to rendering the whole
            # page and letting the model decide whether it contains a figure.
            if not page_images:
                rendered_png = _render_page_as_png(page, dpi=self._render_dpi)
                if self._page_likely_contains_figure(page_text):
                    page_images = [
                        {
                            "image_bytes": rendered_png,
                            "source": "page_render",
                            "bbox": None,
                        }
                    ]

            for img_info in page_images:
                figure_counter += 1
                fig_id = f"fig_{figure_counter}"

                # Try to match a caption from the page text
                caption = self._match_caption(captions, figure_counter)

                logger.info(
                    "  Analysing figure %s (source=%s)", fig_id, img_info["source"]
                )
                ai_data = self._analyse_figure_with_ai(
                    img_info["image_bytes"], caption
                )

                entry: dict[str, Any] = {
                    "figure_id": fig_id,
                    "page_number": page_number,
                    "caption": caption,
                }
                entry.update(ai_data)

                if self._include_page_images:
                    entry["page_image_b64"] = _image_to_b64(
                        _render_page_as_png(page, dpi=self._render_dpi)
                    )

                figures.append(entry)

        doc.close()

        # Build extraction summary
        type_counts: dict[str, int] = {}
        for fig in figures:
            ftype = fig.get("figure_type", "other") or "other"
            type_counts[ftype] = type_counts.get(ftype, 0) + 1

        result: dict[str, Any] = {
            "source_file": pdf_path.name,
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "figures": figures,
            "extraction_summary": {
                "total_figures": len(figures),
                "figure_types": type_counts,
            },
        }

        return result

    def save_json(
        self,
        pdf_path: str | Path,
        output_path: str | Path | None = None,
        indent: int = 2,
    ) -> Path:
        """
        Extract data from *pdf_path* and write the result to *output_path*.

        If *output_path* is *None* a file with the same stem as the PDF and
        a ``.json`` extension is created in the same directory.

        Returns the path of the written file.
        """
        pdf_path = Path(pdf_path)
        if output_path is None:
            output_path = pdf_path.with_suffix(".json")
        output_path = Path(output_path)

        data = self.extract_from_pdf(pdf_path)
        output_path.write_text(json.dumps(data, indent=indent, ensure_ascii=False))
        logger.info("Results written to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_page_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_number: int,
    ) -> list[dict[str, Any]]:
        """Return a list of image dicts from embedded images on *page*."""
        results: list[dict[str, Any]] = []
        image_list = page.get_images(full=True)
        seen_xrefs: set[int] = set()

        for img_item in image_list:
            xref = img_item[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
            except Exception as exc:
                logger.debug("Could not extract image xref=%d: %s", xref, exc)
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            if width < self._min_image_size or height < self._min_image_size:
                continue

            img_bytes = base_image["image"]
            # Normalise to PNG for consistent API consumption
            try:
                img_bytes = _bytes_to_png(img_bytes)
            except Exception as exc:
                logger.debug("Image conversion failed for xref=%d: %s", xref, exc)
                continue

            # Get bounding box on page for reference
            bbox = None
            for item in page.get_image_rects(xref):
                bbox = list(item)
                break

            results.append(
                {
                    "image_bytes": img_bytes,
                    "source": "embedded",
                    "xref": xref,
                    "width": width,
                    "height": height,
                    "bbox": bbox,
                }
            )

        return results

    @staticmethod
    def _page_likely_contains_figure(page_text: str) -> bool:
        """Heuristic: return True if the page text suggests a figure is present."""
        keywords = re.compile(
            r"\b(fig(ure)?\.?\s*\d|graph|chart|plot|diagram|survival|curve|kaplan|forest)\b",
            re.IGNORECASE,
        )
        return bool(keywords.search(page_text))

    @staticmethod
    def _match_caption(
        captions: dict[str, str], figure_index: int
    ) -> str | None:
        """Try to find a caption matching the given figure index."""
        for key in (f"figure{figure_index}", f"fig{figure_index}", f"fig.{figure_index}"):
            if key in captions:
                return captions[key]
        return None

    def _analyse_figure_with_ai(
        self,
        image_bytes: bytes,
        caption: str | None = None,
    ) -> dict[str, Any]:
        """
        Send *image_bytes* to the vision model and parse the returned JSON.

        Returns a dict with the extracted figure metadata and data.
        """
        prompt = _EXTRACTION_PROMPT.replace(
            "{caption_section}", _caption_section(caption)
        )
        b64_image = _image_to_b64(image_bytes)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=4096,
                temperature=0,
            )
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            return {
                "figure_type": "unknown",
                "extraction_confidence": "low",
                "notes": f"API error: {exc}",
            }

        raw_text = response.choices[0].message.content or ""
        return self._parse_ai_response(raw_text)

    @staticmethod
    def _parse_ai_response(raw: str) -> dict[str, Any]:
        """
        Parse the model response into a Python dict.

        Handles the common case where the model wraps JSON in markdown fences.
        """
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse AI response as JSON; storing raw text.")
            return {
                "figure_type": "unknown",
                "raw_response": raw,
                "extraction_confidence": "low",
                "notes": "AI response could not be parsed as JSON.",
            }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract figures and graph data from a clinical trial PDF and "
            "write the results as structured JSON."
        )
    )
    parser.add_argument("pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Path for the output JSON file.  Defaults to the PDF path with "
            "a .json extension."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use (must support vision).  Default: gpt-4o.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "OpenAI API key.  Defaults to the OPENAI_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution (DPI) for page rendering.  Default: 150.",
    )
    parser.add_argument(
        "--include-page-images",
        action="store_true",
        help="Embed base-64 page images in the output JSON.",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=100,
        help=(
            "Minimum width or height (pixels) for an image to be treated as "
            "a figure.  Default: 100."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level.  Default: 2.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    extractor = ClinicalTrialFigureExtractor(
        api_key=args.api_key,
        model=args.model,
        render_dpi=args.dpi,
        include_page_images=args.include_page_images,
        min_image_size=args.min_image_size,
    )

    output_path = extractor.save_json(
        pdf_path=args.pdf,
        output_path=args.output,
        indent=args.indent,
    )
    print(f"Extraction complete. Results written to: {output_path}")


if __name__ == "__main__":
    main()
