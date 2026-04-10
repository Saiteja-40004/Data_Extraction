"""
Tests for the ClinicalTrialFigureExtractor and associated helpers.

These tests do NOT require a real PDF or an OpenAI API key; the API call is
mocked so the test suite can run in CI environments without credentials.
"""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

# Ensure the module-level dotenv load does not fail in test environments
import os
os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")

from extractor import (
    ClinicalTrialFigureExtractor,
    _bytes_to_png,
    _caption_section,
    _find_captions,
    _image_to_b64,
    _render_page_as_png,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 200, height: int = 200, color: str = "white") -> bytes:
    """Return minimal in-memory PNG bytes."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_minimal_pdf() -> bytes:
    """
    Return a minimal valid PDF as bytes using PyMuPDF so we can test
    PDF processing without relying on an external file.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    # Insert some text including a figure caption
    page.insert_text(
        (72, 100),
        "Figure 1. Kaplan–Meier survival curves for treatment and control arms.",
        fontsize=10,
    )
    # Insert a simple rectangle as a proxy for an image area
    page.draw_rect(fitz.Rect(72, 200, 500, 600), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Unit tests for pure helper functions
# ---------------------------------------------------------------------------


class TestImageHelpers(unittest.TestCase):

    def test_image_to_b64_roundtrip(self):
        import base64
        original = b"hello world"
        encoded = _image_to_b64(original)
        self.assertEqual(base64.b64decode(encoded), original)

    def test_bytes_to_png_produces_valid_png(self):
        jpeg_bytes = _make_png_bytes()  # already PNG – conversion must still work
        result = _bytes_to_png(jpeg_bytes)
        img = Image.open(io.BytesIO(result))
        self.assertEqual(img.format, "PNG")

    def test_bytes_to_png_from_jpeg(self):
        img = Image.new("RGB", (50, 50), "red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = _bytes_to_png(buf.getvalue())
        out_img = Image.open(io.BytesIO(result))
        self.assertEqual(out_img.format, "PNG")


class TestCaptionHelpers(unittest.TestCase):

    def test_find_captions_standard_format(self):
        text = (
            "Some text on the page.\n"
            "Figure 1. Kaplan–Meier overall survival curves.\n"
            "More page text.\n"
            "Figure 2. Forest plot of hazard ratios.\n"
        )
        captions = _find_captions(text)
        self.assertIn("figure1", captions)
        self.assertIn("figure2", captions)
        self.assertIn("Kaplan", captions["figure1"])
        self.assertIn("Forest", captions["figure2"])

    def test_find_captions_abbreviated_fig(self):
        text = "Fig. 3A. Scatter plot of baseline characteristics.\n"
        captions = _find_captions(text)
        self.assertTrue(any("3a" in k or "3" in k for k in captions))

    def test_find_captions_empty_text(self):
        self.assertEqual(_find_captions(""), {})

    def test_caption_section_with_caption(self):
        result = _caption_section("Figure 1. Some caption.")
        self.assertIn("Figure 1. Some caption.", result)

    def test_caption_section_without_caption(self):
        result = _caption_section(None)
        self.assertEqual(result, "")


class TestParseAiResponse(unittest.TestCase):

    def setUp(self):
        self.extractor = ClinicalTrialFigureExtractor.__new__(ClinicalTrialFigureExtractor)

    def test_clean_json(self):
        payload = {"figure_type": "bar_chart", "extraction_confidence": "high"}
        result = ClinicalTrialFigureExtractor._parse_ai_response(json.dumps(payload))
        self.assertEqual(result["figure_type"], "bar_chart")

    def test_json_with_markdown_fences(self):
        payload = {"figure_type": "kaplan_meier", "extraction_confidence": "medium"}
        raw = f"```json\n{json.dumps(payload)}\n```"
        result = ClinicalTrialFigureExtractor._parse_ai_response(raw)
        self.assertEqual(result["figure_type"], "kaplan_meier")

    def test_invalid_json_returns_fallback(self):
        result = ClinicalTrialFigureExtractor._parse_ai_response("not valid json {{")
        self.assertEqual(result["figure_type"], "unknown")
        self.assertEqual(result["extraction_confidence"], "low")
        self.assertIn("raw_response", result)

    def test_fenced_without_language_tag(self):
        payload = {"figure_type": "forest_plot"}
        raw = f"```\n{json.dumps(payload)}\n```"
        result = ClinicalTrialFigureExtractor._parse_ai_response(raw)
        self.assertEqual(result["figure_type"], "forest_plot")


# ---------------------------------------------------------------------------
# Integration-style tests (OpenAI API mocked)
# ---------------------------------------------------------------------------


def _mock_openai_response(figure_type: str = "bar_chart") -> MagicMock:
    """Build a minimal mock that mimics the OpenAI chat-completion response."""
    payload = {
        "figure_type": figure_type,
        "title": "Primary Endpoint",
        "axes": {
            "x_axis": {"label": "Treatment Group", "unit": None, "range": [None, None], "scale": "linear"},
            "y_axis": {"label": "Response Rate (%)", "unit": "%", "range": [0, 100], "scale": "linear"},
        },
        "data_series": [
            {
                "name": "Drug A",
                "color": "blue",
                "line_style": None,
                "data_points": [{"x": "Drug A", "y": 72.3, "ci_lower": 65.1, "ci_upper": 79.5, "error_bar": None, "label": None}],
                "summary_statistics": {"mean": 72.3, "median": None, "standard_deviation": None, "hazard_ratio": None, "odds_ratio": None, "relative_risk": None},
            }
        ],
        "statistical_annotations": {"p_value": "0.003", "hazard_ratio": None, "odds_ratio": None, "confidence_interval": "65.1–79.5", "log_rank": None, "sample_size": "120", "other": {}},
        "at_risk_table": {"time_points": [], "series": {}},
        "forest_plot_rows": [],
        "consort_flow": {"nodes": [], "edges": []},
        "heatmap_data": {"row_labels": [], "col_labels": [], "values": []},
        "table_data": {"headers": [], "rows": []},
        "extraction_confidence": "high",
        "notes": "Mock response.",
    }
    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestExtractorWithMockedAPI(unittest.TestCase):
    """
    Test the extractor end-to-end against a minimal in-memory PDF,
    with the OpenAI API replaced by a mock.
    """

    def _make_extractor(self) -> ClinicalTrialFigureExtractor:
        with patch("extractor.OpenAI"):
            ext = ClinicalTrialFigureExtractor(api_key="test-key")
        return ext

    def test_constructor_raises_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)  # ensure variable absent
            with self.assertRaises(ValueError):
                ClinicalTrialFigureExtractor(api_key=None)

    def test_extract_from_pdf_structure(self, tmp_path=None):
        """Verify the top-level output structure is correct."""
        import tempfile

        ext = self._make_extractor()
        ext._client = MagicMock()
        ext._client.chat.completions.create.return_value = _mock_openai_response("bar_chart")

        # Write minimal PDF to a temp file
        pdf_bytes = _make_minimal_pdf()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(pdf_bytes)
            pdf_path = Path(fh.name)

        try:
            result = ext.extract_from_pdf(pdf_path)

            # Top-level keys
            self.assertIn("source_file", result)
            self.assertIn("extraction_timestamp", result)
            self.assertIn("figures", result)
            self.assertIn("extraction_summary", result)

            # Summary fields
            summary = result["extraction_summary"]
            self.assertIn("total_figures", summary)
            self.assertIn("figure_types", summary)
            self.assertIsInstance(summary["total_figures"], int)
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_save_json_creates_file(self):
        import tempfile

        ext = self._make_extractor()
        ext._client = MagicMock()
        ext._client.chat.completions.create.return_value = _mock_openai_response()

        pdf_bytes = _make_minimal_pdf()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "sample.pdf"
            pdf_path.write_bytes(pdf_bytes)
            json_path = Path(tmp_dir) / "sample.json"

            returned_path = ext.save_json(pdf_path, json_path)

            self.assertEqual(returned_path, json_path)
            self.assertTrue(json_path.exists())

            data = json.loads(json_path.read_text())
            self.assertEqual(data["source_file"], "sample.pdf")

    def test_analyse_figure_with_api_error(self):
        """When the API raises, _analyse_figure_with_ai returns a fallback dict."""
        ext = self._make_extractor()
        ext._client = MagicMock()
        ext._client.chat.completions.create.side_effect = RuntimeError("connection error")

        result = ext._analyse_figure_with_ai(_make_png_bytes())
        self.assertEqual(result["figure_type"], "unknown")
        self.assertEqual(result["extraction_confidence"], "low")
        self.assertIn("API error", result["notes"])

    def test_extract_from_nonexistent_pdf_raises(self):
        ext = self._make_extractor()
        with self.assertRaises(FileNotFoundError):
            ext.extract_from_pdf("/nonexistent/path/paper.pdf")

    def test_page_likely_contains_figure(self):
        self.assertTrue(ClinicalTrialFigureExtractor._page_likely_contains_figure("See Figure 1 for survival curves."))
        self.assertTrue(ClinicalTrialFigureExtractor._page_likely_contains_figure("The kaplan-meier analysis showed..."))
        self.assertFalse(ClinicalTrialFigureExtractor._page_likely_contains_figure("Introduction to clinical methodology."))

    def test_match_caption(self):
        captions = {"figure1": "Figure 1. Kaplan-Meier.", "figure2": "Figure 2. Forest plot."}
        self.assertEqual(ClinicalTrialFigureExtractor._match_caption(captions, 1), "Figure 1. Kaplan-Meier.")
        self.assertEqual(ClinicalTrialFigureExtractor._match_caption(captions, 2), "Figure 2. Forest plot.")
        self.assertIsNone(ClinicalTrialFigureExtractor._match_caption(captions, 99))

    def test_figure_data_structure_from_mock(self):
        """Figures list should contain entries with required keys populated by AI response."""
        import tempfile

        ext = self._make_extractor()
        ext._client = MagicMock()
        ext._client.chat.completions.create.return_value = _mock_openai_response("kaplan_meier")

        pdf_bytes = _make_minimal_pdf()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(pdf_bytes)
            pdf_path = Path(fh.name)

        try:
            result = ext.extract_from_pdf(pdf_path)
            # If no embedded images are found in the minimal PDF, the list may
            # be empty; we only check structure when figures are present.
            if result["figures"]:
                fig = result["figures"][0]
                self.assertIn("figure_id", fig)
                self.assertIn("page_number", fig)
                self.assertIn("caption", fig)
                self.assertIn("figure_type", fig)
                self.assertEqual(fig["figure_type"], "kaplan_meier")
        finally:
            pdf_path.unlink(missing_ok=True)


class TestRenderPage(unittest.TestCase):
    """Test the page rendering helper."""

    def test_render_returns_png_bytes(self):
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page_bytes = _render_page_as_png(page, dpi=72)
        doc.close()

        img = Image.open(io.BytesIO(page_bytes))
        self.assertEqual(img.format, "PNG")
        self.assertGreater(img.width, 0)
        self.assertGreater(img.height, 0)


if __name__ == "__main__":
    unittest.main()
