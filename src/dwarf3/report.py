"""
Report generation for dwarf3 pipeline.

Produces:
- run_manifest.json: Machine-readable complete record
- report.json: Summary statistics
- report.md: Human-readable Markdown report

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import RejectedFrame, RejectionReason, StackConfig, StackResult
from .utils import get_platform_info, get_timestamp_iso, get_version

logger = logging.getLogger(__name__)


def _to_native(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def _serialize_config(config: StackConfig) -> dict[str, Any]:
    """Serialize StackConfig to JSON-compatible dict."""
    return {
        "keep_fraction": config.keep_fraction,
        "exclude_failed_prefix": config.exclude_failed_prefix,
        "exclude_stacked_prefix": config.exclude_stacked_prefix,
        "saturation_threshold": config.saturation_threshold,
        "reference": config.reference,
        "backend_align": config.backend_align,
        "max_align_fail": config.max_align_fail,
        "downsample_for_align": config.downsample_for_align,
        "sigma": config.sigma,
        "maxiters": config.maxiters,
        "debayer": config.debayer,
        "cache_aligned": config.cache_aligned,
        "write_quicklook": config.write_quicklook,
        "quicklook_percentiles": list(config.quicklook_percentiles),
        "asinh_a": config.asinh_a,
    }


def _serialize_rejected(rejected: list[RejectedFrame]) -> list[dict[str, str]]:
    """Serialize rejected frames list."""
    return [
        {
            "path": r.path,
            "reason": r.reason.value,
            "detail": r.detail,
        }
        for r in rejected
    ]


def write_manifest(result: StackResult, output_dir: Path) -> Path:
    """
    Write complete run manifest as JSON.

    Parameters
    ----------
    result : StackResult
        Complete stacking result.
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to written manifest file.
    """
    manifest = {
        "dwarf3_version": result.version or get_version(),
        "timestamp": result.timestamp or get_timestamp_iso(),
        "platform": result.platform or get_platform_info(),
        "session": {
            "id": result.session_id,
            "path": result.session_path,
        },
        "config": _serialize_config(result.config) if result.config else {},
        "frames": {
            "discovered": len(result.inputs),
            "kept": len(result.kept),
            "rejected": len(result.rejected),
            "alignment_failures": len(result.alignment_failures),
        },
        "inputs": result.inputs,
        "kept": result.kept,
        "rejected": _serialize_rejected(result.rejected),
        "alignment_failures": result.alignment_failures,
        "reference_frame": result.reference_frame,
        "outputs": result.outputs,
        "statistics": result.stats,
        "scores": [
            {
                "path": s.path,
                "background_median": s.background_median,
                "background_mad": s.background_mad,
                "noise_proxy": s.noise_proxy,
                "saturation_fraction": s.saturation_fraction,
                "composite_score": s.composite_score,
            }
            for s in result.scores
        ],
    }

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(_to_native(manifest), f, indent=2)

    logger.info("Wrote manifest: %s", manifest_path)
    return manifest_path


def write_report_json(result: StackResult, output_dir: Path) -> Path:
    """
    Write summary report as JSON.

    Parameters
    ----------
    result : StackResult
        Stacking result.
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to written report file.
    """
    report = {
        "session_id": result.session_id,
        "timestamp": result.timestamp or get_timestamp_iso(),
        "summary": {
            "frames_discovered": len(result.inputs),
            "frames_kept": len(result.kept),
            "frames_rejected": len(result.rejected),
            "alignment_failures": len(result.alignment_failures),
            "reference_frame": Path(result.reference_frame).name if result.reference_frame else "",
        },
        "rejection_reasons": _count_rejection_reasons(result.rejected),
        "statistics": result.stats,
        "outputs": {k: str(v) for k, v in result.outputs.items()},
    }

    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(_to_native(report), f, indent=2)

    logger.info("Wrote report: %s", report_path)
    return report_path


def _count_rejection_reasons(rejected: list[RejectedFrame]) -> dict[str, int]:
    """Count frames by rejection reason."""
    counts: dict[str, int] = {}
    for r in rejected:
        reason = r.reason.value
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def write_report_markdown(result: StackResult, output_dir: Path) -> Path:
    """
    Write human-readable Markdown report.

    Parameters
    ----------
    result : StackResult
        Stacking result.
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to written report file.
    """
    lines = [
        f"# Stacking Report: {result.session_id}",
        "",
        f"**Generated:** {result.timestamp or get_timestamp_iso()}",
        f"**dwarf3 version:** {result.version or get_version()}",
        f"**Platform:** {result.platform or get_platform_info()}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Frames discovered | {len(result.inputs)} |",
        f"| Frames kept | {len(result.kept)} |",
        f"| Frames rejected | {len(result.rejected)} |",
        f"| Alignment failures | {len(result.alignment_failures)} |",
        f"| Reference frame | `{Path(result.reference_frame).name if result.reference_frame else 'N/A'}` |",
        "",
    ]

    # Configuration
    if result.config:
        lines.extend([
            "## Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| keep_fraction | {result.config.keep_fraction} |",
            f"| sigma | {result.config.sigma} |",
            f"| maxiters | {result.config.maxiters} |",
            f"| reference | {result.config.reference} |",
            f"| debayer | {result.config.debayer} |",
            "",
        ])

    # Statistics
    if result.stats:
        lines.extend([
            "## Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])
        for key, value in result.stats.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
            else:
                lines.append(f"| {key} | {value} |")
        lines.append("")

    # Rejection breakdown
    rejection_counts = _count_rejection_reasons(result.rejected)
    if rejection_counts:
        lines.extend([
            "## Rejection Breakdown",
            "",
            "| Reason | Count |",
            "|--------|-------|",
        ])
        for reason, count in sorted(rejection_counts.items()):
            lines.append(f"| {reason} | {count} |")
        lines.append("")

    # Outputs
    if result.outputs:
        lines.extend([
            "## Outputs",
            "",
        ])
        for name, path in result.outputs.items():
            lines.append(f"- **{name}:** `{path}`")
        lines.append("")

    # Top/bottom frames by quality
    if result.scores:
        lines.extend([
            "## Quality Scores (Top 5)",
            "",
            "| Rank | Frame | Score |",
            "|------|-------|-------|",
        ])
        for i, score in enumerate(result.scores[:5], 1):
            name = Path(score.path).name
            lines.append(f"| {i} | `{name}` | {score.composite_score:.4f} |")
        lines.append("")

        if len(result.scores) > 5:
            lines.extend([
                "## Quality Scores (Bottom 5)",
                "",
                "| Rank | Frame | Score |",
                "|------|-------|-------|",
            ])
            for i, score in enumerate(result.scores[-5:], len(result.scores) - 4):
                name = Path(score.path).name
                lines.append(f"| {i} | `{name}` | {score.composite_score:.4f} |")
            lines.append("")

    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Wrote Markdown report: %s", report_path)
    return report_path


def write_all_reports(result: StackResult, output_dir: Path) -> dict[str, Path]:
    """
    Write all report files.

    Parameters
    ----------
    result : StackResult
        Stacking result.
    output_dir : Path
        Output directory.

    Returns
    -------
    dict[str, Path]
        Map of report type to path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "manifest": write_manifest(result, output_dir),
        "json": write_report_json(result, output_dir),
        "markdown": write_report_markdown(result, output_dir),
    }
