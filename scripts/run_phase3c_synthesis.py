"""Generate remediated Phase 3C integrated research synthesis JSON and Markdown reports read-only from approved Phase 3A/3B reports."""
import json
from pathlib import Path

from cei.synthesis.phase3c_synthesis import (
    build_phase3c_synthesis,
    render_markdown_report,
    verify_prior_phase3c_checksums,
    verify_protected_checksums,
    verify_source_report_checksums,
)

ROOT = Path(__file__).resolve().parents[1]
JSON_OUTPUT = ROOT / "data/reports/bdg2/panther/phase3c_integrated_research_synthesis_remediated.json"
MD_OUTPUT = ROOT / "docs/PHASE_3C_INTEGRATED_RESEARCH_SYNTHESIS_REMEDIATED.md"


def main():
    before_datasets = verify_protected_checksums(ROOT)
    before_reports = verify_source_report_checksums(ROOT)
    before_prior_p3c = verify_prior_phase3c_checksums(ROOT)

    synthesis = build_phase3c_synthesis(ROOT)
    md_content = render_markdown_report(synthesis)

    after_datasets = verify_protected_checksums(ROOT)
    after_reports = verify_source_report_checksums(ROOT)
    after_prior_p3c = verify_prior_phase3c_checksums(ROOT)
    if (
        before_datasets != after_datasets
        or before_reports != after_reports
        or before_prior_p3c != after_prior_p3c
    ):
        raise RuntimeError("Checksum changed during Phase 3C remediation synthesis")

    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with JSON_OUTPUT.open("x", encoding="utf-8") as dest:
        dest.write(json.dumps(synthesis, indent=2, sort_keys=True, allow_nan=False) + "\n")

    MD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with MD_OUTPUT.open("x", encoding="utf-8") as dest:
        dest.write(md_content)

    print(f"Wrote {JSON_OUTPUT.relative_to(ROOT)} and {MD_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
