"""Execute Phase 3D reproducibility verification and write final academic report, playbook, and manifest."""
import json
import sys
from pathlib import Path

from cei.reproducibility.phase3d_reproducibility import (
    build_phase3d_manifest,
    render_final_research_report,
    render_reproducibility_playbook,
    verify_all_protected_checksums,
    verify_existing_phase3d_outputs,
    verify_prior_phase3d_checksums,
)
from cei.synthesis.phase3c_synthesis import build_phase3c_synthesis

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_OUTPUT = ROOT / "data/reports/bdg2/panther/phase3d_reproducibility_manifest_remediated.json"
FINAL_REPORT_OUTPUT = ROOT / "docs/PHASE_3D_FINAL_RESEARCH_REPORT_REMEDIATED.md"
PLAYBOOK_OUTPUT = ROOT / "docs/REPRODUCIBILITY_PLAYBOOK_REMEDIATED.md"


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    if args and args != ["--verify-only"]:
        raise ValueError(f"Unsupported arguments: {args}. Use no arguments or --verify-only.")

    before = verify_all_protected_checksums(ROOT)
    before_prior_3d = verify_prior_phase3d_checksums(ROOT)
    if args == ["--verify-only"]:
        manifest_disk = verify_existing_phase3d_outputs(ROOT)
        after = verify_all_protected_checksums(ROOT)
        after_prior_3d = verify_prior_phase3d_checksums(ROOT)
        if before != after or before_prior_3d != after_prior_3d:
            raise RuntimeError("Protected checksums changed during --verify-only check")
        print(
            f"Verified all {manifest_disk['protected_artifact_counts']['total_protected_files']} "
            "protected artifacts, 3 prior Phase 3D outputs, and 3 remediated Phase 3D outputs deterministically."
        )
        return

    manifest = build_phase3d_manifest(ROOT)
    syn = build_phase3c_synthesis(ROOT)
    final_report_md = render_final_research_report(syn, manifest)
    playbook_md = render_reproducibility_playbook(manifest)

    after = verify_all_protected_checksums(ROOT)
    after_prior_3d = verify_prior_phase3d_checksums(ROOT)
    if before != after or before_prior_3d != after_prior_3d:
        raise RuntimeError("Protected checksums changed during Phase 3D generation")

    MANIFEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_OUTPUT.open("x", encoding="utf-8") as dest:
        dest.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")

    FINAL_REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with FINAL_REPORT_OUTPUT.open("x", encoding="utf-8") as dest:
        dest.write(final_report_md)

    PLAYBOOK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PLAYBOOK_OUTPUT.open("x", encoding="utf-8") as dest:
        dest.write(playbook_md)

    print(
        f"Wrote {MANIFEST_OUTPUT.relative_to(ROOT)}, "
        f"{FINAL_REPORT_OUTPUT.relative_to(ROOT)}, and "
        f"{PLAYBOOK_OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
