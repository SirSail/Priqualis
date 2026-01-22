"""Report Generator for Priqualis."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from priqualis.rules.models import ValidationReport

logger = logging.getLogger(__name__)

@dataclass(slots=True, frozen=True)
class ReportConfig:
    title: str = "Priqualis Validation Report"
    include_violations_detail: bool = True
    include_recommendations: bool = True
    max_violations_shown: int = 50
    language: Literal["pl", "en"] = "en"

DEFAULT_CONFIG = ReportConfig()

class ReportGenerator:
    """Generates validation reports (MD, PDF, JSON)."""

    def __init__(self, config: ReportConfig | None = None):
        self.config = config or DEFAULT_CONFIG

    def generate_markdown(self, report: ValidationReport, batch_id: str | None = None) -> str:
        batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        pass_rate = report.pass_rate * 100
        
        # Summary
        lines = [
            f"# {self.config.title}",
            f"\n**Batch:** `{batch_id}` | **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            "## 📊 Summary",
            f"**Total:** {report.total_records:,} | **Passed:** {report.total_records - report.violation_count:,} ({pass_rate:.1f}%) | **Violations:** {report.violation_count:,} | **Warnings:** {report.warning_count:,}\n"
        ]

        # Top Rules
        from collections import Counter
        top_rules = Counter(v.rule_id for v in report.violations).most_common(5)
        if top_rules:
            lines.extend(["## 🔝 Top Violations", "| Rule | Count | % |", "|---|---|---|"])
            for r, c in top_rules:
                pct = c / report.violation_count * 100 if report.violation_count else 0
                lines.append(f"| `{r}` | {c:,} | {pct:.1f}% |")
            lines.append("")

        # Violations Detail
        if self.config.include_violations_detail and report.violations:
            lines.append("## ❌ Detail")
            for v in report.violations[:self.config.max_violations_shown]:
                lines.append(f"- **{v.case_id}** (`{v.rule_id}`): {v.message or ''} [{v.state}]")
            if len(report.violations) > self.config.max_violations_shown:
                lines.append(f"\n*(...and {len(report.violations) - self.config.max_violations_shown} more)*")

        # Recommendations
        if self.config.include_recommendations:
            recs = []
            if report.pass_rate < 0.9: recs.append(f"High violation rate ({100-pass_rate:.1f}%). Review data entry.")
            if top_rules and (top_rules[0][1] / report.violation_count > 0.3): recs.append(f"Rule `{top_rules[0][0]}` is dominant.")
            if recs:
                lines.extend(["\n## 💡 Recommendations"] + [f"- {r}" for r in recs])

        return "\n".join(lines)

    def generate_json(self, report: ValidationReport, batch_id: str | None = None) -> dict[str, Any]:
        from collections import Counter
        return {
            "batch_id": batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "summary": {"total": report.total_records, "violations": report.violation_count, "pass_rate": report.pass_rate},
            "violations_by_rule": dict(Counter(v.rule_id for v in report.violations)),
            "details": [{"case": v.case_id, "rule": v.rule_id, "msg": v.message} for v in report.violations[:self.config.max_violations_shown]]
        }

    def generate_pdf(self, report: ValidationReport, output_path: Path, batch_id: str | None = None) -> Path:
        import markdown
        from weasyprint import HTML
        
        md = self.generate_markdown(report, batch_id)
        html = f"""
        <html><head><style>
            body {{ font-family: sans-serif; max-width: 800px; margin: 20px auto; }}
            h1 {{ border-bottom: 2px solid #1a5f7a; color: #1a5f7a; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #1a5f7a; color: white; }}
            code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        </style></head><body>{markdown.markdown(md, extensions=['tables'])}</body></html>
        """
        HTML(string=html).write_pdf(target=output_path)
        return Path(output_path)

def generate_batch_report(report: ValidationReport, output_dir: Path | str, batch_id: str | None = None, formats: list[str] | None = None, config: ReportConfig | None = None) -> dict[str, Path]:
    out, gen = Path(output_dir), ReportGenerator(config)
    out.mkdir(parents=True, exist_ok=True)
    formats, results, bid = formats or ["markdown", "json"], {}, batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if "markdown" in formats:
        (p := out / f"{bid}.md").write_text(gen.generate_markdown(report, bid), "utf-8"); results["markdown"] = p
    if "json" in formats:
        import json; (p := out / f"{bid}.json").write_text(json.dumps(gen.generate_json(report, bid), indent=2), "utf-8"); results["json"] = p
    if "pdf" in formats:
        results["pdf"] = gen.generate_pdf(report, out / f"{bid}.pdf", bid)
    
    return results
