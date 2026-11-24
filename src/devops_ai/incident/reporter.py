# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Rich terminal and document post-mortem reporting for operational incidents."""

import json
from pathlib import Path
import re
from typing import List, Optional, Union

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from devops_ai.core.config import get_settings
from devops_ai.core.exceptions import DevOpsAIError
from devops_ai.core.file_io import atomic_write_text
from devops_ai.incident.models import (
    Incident,
    IncidentStatus,
    RiskLevel,
    RootCauseHypothesis,
    Severity,
)


def sanitize_text(text: str) -> str:
    """Best-effort redaction of API keys and credentials before export or display."""
    if not text:
        return ""
    try:
        settings = get_settings()
        sanitized = text
        if settings.google_api_key:
            raw = settings.google_api_key.get_secret_value()
            if raw and raw in sanitized:
                sanitized = sanitized.replace(raw, "[REDACTED_API_KEY]")
        sanitized = re.sub(r"AIza[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", sanitized)
        return sanitized
    except Exception:
        return re.sub(r"AIza[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", text)


def _get_severity_color(severity: Union[Severity, str]) -> str:
    """Map incident and alert severity to Rich markup color."""
    val = severity.value if isinstance(severity, Severity) else str(severity).upper()
    mapping = {
        "CRITICAL": "bold red",
        "HIGH": "red",
        "MEDIUM": "yellow",
        "LOW": "blue",
        "INFO": "dim cyan",
    }
    return mapping.get(val, "white")


def _get_status_color(status: Union[IncidentStatus, str]) -> str:
    """Map incident status to Rich markup color."""
    val = status.value if isinstance(status, IncidentStatus) else str(status).upper()
    mapping = {
        "DETECTED": "yellow",
        "CORRELATED": "cyan",
        "INVESTIGATING": "magenta",
        "ANALYZED": "bright_blue",
        "RESOLVED": "green",
        "CLOSED": "dim green",
    }
    return mapping.get(val, "white")


def _get_risk_color(risk: Union[RiskLevel, str]) -> str:
    """Map remediation risk level to Rich markup color."""
    val = risk.value if isinstance(risk, RiskLevel) else str(risk).upper()
    mapping = {
        "SAFE": "bold green",
        "MODERATE": "bold yellow",
        "DANGEROUS": "bold red",
    }
    return mapping.get(val, "white")


class IncidentReporter:
    """Multi-format reporting engine for operational incidents."""

    @classmethod
    def render_terminal(cls, incident: Incident, console: Optional[Console] = None) -> None:
        """Render a formatted, colorized incident investigation view to the terminal."""
        con = console or Console()

        # 1. Header Overview Panel
        sev_color = _get_severity_color(incident.severity)
        stat_color = _get_status_color(incident.status)
        components_str = ", ".join(incident.components_affected) or "None"

        overview_text = Text()
        overview_text.append(f"Incident ID: {incident.incident_id}\n", style="bold cyan")
        overview_text.append("Status: ", style="bold")
        overview_text.append(f"{incident.status.value}\n", style=stat_color)
        overview_text.append("Severity: ", style="bold")
        overview_text.append(f"{incident.severity.value}\n", style=sev_color)
        overview_text.append(f"Category: {incident.category.value}\n", style="white")
        overview_text.append(f"Components Affected: {components_str}\n", style="magenta")
        overview_text.append(f"Created: {incident.created_at.isoformat()}\n", style="dim")
        overview_text.append(f"Updated: {incident.updated_at.isoformat()}\n", style="dim")
        if incident.resolved_at:
            overview_text.append(
                f"Resolved: {incident.resolved_at.isoformat()}\n", style="green dim"
            )
        if incident.closed_at:
            overview_text.append(f"Closed: {incident.closed_at.isoformat()}\n", style="dim")
        if incident.resolution_notes:
            overview_text.append(
                f"\nResolution Notes:\n{sanitize_text(incident.resolution_notes)}\n",
                style="italic green",
            )

        con.print(
            Panel(
                overview_text,
                title=f"[bold]Incident: {escape(sanitize_text(incident.title))}[/bold]",
                border_style="cyan",
            )
        )

        # 2. Primary Root Cause Panel
        if incident.root_cause:
            rc: RootCauseHypothesis = incident.root_cause
            rc_status = (
                "[bold green]CONFIRMED[/bold green]"
                if rc.is_confirmed
                else (
                    "[bold red]UNSUPPORTED[/bold red]"
                    if rc.is_unsupported
                    else "[yellow]UNCONFIRMED[/yellow]"
                )
            )
            tier_badge = (
                "[bold blue]Tier 1 (Deterministic)[/bold blue]"
                if rc.tier == "tier_1_deterministic"
                else "[bold magenta]Tier 2 (AI-Assisted)[/bold magenta]"
            )

            rc_content = (
                f"Title: [bold]{escape(sanitize_text(rc.title))}[/bold]\n"
                f"Engine: {tier_badge} | Status: {rc_status} | Confidence: [bold]{rc.confidence:.0%}[/bold] ({rc.confidence:.2f})\n\n"
                f"Description:\n{escape(sanitize_text(rc.description))}\n"
            )
            if rc.evidence_refs:
                rc_content += f"\nEvidence Citations: {escape(', '.join(rc.evidence_refs))}\n"
            if rc.supporting_evidence:
                rc_content += "\nSupporting Facts:\n"
                for sf in rc.supporting_evidence:
                    rc_content += f"  • {escape(sanitize_text(sf))}\n"
            if rc.contradicting_evidence:
                rc_content += "\nContradicting Evidence:\n"
                for ce in rc.contradicting_evidence:
                    rc_content += f"  • [red]{escape(sanitize_text(ce))}[/red]\n"

            con.print(
                Panel(
                    rc_content.strip(),
                    title="[bold yellow]Primary Root Cause Analysis[/bold yellow]",
                    border_style="yellow",
                )
            )
        else:
            con.print(
                Panel(
                    "[dim]No root cause analysis performed yet for this incident.[/dim]",
                    title="Root Cause Analysis",
                    border_style="dim",
                )
            )

        # 3. Alternative Hypotheses Table
        if incident.alternative_hypotheses:
            alt_table = Table(
                title="Alternative Hypotheses Considered", show_header=True, header_style="bold"
            )
            alt_table.add_column("Title", style="white")
            alt_table.add_column("Tier", style="cyan")
            alt_table.add_column("Confidence", style="bold")
            alt_table.add_column("Status", style="magenta")
            alt_table.add_column("Evidence Refs", style="dim")

            for alt in incident.alternative_hypotheses:
                status_str = (
                    "Confirmed"
                    if alt.is_confirmed
                    else ("Unsupported" if alt.is_unsupported else "Unconfirmed")
                )
                alt_table.add_row(
                    escape(sanitize_text(alt.title)),
                    "Tier 1" if alt.tier == "tier_1_deterministic" else "Tier 2",
                    f"{alt.confidence:.0%}",
                    status_str,
                    escape(", ".join(alt.evidence_refs) if alt.evidence_refs else "None"),
                )
            con.print(alt_table)

        # 4. Remediation Recommendations Table
        if incident.recommendations:
            con.print(
                Panel(
                    "[bold yellow]ADVISORY ONLY:[/bold yellow] All actions must be manually reviewed and executed by a human engineer. "
                    "Automated execution is strictly disabled.",
                    title="[bold red]Remediation Advisory Guard Rails[/bold red]",
                    border_style="red",
                )
            )
            rec_table = Table(
                title="Recommended Action Items", show_header=True, header_style="bold"
            )
            rec_table.add_column("ID", style="dim", width=12)
            rec_table.add_column("Type", style="cyan", width=20)
            rec_table.add_column("Risk", width=12)
            rec_table.add_column("Command / Action", style="white")
            rec_table.add_column("Verification Step", style="green")

            for rec in incident.recommendations:
                risk_style = _get_risk_color(rec.risk_level)
                cmd_display = (
                    f"[bold]{escape(sanitize_text(rec.command))}[/bold]"
                    if rec.command
                    else f"[italic]{escape(sanitize_text(rec.title))}[/italic]"
                )
                rec_table.add_row(
                    escape(rec.action_id),
                    escape(rec.action_type),
                    f"[{risk_style}]{escape(rec.risk_level.value)}[/{risk_style}]",
                    cmd_display,
                    escape(sanitize_text(rec.verification_step)),
                )
            con.print(rec_table)

        # 5. Timeline Table
        if incident.timeline:
            tl_table = Table(title="Incident Timeline", show_header=True, header_style="bold")
            tl_table.add_column("Timestamp (UTC)", style="dim", width=24)
            tl_table.add_column("Source", style="cyan", width=14)
            tl_table.add_column("Component", style="magenta", width=18)
            tl_table.add_column("Severity", width=10)
            tl_table.add_column("Event Description", style="white")

            for entry in incident.timeline:
                s_color = _get_severity_color(entry.severity)
                tl_table.add_row(
                    escape(entry.timestamp.isoformat()),
                    escape(entry.source.value),
                    escape(entry.component),
                    f"[{s_color}]{escape(entry.severity.value)}[/{s_color}]",
                    escape(sanitize_text(entry.event)),
                )
            con.print(tl_table)

        # 6. Correlated Evidence Summary
        catalog = incident.evidence.evidence_catalog
        if catalog:
            evid_table = Table(
                title="Correlated Evidence Catalog", show_header=True, header_style="bold"
            )
            evid_table.add_column("Evidence ID", style="bold cyan", width=18)
            evid_table.add_column("Details", style="white")
            for eid, evalue in sorted(catalog.items()):
                clean_val = sanitize_text(str(evalue))
                if len(clean_val) > 120:
                    clean_val = clean_val[:117] + "..."
                evid_table.add_row(escape(eid), escape(clean_val))
            con.print(evid_table)

    @classmethod
    def render_markdown(cls, incident: Incident) -> str:
        """Render a comprehensive GitHub Flavored Markdown incident post-mortem."""
        lines: List[str] = []

        # Title
        title_sanitized = sanitize_text(incident.title)
        lines.append(f"# Incident Post-Mortem: {incident.incident_id} — {title_sanitized}\n")

        # Executive Summary Table
        lines.append("## Executive Summary\n")
        lines.append("| Attribute | Value |")
        lines.append("| :--- | :--- |")
        lines.append(f"| **Incident ID** | `{incident.incident_id}` |")
        lines.append(f"| **Status** | `{incident.status.value}` |")
        lines.append(f"| **Severity** | `{incident.severity.value}` |")
        lines.append(f"| **Category** | `{incident.category.value}` |")
        lines.append(
            f"| **Components Affected** | `{', '.join(incident.components_affected) or 'None'}` |"
        )
        lines.append(f"| **Created At (UTC)** | `{incident.created_at.isoformat()}` |")
        lines.append(f"| **Updated At (UTC)** | `{incident.updated_at.isoformat()}` |")
        resolved_str = incident.resolved_at.isoformat() if incident.resolved_at else "N/A"
        lines.append(f"| **Resolved At (UTC)** | `{resolved_str}` |")
        closed_str = incident.closed_at.isoformat() if incident.closed_at else "N/A"
        lines.append(f"| **Closed At (UTC)** | `{closed_str}` |")
        lines.append("")

        if incident.resolution_notes:
            lines.append("### Resolution Summary\n")
            lines.append(f"{sanitize_text(incident.resolution_notes)}\n")

        # Primary Root Cause Analysis
        lines.append("## Primary Root Cause Analysis\n")
        if incident.root_cause:
            rc = incident.root_cause
            tier_display = (
                "Tier 1 Deterministic"
                if rc.tier == "tier_1_deterministic"
                else "Tier 2 AI-Assisted"
            )
            status_display = (
                "Confirmed"
                if rc.is_confirmed
                else ("Unsupported Hypothesis" if rc.is_unsupported else "Unconfirmed")
            )
            lines.append(f"### {sanitize_text(rc.title)}\n")
            lines.append(f"- **Investigation Engine:** `{tier_display}`")
            lines.append(
                f"- **Confidence Score:** `{rc.confidence:.0%}` ({rc.confidence:.2f} / 1.00)"
            )
            lines.append(f"- **Verification Status:** `{status_display}`\n")
            lines.append("**Detailed Explanation:**\n")
            lines.append(f"{sanitize_text(rc.description)}\n")

            if rc.evidence_refs:
                lines.append("#### Supporting Evidence Citations\n")
                for ref in rc.evidence_refs:
                    detail = incident.evidence.evidence_catalog.get(ref, "")
                    if detail:
                        lines.append(f"- **`{ref}`**: {sanitize_text(detail)}")
                    else:
                        lines.append(f"- **`{ref}`**")
                lines.append("")

            if rc.supporting_evidence:
                lines.append("#### Corroborating Signals\n")
                for item in rc.supporting_evidence:
                    lines.append(f"- {sanitize_text(item)}")
                lines.append("")

            if rc.contradicting_evidence:
                lines.append("#### Contradicting Signals\n")
                for item in rc.contradicting_evidence:
                    lines.append(f"- ⚠️ {sanitize_text(item)}")
                lines.append("")
        else:
            lines.append("*No root cause analysis performed yet for this incident.*\n")

        # Alternative Hypotheses
        if incident.alternative_hypotheses:
            lines.append("## Alternative Hypotheses Considered\n")
            lines.append("| Hypothesis | Engine | Confidence | Status | Citations |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for alt in incident.alternative_hypotheses:
                engine = "Tier 1" if alt.tier == "tier_1_deterministic" else "Tier 2"
                stat = (
                    "Confirmed"
                    if alt.is_confirmed
                    else ("Unsupported" if alt.is_unsupported else "Unconfirmed")
                )
                citations = ", ".join(alt.evidence_refs) if alt.evidence_refs else "None"
                lines.append(
                    f"| {sanitize_text(alt.title)} | `{engine}` | `{alt.confidence:.0%}` | `{stat}` | `{citations}` |"
                )
            lines.append("")

        # Remediation Advisory
        lines.append("## Remediation Advisory\n")
        lines.append("> [!WARNING]")
        lines.append(
            "> **ADVISORY ONLY:** All remediation steps must be manually validated and executed by authorized human operators."
        )
        lines.append("> Automated execution is strictly disabled across the platform.\n")

        if incident.recommendations:
            lines.append("| Action ID | Type | Risk | Description | Command | Verification |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for rec in incident.recommendations:
                cmd_block = f"`{sanitize_text(rec.command)}`" if rec.command else "—"
                verif = sanitize_text(rec.verification_step)
                desc = sanitize_text(rec.description)
                lines.append(
                    f"| `{rec.action_id}` | `{rec.action_type}` | `{rec.risk_level.value}` | {desc} | {cmd_block} | {verif} |"
                )
            lines.append("")

            # Action Details
            lines.append("### Detailed Remediation Guidance\n")
            for rec in incident.recommendations:
                lines.append(
                    f"#### `{rec.action_id}`: {sanitize_text(rec.title)} (`{rec.risk_level.value}`)\n"
                )
                lines.append(f"- **Action Type:** `{rec.action_type}`")
                if rec.command:
                    lines.append(f"- **Suggested Command:** `{sanitize_text(rec.command)}`")
                lines.append(
                    f"- **Verification Procedure:** {sanitize_text(rec.verification_step)}"
                )
                if rec.rollback_guidance:
                    lines.append(
                        f"- **Rollback Procedure:** {sanitize_text(rec.rollback_guidance)}"
                    )
                if rec.preconditions:
                    lines.append(
                        f"- **Preconditions:** {', '.join(sanitize_text(p) for p in rec.preconditions)}"
                    )
                lines.append("")
        else:
            lines.append("*No remediation recommendations generated yet for this incident.*\n")

        # Incident Timeline Table
        lines.append("## Incident Timeline\n")
        if incident.timeline:
            lines.append("| Timestamp (UTC) | Source | Component | Severity | Event |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for entry in incident.timeline:
                lines.append(
                    f"| `{entry.timestamp.isoformat()}` | `{entry.source.value}` | `{entry.component}` | `{entry.severity.value}` | {sanitize_text(entry.event)} |"
                )
            lines.append("")
        else:
            lines.append("*No timeline entries recorded.*\n")

        # Correlated Evidence Catalog
        lines.append("## Correlated Evidence Catalog\n")
        catalog = incident.evidence.evidence_catalog
        if catalog:
            lines.append("| Evidence ID | Summary / Value |")
            lines.append("| :--- | :--- |")
            for eid, evalue in sorted(catalog.items()):
                clean_val = sanitize_text(str(evalue)).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{eid}` | {clean_val} |")
            lines.append("")

        if incident.evidence.log_snippets:
            lines.append("### Correlated Log Snippets\n")
            for idx, snippet in enumerate(incident.evidence.log_snippets, 1):
                lines.append(f"```text\n{sanitize_text(snippet.strip())}\n```\n")

        if incident.evidence.metric_anomalies:
            lines.append("### Metric Anomalies\n")
            lines.append("| Metric Name | Recorded Value |")
            lines.append("| :--- | :--- |")
            for mname, mval in sorted(incident.evidence.metric_anomalies.items()):
                lines.append(f"| `{sanitize_text(mname)}` | `{mval}` |")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    @classmethod
    def render_json(cls, incident: Incident, indent: int = 2) -> str:
        """Render the complete incident record as a formatted, sanitized JSON string."""
        raw_dict = incident.model_dump(mode="json")
        raw_json = json.dumps(raw_dict, indent=indent, default=str)
        return sanitize_text(raw_json)

    @classmethod
    def export_report(
        cls,
        incident: Incident,
        format: str = "markdown",
        output_path: Optional[Union[str, Path]] = None,
        console: Optional[Console] = None,
    ) -> str:
        """Export an incident report to a string and optionally write it to a destination file.

        Args:
            incident: The incident model to render.
            format: Output format ('markdown', 'md', 'json', or 'terminal').
            output_path: Destination file path for saving.
            console: Rich console instance for terminal rendering.

        Returns:
            The rendered content string.
        """
        fmt = format.strip().lower()

        if fmt in ("markdown", "md"):
            content = cls.render_markdown(incident)
            if output_path:
                atomic_write_text(output_path, content, force=True)
            return content

        elif fmt == "json":
            content = cls.render_json(incident)
            if output_path:
                atomic_write_text(output_path, content, force=True)
            return content

        elif fmt == "terminal":
            if output_path:
                rec_con = Console(record=True, width=120)
                cls.render_terminal(incident, console=rec_con)
                exported_text = rec_con.export_text()
                atomic_write_text(output_path, exported_text, force=True)
                return exported_text
            else:
                cls.render_terminal(incident, console=console)
                return ""

        else:
            raise DevOpsAIError(
                f"Unsupported report format: '{format}'. Supported formats: markdown, json, terminal"
            )
