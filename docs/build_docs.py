#!/usr/bin/env python
"""Build the manual from ``docs/src`` into a PDF and the GUI's HTML subset.

Run it after editing anything under ``docs/src``::

    python docs/build_docs.py

Three things are worth knowing about how this works.

The parameter tables are **generated** from :mod:`llmft.config` rather than
written by hand, so the manual, the CLI editor and the web form all describe a
parameter with the same words and the same permitted range. Adding a tunable to
the engine adds it to the manual.

The online documentation is a **subset**, chosen by :data:`ONLINE_SECTIONS`: the
specification asks the browser to carry GUI usage, the concepts, and the
parameter reference, but not the installation instructions or the code tour.
Both outputs come from the same source, so they cannot disagree.

The PDF is built **twice**. The contents list and the index of tables need page
numbers, which are only known once everything has been laid out, so the first
pass collects them and the second uses them.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

DOCS_DIR = Path(__file__).resolve().parent
SOURCE_DIR = DOCS_DIR / "src"
PDF_TARGET = DOCS_DIR / "llm-finetune-manual.pdf"
HTML_TARGET = DOCS_DIR.parent / "llmft" / "web" / "templates" / "manual.html"

TITLE = "LLM Finetune"
SUBTITLE = "QLoRA fine-tuning for open-weight models"
AUTHOR = "Renaud Bidou"
EMAIL = "rbidou@gmail.com"

# A line beginning with this titles the table that follows it.
CAPTION_PREFIX = "Table: "

# Sections the browser carries. A part with no sections named takes the whole
# part; naming sections takes only those, in the order given.
ONLINE_SECTIONS: dict[str, tuple[str, ...]] = {
    "02-user-guide.md": ("GUI usage",),
    "03-concepts.md": (),
    "04-implementation.md": ("Training parameters", "Quantization parameters"),
}

PALETTE = {
    "ink": "#1a2430",
    "accent": "#0b5cad",
    "deep": "#0d2a4a",
    "muted": "#5c6b7a",
    "rule": "#c6d2e0",
    "code_bg": "#f3f6fa",
    "code_edge": "#dde5ee",
    "stripe": "#f1f6fc",
    "cover_text": "#ffffff",
    "cover_dim": "#9dc4ec",
}


@dataclass(frozen=True)
class Section:
    """One ``##`` section of one part.

    The part it came from is deliberately not carried. Online the sections are a
    curated subset, so labelling one "Part 2 — User Guide" would point at a
    structure the reader cannot see and imply four parts are present when three
    of them are not.
    """

    heading: str
    body: str


def main() -> int:
    """Build both outputs. Returns a process exit code."""
    sys.path.insert(0, str(DOCS_DIR.parent))
    parts = _read_parts()
    _write_pdf(parts)
    _write_html(_online_subset(parts))
    print(f"Wrote {PDF_TARGET}")
    print(f"Wrote {HTML_TARGET}")
    return 0


# --- source ----------------------------------------------------------------


def _read_parts() -> dict[str, str]:
    """Read every part, substituting the generated tables."""
    tables = _generated_tables()
    parts: dict[str, str] = {}
    for source in sorted(SOURCE_DIR.glob("*.md")):
        text = source.read_text(encoding="utf-8")
        for name, table in tables.items():
            text = text.replace(f"{{{{{name}}}}}", table)
        remaining = re.findall(r"\{\{(\w+)\}\}", text)
        if remaining:
            raise SystemExit(f"{source.name}: no table named {remaining[0]}")
        parts[source.name] = text
    return parts


def _generated_tables() -> dict[str, str]:
    """Build the parameter tables from the application's own definitions."""
    from llmft.config import QUANT_TYPES, LoraSettings, ParamKind, SftSettings, describe

    def rows(settings: type) -> str:
        lines = ["| Parameter | Default | Accepts | What it does |", "|---|---|---|---|"]
        for spec in describe(settings):
            default = ", ".join(spec.default) if isinstance(spec.default, tuple) else spec.default
            lines.append(f"| `{spec.name}` | {default} | {_accepts(spec, ParamKind)} | {spec.explanation} |")
        return "\n".join(lines)

    quant = ["| Type | Two-pass | Description |", "|---|---|---|"]
    quant += [
        f"| `{info.name}` | {'yes' if info.requires_llama_quantize else 'no'} | {info.explanation} |"
        for info in QUANT_TYPES
    ]

    return {
        "LORA_PARAMETERS": rows(LoraSettings),
        "SFT_PARAMETERS": rows(SftSettings),
        "QUANT_TYPES": "\n".join(quant),
    }


def _accepts(spec, kinds) -> str:
    """Describe what a parameter will accept, for the reference table."""
    if spec.kind is kinds.BOOL:
        return "true / false"
    if spec.choices:
        return ", ".join(f"`{choice}`" for choice in spec.choices)
    if spec.minimum is not None and spec.maximum is not None:
        return f"{_number(spec.minimum)} to {_number(spec.maximum)}"
    return ""


def _number(value: float) -> str:
    """Render a bound without scientific-notation noise for whole numbers."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _online_subset(parts: dict[str, str]) -> list[Section]:
    """Select the sections the browser carries."""
    chosen: list[Section] = []
    for filename, headings in ONLINE_SECTIONS.items():
        sections = _split_sections(parts[filename])
        for heading in headings or tuple(sections):
            if heading not in sections:
                raise SystemExit(f"{filename}: no section titled '{heading}'")
            chosen.append(Section(heading, sections[heading]))
    return chosen


def _split_sections(text: str) -> dict[str, str]:
    """Split a part into its ``##`` sections, keeping order."""
    pieces = re.split(r"^## (.+)$", text, flags=re.MULTILINE)[1:]
    return {heading.strip(): body.strip() for heading, body in zip(pieces[0::2], pieces[1::2])}


# --- HTML ------------------------------------------------------------------


def _write_html(sections: list[Section]) -> None:
    """Render the online subset into a template the Flask app serves."""
    import markdown

    renderer = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])
    counter = {"table": 0}
    articles = []
    contents = []
    for index, section in enumerate(sections):
        anchor = f"doc-{index}"
        contents.append(f'<li data-target="{anchor}"><a href="#{anchor}">{section.heading}</a></li>')
        renderer.reset()
        # A <details> per chapter, closed: collapsing and expanding by hand then
        # works natively, and the reader chooses what to open.
        articles.append(
            f'<details class="doc-section" id="{anchor}">'
            f'<summary><span class="doc-heading">{section.heading}</span></summary>'
            f'<div class="doc-section-body">{renderer.convert(_html_captions(section.body, counter))}</div>'
            f"</details>"
        )

    HTML_TARGET.parent.mkdir(parents=True, exist_ok=True)
    HTML_TARGET.write_text(_HTML_TEMPLATE.format(contents="\n".join(contents), articles="\n".join(articles)), "utf-8")


def _html_captions(body: str, counter: dict) -> str:
    """Turn ``Table:`` lines into numbered captions markdown will pass through."""

    def replace(match: re.Match) -> str:
        counter["table"] += 1
        return f'<p class="doc-table-caption">Table {counter["table"]} &mdash; {match.group(1).strip()}</p>'

    return re.sub(rf"^{CAPTION_PREFIX}(.+)$", replace, body, flags=re.MULTILINE)


# A Jinja template, so the Flask app can serve it with its own layout. The
# braces Jinja needs are doubled because this is also a format string.
_HTML_TEMPLATE = """<!-- Generated by docs/build_docs.py. Edit docs/src/*.md instead. -->
<nav class="doc-contents">
  <div class="doc-search">
    <input type="search" class="form-control" id="doc-search" placeholder="Search the manual&hellip;"
           autocomplete="off" aria-label="Search the manual">
    <span class="doc-search-status" id="doc-search-status"></span>
  </div>
  <h2>Contents</h2>
  <ul id="doc-contents-list">
{contents}
  </ul>
  <p style="margin-top: 16px;">
    <a class="btn btn-sm" href="{{{{ url_for('download_manual') }}}}">&#128196; Download the full manual (PDF)</a>
  </p>
</nav>
<div class="doc-body" id="doc-body">
{articles}
</div>
"""


# --- PDF -------------------------------------------------------------------


class TableIndex(TableOfContents):
    """A second contents list, fed by table captions rather than headings."""

    def notify(self, kind, stuff) -> None:
        if kind == "TableEntry":
            self.addEntry(*stuff)


class Manual(BaseDocTemplate):
    """The manual: a cover with its own background, then the body pages.

    Headings and table captions announce themselves as they are drawn. That is
    what gives the two contents lists their page numbers, and what keeps the
    running header and footer naming the part and chapter the reader is
    actually looking at.
    """

    def __init__(self, target: Path) -> None:
        super().__init__(
            str(target),
            pagesize=A4,
            leftMargin=2.2 * cm,
            rightMargin=2.2 * cm,
            topMargin=2.7 * cm,
            bottomMargin=2.3 * cm,
            title=f"{TITLE} - Manual",
            author=AUTHOR,
            subject=SUBTITLE,
        )
        self.part = ""
        self.chapter = ""
        self._anchors = 0
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        cover = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="cover")
        self.addPageTemplates(
            [
                PageTemplate("cover", [cover], onPage=_draw_cover_background),
                PageTemplate("body", [frame], onPageEnd=self._draw_furniture),
            ]
        )

    def beforeDocument(self) -> None:
        """Reset per-pass state.

        ``multiBuild`` lays the document out repeatedly until the page numbers
        settle. Without this the anchor counter would keep climbing, giving
        every heading a different key on each pass, so the contents list would
        never compare equal to the previous one and the build would never
        converge.
        """
        self._anchors = 0
        self.part = ""
        self.chapter = ""

    def afterFlowable(self, flowable) -> None:
        """Record headings and captions, now that their page is known."""
        if not isinstance(flowable, Paragraph):
            return
        style = flowable.style.name
        if style not in {"h1", "h2", "h3", "caption"}:
            return

        text = flowable.getPlainText()
        self._anchors += 1
        key = f"anchor-{self._anchors}"
        self.canv.bookmarkPage(key)

        if style == "caption":
            self.notify("TableEntry", (0, text, self.page, key))
            return

        level = {"h1": 0, "h2": 1, "h3": 2}[style]
        self.notify("TOCEntry", (level, text, self.page, key))
        if level == 0:
            self.part, self.chapter = text, ""
        elif level == 1:
            self.chapter = text

    def _draw_furniture(self, canvas, document) -> None:
        """Draw the running header and footer, once the page is complete.

        At page *end* rather than page start, so the chapter named in the footer
        is the one the page actually finished in.
        """
        width, height = document.pagesize
        left, right = document.leftMargin, width - document.rightMargin
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(PALETTE["muted"]))
        canvas.setStrokeColor(colors.HexColor(PALETTE["rule"]))
        canvas.setLineWidth(0.5)

        canvas.drawString(left, height - 1.55 * cm, f"{TITLE} — Manual")
        canvas.drawRightString(right, height - 1.55 * cm, self.part)
        canvas.line(left, height - 1.75 * cm, right, height - 1.75 * cm)

        canvas.line(left, 1.65 * cm, right, 1.65 * cm)
        canvas.drawString(left, 1.25 * cm, self.chapter or self.part)
        canvas.setFillColor(colors.HexColor(PALETTE["accent"]))
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawRightString(right, 1.25 * cm, str(document.page))
        canvas.restoreState()


def _draw_cover_background(canvas, document) -> None:
    """Fill the cover and lay an accent band across it."""
    width, height = document.pagesize
    canvas.saveState()
    canvas.setFillColor(colors.HexColor(PALETTE["deep"]))
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor(PALETTE["accent"]))
    canvas.rect(0, height * 0.50, width, 0.30 * cm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor(PALETTE["cover_dim"]))
    canvas.rect(0, height * 0.50 - 0.14 * cm, width * 0.34, 0.08 * cm, stroke=0, fill=1)
    canvas.restoreState()


def _styles() -> dict:
    """Every paragraph style the manual uses."""
    base = getSampleStyleSheet()
    ink = colors.HexColor(PALETTE["ink"])
    accent = colors.HexColor(PALETTE["accent"])
    return {
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=21, leading=25, textColor=accent,
                             spaceBefore=0, spaceAfter=2),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=14.5, leading=18, textColor=accent,
                             spaceBefore=18, spaceAfter=2),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11.5, leading=14, textColor=ink,
                             spaceBefore=13, spaceAfter=5),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13.6, textColor=ink,
                               alignment=TA_LEFT, spaceAfter=7),
        "code": ParagraphStyle("code", parent=base["BodyText"], fontName="Courier", fontSize=8, leading=10.8,
                               textColor=ink, leftIndent=8, borderPadding=6, spaceBefore=2, spaceAfter=10,
                               backColor=colors.HexColor(PALETTE["code_bg"]),
                               borderColor=colors.HexColor(PALETTE["code_edge"]), borderWidth=0.5),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontSize=9.5, leading=13.6, textColor=ink,
                                 leftIndent=15, bulletIndent=4, spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontSize=8.5, leading=11, textColor=accent,
                                  fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=8, leading=10.6, textColor=ink,
                               spaceAfter=0),
        "cellhead": ParagraphStyle("cellhead", parent=base["BodyText"], fontSize=8, leading=10.6, spaceAfter=0,
                                   fontName="Helvetica-Bold",
                                   textColor=colors.HexColor(PALETTE["cover_text"])),
        "coverTitle": ParagraphStyle("coverTitle", parent=base["Title"], fontSize=34, leading=40, alignment=TA_LEFT,
                                     textColor=colors.HexColor(PALETTE["cover_text"]), spaceAfter=6),
        "coverSub": ParagraphStyle("coverSub", parent=base["BodyText"], fontSize=13, leading=18, alignment=TA_LEFT,
                                   textColor=colors.HexColor(PALETTE["cover_dim"]), spaceAfter=0),
        "coverFoot": ParagraphStyle("coverFoot", parent=base["BodyText"], fontSize=10, leading=15, alignment=TA_LEFT,
                                    textColor=colors.HexColor(PALETTE["cover_text"]), spaceAfter=0),
        "tocTitle": ParagraphStyle("tocTitle", parent=base["Heading1"], fontSize=17, leading=21, textColor=accent,
                                   spaceBefore=0, spaceAfter=10),
    }


def _toc_levels() -> list:
    """Indentation and weight for each depth of the contents list."""
    ink = colors.HexColor(PALETTE["ink"])
    accent = colors.HexColor(PALETTE["accent"])
    muted = colors.HexColor(PALETTE["muted"])
    return [
        ParagraphStyle("toc0", fontName="Helvetica-Bold", fontSize=10.5, leading=17, textColor=accent, spaceBefore=9),
        ParagraphStyle("toc1", fontName="Helvetica", fontSize=9.5, leading=14, textColor=ink, leftIndent=14),
        ParagraphStyle("toc2", fontName="Helvetica", fontSize=8.5, leading=12.5, textColor=muted, leftIndent=30),
    ]


def _write_pdf(parts: dict[str, str]) -> None:
    """Render every part into the printable manual."""
    styles = _styles()
    document = Manual(PDF_TARGET)
    accent = colors.HexColor(PALETTE["accent"])

    contents = TableOfContents(dotsMinLevel=0)
    contents.levelStyles = _toc_levels()
    tables = TableIndex(dotsMinLevel=0)
    tables.levelStyles = [_toc_levels()[1]]

    story = _cover(styles)
    story += [
        NextPageTemplate("body"),
        PageBreak(),
        Paragraph("Contents", styles["tocTitle"]),
        HRFlowable(width="100%", thickness=1.2, color=accent, spaceAfter=12),
        contents,
        PageBreak(),
        Paragraph("Index of tables", styles["tocTitle"]),
        HRFlowable(width="100%", thickness=1.2, color=accent, spaceAfter=12),
        tables,
    ]

    counter = {"table": 0}
    for name in sorted(parts):
        story.append(PageBreak())
        story.extend(_render_markdown(parts[name], styles, counter))

    # Built twice, so the contents lists learn the page numbers they point at.
    document.multiBuild(story)


def _cover(styles: dict) -> list:
    """The title page, laid over its own background."""
    return [
        Spacer(1, 250),
        Paragraph(TITLE, styles["coverTitle"]),
        Paragraph(SUBTITLE, styles["coverSub"]),
        Spacer(1, 44),
        Paragraph("Manual", styles["coverSub"]),
        Spacer(1, 200),
        Paragraph(f"{AUTHOR}<br/>{EMAIL}", styles["coverFoot"]),
    ]


def _render_markdown(text: str, styles: dict, counter: dict) -> list:
    """Convert one part's markdown into reportlab flowables.

    A deliberately small subset: headings, paragraphs, bullets, fenced code,
    pipe tables and the ``Table:`` caption convention. That is everything the
    manual uses, and a full markdown engine would be far more machinery than the
    job needs.
    """
    flowables: list = []
    lines = text.splitlines()
    caption = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            block, index = _collect(lines, index + 1, lambda item: item.startswith("```"))
            flowables.append(Paragraph("<br/>".join(_escape(item) for item in block) or " ", styles["code"]))
            index += 1
        elif line.startswith(CAPTION_PREFIX):
            caption = line[len(CAPTION_PREFIX):].strip()
            index += 1
        elif line.startswith("|"):
            block, index = _collect(lines, index, lambda item: not item.startswith("|"))
            flowables.extend(_captioned_table(block, caption, styles, counter))
            caption = ""
        elif line.startswith("### "):
            flowables.append(Paragraph(_inline(line[4:]), styles["h3"]))
            index += 1
        elif line.startswith("## "):
            flowables.append(Paragraph(_inline(line[3:]), styles["h2"]))
            flowables.append(HRFlowable(width="100%", thickness=0.75,
                                        color=colors.HexColor(PALETTE["rule"]), spaceAfter=9))
            index += 1
        elif line.startswith("# "):
            flowables.append(Paragraph(_inline(line[2:]), styles["h1"]))
            flowables.append(HRFlowable(width="100%", thickness=2.5,
                                        color=colors.HexColor(PALETTE["accent"]), spaceAfter=14))
            index += 1
        elif line.startswith("- "):
            flowables.append(Paragraph(_inline(line[2:]), styles["bullet"], bulletText="•"))
            index += 1
        elif line.strip():
            block, index = _collect(lines, index, lambda item: not item.strip() or _starts_block(item))
            flowables.append(Paragraph(_inline(" ".join(block)), styles["body"]))
        else:
            index += 1
    return flowables


def _starts_block(line: str) -> bool:
    """Whether a line begins something that is not a paragraph continuation."""
    return line.startswith(("#", "-", "|", "```", CAPTION_PREFIX))


def _collect(lines: list[str], start: int, stop) -> tuple[list[str], int]:
    """Gather lines from ``start`` until ``stop`` says otherwise."""
    index = start
    while index < len(lines) and not stop(lines[index]):
        index += 1
    return lines[start:index], index


def _captioned_table(block: list[str], caption: str, styles: dict, counter: dict) -> list:
    """A numbered caption and the table it introduces.

    The two are kept together so a caption is never orphaned at the foot of a
    page, separated from the table it names.
    """
    counter["table"] += 1
    title = _escape(caption) if caption else "Table"
    heading = Paragraph(f"Table {counter['table']} — {title}", styles["caption"])
    return [KeepTogether([heading, _table(block, styles)]), Spacer(1, 10)]


def _table(block: list[str], styles: dict) -> Table:
    """Build a table from markdown pipe rows."""
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in block]
    rows = [row for row in rows if not all(set(cell) <= set("-: ") for cell in row)]
    data = [
        [Paragraph(_inline(cell), styles["cellhead" if index == 0 else "cell"]) for cell in row]
        for index, row in enumerate(rows)
    ]
    table = Table(data, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PALETTE["accent"])),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(PALETTE["rule"])),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    commands += [
        ("BACKGROUND", (0, row), (-1, row), colors.HexColor(PALETTE["stripe"]))
        for row in range(2, len(data), 2)
    ]
    table.setStyle(TableStyle(commands))
    return table


def _escape(text: str) -> str:
    """Escape the characters reportlab treats as markup."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline(text: str) -> str:
    """Convert inline markdown into the tags reportlab understands."""
    deep, accent = PALETTE["deep"], PALETTE["accent"]
    escaped = _escape(text)
    escaped = re.sub(r"`([^`]+)`", rf'<font face="Courier" size="8.5" color="{deep}">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", rf'<link href="\2" color="{accent}">\1</link>', escaped)
    return re.sub(r"&lt;(https?://[^&]+)&gt;", rf'<link href="\1" color="{accent}">\1</link>', escaped)


if __name__ == "__main__":
    raise SystemExit(main())
