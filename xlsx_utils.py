import math
import numbers
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


def write_dataframes_to_xlsx(path, sheets):
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, dataframe in sheets:
                dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
        return
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise

    _write_minimal_xlsx(path, sheets)


def _write_minimal_xlsx(path, sheets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sheet_entries = []
    for idx, (sheet_name, dataframe) in enumerate(sheets, start=1):
        safe_name = str(sheet_name)[:31]
        sheet_entries.append((idx, safe_name, dataframe))

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml(len(sheet_entries)))
        zf.writestr("_rels/.rels", _root_rels_xml())
        zf.writestr("xl/workbook.xml", _workbook_xml(sheet_entries))
        zf.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(sheet_entries))
        zf.writestr("docProps/app.xml", _app_xml())
        zf.writestr("docProps/core.xml", _core_xml())
        for idx, _sheet_name, dataframe in sheet_entries:
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _worksheet_xml(dataframe))


def _worksheet_xml(dataframe):
    rows = [list(dataframe.columns)]
    rows.extend(list(dataframe.itertuples(index=False, name=None)))

    row_xml = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            cell_ref = f"{_column_name(col_idx)}{row_idx}"
            cell_xml = _cell_xml(cell_ref, value)
            if cell_xml:
                cells.append(cell_xml)
        row_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        "</worksheet>"
    )


def _cell_xml(cell_ref, value):
    if _is_blank(value):
        return ""

    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return f'<c r="{cell_ref}"><v>{numeric_value:.12g}</v></c>'

    text = escape(str(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _is_blank(value):
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _column_name(index):
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _content_types_xml(sheet_count):
    sheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f"{sheets}</Types>"
    )


def _root_rels_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def _workbook_xml(sheet_entries):
    sheet_xml = "".join(
        f'<sheet name="{_escape_attr(sheet_name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, sheet_name, _dataframe in sheet_entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheet_xml}</sheets></workbook>"
    )


def _escape_attr(value):
    return escape(str(value), {'"': "&quot;"})


def _workbook_rels_xml(sheet_entries):
    relationships = "".join(
        f'<Relationship Id="rId{idx}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{idx}.xml"/>'
        for idx, _sheet_name, _dataframe in sheet_entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{relationships}</Relationships>"
    )


def _app_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Python</Application></Properties>"
    )


def _core_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:creator>Python</dc:creator></cp:coreProperties>"
    )
