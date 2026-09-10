from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


TABLES = [
    "metal-gpu-intervals",
    "mps-hw-intervals",
    "metal-application-encoders-list",
    "metal-application-intervals",
    "metal-shader-profiler-intervals",
]


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())


def element_value(
    element: ET.Element,
    references: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    """
    返回：
      display: 人类可读值，优先使用 fmt
      raw:     XML 中的原始数值
    """
    ref = element.attrib.get("ref")
    if ref is not None:
        return references.get(
            ref,
            (f"<ref:{ref}>", ""),
        )

    display = clean_text(element.attrib.get("fmt"))
    raw = clean_text(element.text)

    if not display:
        # 某些值放在子节点中。
        descendant_values = []

        for child in element.iter():
            if child is element:
                continue

            child_fmt = clean_text(
                child.attrib.get("fmt")
            )
            child_text = clean_text(child.text)

            value = child_fmt or child_text

            if value:
                descendant_values.append(value)

        display = ", ".join(
            dict.fromkeys(descendant_values)
        )

    if not display:
        display = raw

    element_id = element.attrib.get("id")

    if element_id is not None:
        references[element_id] = (
            display,
            raw,
        )

    return display, raw


def duration_number(row: dict[str, dict[str, str]]) -> float:
    """
    找 Duration 列并读取原始数值。
    xctrace 通常以纳秒保存 duration。
    """
    for key, value in row.items():
        if "duration" in key.lower():
            raw = value["raw"]

            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

    return 0.0


def shorten(value: str, limit: int = 100) -> str:
    value = clean_text(value)

    if len(value) <= limit:
        return value

    return value[: limit - 3] + "..."


def export_table(
    trace: Path,
    schema: str,
    output: Path,
) -> None:
    xpath = (
        '/trace-toc/run[@number="1"]/data/'
        f'table[@schema="{schema}"]'
    )

    command = [
        "xcrun",
        "xctrace",
        "export",
        "--input",
        str(trace),
        "--xpath",
        xpath,
        "--output",
        str(output),
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def parse_table(
    xml_path: Path,
) -> tuple[list[str], list[dict[str, dict[str, str]]]]:
    root = ET.parse(xml_path).getroot()

    schema_node = root.find(".//schema")

    if schema_node is None:
        return [], []

    columns: list[str] = []

    for index, col in enumerate(
        schema_node.findall("./col")
    ):
        mnemonic = clean_text(
            col.findtext("./mnemonic")
        )
        name = clean_text(
            col.findtext("./name")
        )

        columns.append(
            mnemonic
            or name
            or f"column_{index}"
        )

    references: dict[
        str,
        tuple[str, str],
    ] = {}

    previous_values: list[
        tuple[str, str]
    ] = [
        ("", "")
        for _ in columns
    ]

    rows: list[
        dict[str, dict[str, str]]
    ] = []

    for row_node in root.findall(".//row"):
        children = list(row_node)
        parsed_values: list[
            tuple[str, str]
        ] = []

        for index in range(len(columns)):
            if index >= len(children):
                parsed_values.append(("", ""))
                continue

            element = children[index]
            tag = element.tag.rsplit("}", 1)[-1]

            if tag == "sentinel":
                value = previous_values[index]
            else:
                value = element_value(
                    element,
                    references,
                )

            parsed_values.append(value)

        previous_values = parsed_values

        row: dict[str, dict[str, str]] = {}

        for column, (display, raw) in zip(
            columns,
            parsed_values,
        ):
            row[column] = {
                "display": display,
                "raw": raw,
            }

        rows.append(row)

    return columns, rows


def print_table(
    schema: str,
    columns: list[str],
    rows: list[dict[str, dict[str, str]]],
) -> None:
    print()
    print("=" * 90)
    print(f"TABLE: {schema}")
    print(f"ROWS : {len(rows)}")
    print("COLUMNS:")
    print("  " + "\n  ".join(columns))

    if not rows:
        print("该表没有记录。")
        return

    has_duration = any(
        "duration" in column.lower()
        for column in columns
    )

    if has_duration:
        selected_rows = sorted(
            rows,
            key=duration_number,
            reverse=True,
        )[:30]

        print("\nTOP 30 BY DURATION")
    else:
        selected_rows = rows[:30]
        print("\nFIRST 30 ROWS")

    for index, row in enumerate(
        selected_rows,
        start=1,
    ):
        print(f"\n[{index}]")

        for column in columns:
            value = shorten(
                row[column]["display"]
            )

            if value:
                print(
                    f"  {column:32s} = {value}"
                )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "用法：python3 dump_metal_tables.py "
            "<trace 路径>"
        )

    trace = Path(sys.argv[1]).expanduser()

    if not trace.exists():
        raise SystemExit(
            f"找不到 trace：{trace}"
        )

    print(f"TRACE: {trace}")

    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)

        for schema in TABLES:
            output = temp_dir / f"{schema}.xml"

            try:
                export_table(
                    trace,
                    schema,
                    output,
                )

                columns, rows = parse_table(
                    output
                )

                print_table(
                    schema,
                    columns,
                    rows,
                )

            except subprocess.CalledProcessError as error:
                print()
                print("=" * 90)
                print(f"TABLE: {schema}")
                print(
                    "导出失败：",
                    error,
                )


if __name__ == "__main__":
    main()
