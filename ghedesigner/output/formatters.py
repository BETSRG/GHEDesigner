import warnings
from typing import Any


def create_title(width: int, title: str, filler_symbol: str = " ") -> str:
    """Centered title line, e.g. `----- Results -----`"""
    return f"{f' {title} ':{filler_symbol}^{width}s}\n"


def create_row(
    width: int,
    row_data: list[Any],
    data_formats: list[str],
    centering: str = ">",
) -> str:
    """
    Format a single row of data into fixed-width columns.
    data_formats: list of format specifiers like ".2f" or "s", one per column.
    """
    out = ""
    n_cols = len(row_data)
    col_w = width // n_cols
    leftover = width % n_cols
    for datum, fmt in zip(row_data, data_formats):
        w = col_w + (1 if leftover > 0 else 0)
        leftover -= 1 if leftover > 0 else 0
        try:
            out += f"{datum:{centering}{w}{fmt}}"
        except Exception as e:
            print(f"Row formatting error ({fmt}):", e)
            raise
    return f"{out}\n"


def create_table(
    title: str,
    col_titles: list[list[str]],
    rows: list[list[Any]],
    width: int,
    col_formats: list[str],
    filler_symbol: str = " ",
    centering: str = ">",
) -> str:
    """
    Build a table with a header title, column headers, dashed lines, and data rows.
    col_titles: list of header rows, each a list of strings.
    """
    txt = create_title(width, title, filler_symbol)
    txt += create_line(width)
    header_fmt = ["s"] * len(col_titles[0])
    for header_row in col_titles:
        txt += create_row(width, header_row, header_fmt, centering="^")
    txt += create_line(width)
    for row in rows:
        txt += create_row(width, row, col_formats, centering=centering)
    txt += create_line(width)
    return txt


def d_row(
    width: int,
    label: str,
    value: Any,
    fmt: str,
    n_tabs: int = 0,
) -> str:
    """
    A simple two-column row: left label, right value, padded in between.
    n_tabs inserts 4*n_tabs spaces before label.
    """
    tab_w = 4
    lead = " " * (n_tabs * tab_w)
    left = f"{lead}{label}"
    right = f"{value:{fmt}}"
    pad = width - len(left) - len(right)
    if pad < 0:
        warnings.warn("Formatting width too small for d_row; truncating")
        pad = 4
    return f"{left}{' ' * pad}{right}\n"


def create_line(width: int, character: str = "*") -> str:
    """A full-width line of a single repeat character."""
    return f"{character * width}\n"
