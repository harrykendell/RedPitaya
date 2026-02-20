#!/usr/bin/env python3
"""
Manual Curve Fitting GUI for MI Model
Allows manual adjustment of parameters for failed automatic fits.
"""

import sys
import re
import csv
import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Pattern, Sequence, Tuple
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score
from pandas import read_csv
from scipy.optimize import curve_fit

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
    QSlider,
    QPushButton,
    QComboBox,
    QTextEdit,
    QFileDialog,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
    QAbstractSpinBox,
    QSizePolicy,
    QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QPalette, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

# use Qt5Agg backend for better performance
from matplotlib.pyplot import switch_backend


def mi_model(x, a, b, phi, d):
    """MI model: Voltage = |A*sin(B*Sig gen Output + phi_0) + D|^2"""
    return np.abs(a * np.sin(b * x + np.pi * phi) + d) ** 2


def sine_model(x, amp, freq, phase, offset):
    """Analytic sinusoidal model in the independent variable x."""
    return offset + amp * np.sin(2.0 * np.pi * freq * x + phase)


def interferometer_delay_model(x, amp, freq, delay, phase, offset):
    """Interferometer-oriented model with explicit delay term."""
    return offset + amp * np.sin(2.0 * np.pi * freq * (x - delay) + phase)


def normalize_column_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("(s)", "").replace("(v)", "")
    if text in {"time", "times"}:
        return "TIME"
    if text.startswith("ch"):
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return f"CH{digits}"
    return str(name).strip().upper()


def read_measurement_csv(file_ref: str):
    """Read CSV data from plain files or zip members and normalize channel names."""
    if "::" in file_ref and file_ref.split("::", 1)[0].lower().endswith(".zip"):
        zip_path, member = file_ref.split("::", 1)
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member) as handle:
                raw = handle.read()
        frame = read_csv(BytesIO(raw), header=0)
    else:
        frame = read_csv(file_ref, header=0)
        if frame.shape[1] < 2:
            frame = read_csv(file_ref, skiprows=13, header=0)

    frame = frame.rename(columns={col: normalize_column_name(col) for col in frame.columns})
    if "TIME" not in frame.columns and "TIME(S)" in frame.columns:
        frame = frame.rename(columns={"TIME(S)": "TIME"})
    return frame


def stem_for_file_ref(file_ref: str) -> str:
    if "::" in file_ref:
        _zip_path, member = file_ref.split("::", 1)
        return Path(member).stem
    return Path(file_ref).stem


def display_name_for_file_ref(file_ref: str) -> str:
    """Display only the data-file stem (no zip archive name, no .csv suffix)."""
    return stem_for_file_ref(file_ref)


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            clear_layout(child_layout)


class NumericSortTableWidgetItem(QTableWidgetItem):
    """Sort numerically when both cells contain valid finite numbers."""

    @staticmethod
    def _to_number(text):
        if text is None:
            return None
        stripped = str(text).strip()
        if not stripped:
            return None
        try:
            value = float(stripped)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    def __lt__(self, other):
        if isinstance(other, QTableWidgetItem):
            left_num = self._to_number(self.text())
            right_num = self._to_number(other.text())
            if left_num is not None and right_num is not None:
                return left_num < right_num
        return super().__lt__(other)


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    symbol: str
    description: str
    default: float
    min_value: float
    max_value: float
    slider_scale: int = 1000
    decimals: int = 4

    def slider_limits(self) -> Tuple[int, int]:
        return (
            int(round(self.min_value * self.slider_scale)),
            int(round(self.max_value * self.slider_scale)),
        )

    def to_slider(self, value: float) -> int:
        return int(round(value * self.slider_scale))

    def from_slider(self, value: int) -> float:
        return value / self.slider_scale

    @property
    def inferred_step(self) -> float:
        # Infer a sensible spinbox step from parameter span.
        precision_step = 10 ** (-self.decimals)
        span = abs(self.max_value - self.min_value)
        if span <= 0.0:
            return precision_step
        span_order = int(np.floor(np.log10(span)))
        span_step = 10 ** (span_order - 4)
        return max(precision_step, span_step)

    @property
    def column_name(self) -> str:
        return self.key


@dataclass(frozen=True)
class FitModelSpec:
    key: str
    display_name: str
    formula_latex: str
    formula_fallback: str
    function: Callable[..., np.ndarray]
    params: Sequence[ParameterSpec]

    @property
    def defaults(self):
        return [spec.default for spec in self.params]

    @property
    def bounds(self):
        lower = [spec.min_value for spec in self.params]
        upper = [spec.max_value for spec in self.params]
        return (lower, upper)


# Add/edit entries here to swap fitting models and parameter metadata.
FIT_MODELS = {
    "mi_abs_squared": FitModelSpec(
        key="mi_abs_squared",
        display_name="MI Abs-Squared",
        formula_latex=r"Voltage = \left|A\sin(Bx+\pi\times\phi)+D\right|^2",
        formula_fallback="Voltage = |A·sin(B·x + φ) + D|²",
        function=mi_model,
        params=(
            ParameterSpec(
                key="a",
                symbol="A",
                description="MI Amplitude",
                default=0.74545,
                min_value=0.0,
                max_value=10.0,
            ),
            ParameterSpec(
                key="b",
                symbol="B",
                description="Voltage to Phase",
                default=-0.2175,
                min_value=-2.0,
                max_value=2.0,
            ),
            ParameterSpec(
                key="phi",
                symbol="φ",
                description="MI Phase",
                default=0.0,
                min_value=-2,
                max_value=2,
            ),
            ParameterSpec(
                key="d",
                symbol="D",
                description="MI Offset",
                default=1.7019,
                min_value=-10.0,
                max_value=10.0,
            ),
        ),
    ),
    "sine": FitModelSpec(
        key="sine",
        display_name="Sine (analytic)",
        formula_latex=r"y = C + A\sin(2\pi f x + \phi)",
        formula_fallback="y = C + A·sin(2π·f·x + φ)",
        function=sine_model,
        params=(
            ParameterSpec("amp", "A", "Amplitude", 1.0, -20.0, 20.0),
            ParameterSpec("freq", "f", "Frequency (1/x)", 1.0, -1e6, 1e6, decimals=6),
            ParameterSpec("phase", "φ", "Phase (rad)", 0.0, -20.0, 20.0),
            ParameterSpec("offset", "C", "Offset", 0.0, -20.0, 20.0),
        ),
    ),
    "interferometer_delay": FitModelSpec(
        key="interferometer_delay",
        display_name="Interferometer Delay",
        formula_latex=r"y = C + A\sin(2\pi f(x-\tau)+\phi)",
        formula_fallback="y = C + A·sin(2π·f·(x-τ) + φ)",
        function=interferometer_delay_model,
        params=(
            ParameterSpec("amp", "A", "Amplitude", 1.0, -20.0, 20.0),
            ParameterSpec("freq", "f", "Frequency (1/x)", 1.0, -1e6, 1e6, decimals=6),
            ParameterSpec("delay", "τ", "Delay in x-units", 0.0, -1.0, 1.0, decimals=6),
            ParameterSpec("phase", "φ", "Relative phase (rad)", 0.0, -20.0, 20.0),
            ParameterSpec("offset", "C", "Offset", 0.0, -20.0, 20.0),
        ),
    ),
}
ACTIVE_MODEL = FIT_MODELS["mi_abs_squared"]
FIT_CURVE_COLOR = "#16a34a"

# Initialize backend after model definitions.
switch_backend("Qt5Agg")


def compute_r2(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if np.count_nonzero(valid) < 2:
        return None
    try:
        return float(r2_score(y_true_arr[valid], y_pred_arr[valid]))
    except Exception:
        return None


@dataclass(frozen=True)
class CapturePatternConfig:
    mode: str
    regex_pattern: str
    regex: Optional[Pattern[str]]
    defaults: Dict[str, str]


_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_optional_delimiter(char: str) -> bool:
    """Return True for punctuation delimiters that may wrap optional fields."""
    return bool(char) and (not char.isalnum()) and (char not in "{}*") and (not char.isspace())


def _template_to_regex(template_text: str) -> Tuple[str, Dict[str, str]]:
    """Convert a template into regex and default values for optional fields."""
    if not template_text:
        raise ValueError("Template is empty.")

    pieces = ["^"]
    defaults = {}
    seen_fields = set()
    idx = 0
    length = len(template_text)
    literal_buffer = []

    def flush_literal():
        if literal_buffer:
            pieces.append(re.escape("".join(literal_buffer)))
            literal_buffer.clear()

    while idx < length:
        char = template_text[idx]
        if char == "{":
            end = template_text.find("}", idx + 1)
            if end < 0:
                raise ValueError("Missing closing '}' in template.")
            field_spec = template_text[idx + 1 : end].strip()
            manual_prefix = None
            manual_suffix = None
            if "=" in field_spec:
                field_name, default_spec = field_spec.split("=", 1)
                field_name = field_name.strip()
                affix_parts = default_spec.split("|")
                if len(affix_parts) == 1:
                    default_value = affix_parts[0].strip()
                elif len(affix_parts) == 2:
                    default_value = affix_parts[0].strip()
                    manual_prefix = affix_parts[1]
                elif len(affix_parts) == 3:
                    default_value = affix_parts[0].strip()
                    manual_prefix = affix_parts[1]
                    manual_suffix = affix_parts[2]
                else:
                    raise ValueError(
                        f"Optional field '{field_name}' has too many '|' parts. "
                        "Use {field=default|prefix|suffix}."
                    )
                if default_value == "":
                    raise ValueError(
                        f"Optional field '{field_name}' must define a non-empty default."
                    )
                optional = True
            else:
                field_name = field_spec
                default_value = None
                optional = False

            if not _FIELD_NAME_RE.fullmatch(field_name):
                raise ValueError(
                    f"Invalid field name '{field_name}'. Use letters, numbers, underscore."
                )
            if field_name in seen_fields:
                raise ValueError(f"Duplicate field name '{field_name}'.")

            prefix = ""
            suffix = ""
            if optional:
                if manual_prefix is not None:
                    prefix = manual_prefix
                elif literal_buffer and _is_optional_delimiter(literal_buffer[-1]):
                    prefix = literal_buffer.pop()

            flush_literal()

            if optional:
                defaults[field_name] = default_value
                if manual_suffix is not None:
                    suffix = manual_suffix
                    idx = end + 1
                else:
                    next_idx = end + 1
                    if next_idx < length and _is_optional_delimiter(template_text[next_idx]):
                        suffix = template_text[next_idx]
                        idx = next_idx + 1
                    else:
                        idx = end + 1

                optional_pieces = []
                if prefix:
                    optional_pieces.append(re.escape(prefix))
                optional_pieces.append(f"(?P<{field_name}>.+?)")
                if suffix:
                    optional_pieces.append(re.escape(suffix))
                pieces.append(f"(?:{''.join(optional_pieces)})?")
            else:
                pieces.append(f"(?P<{field_name}>.+?)")
                idx = end + 1
            seen_fields.add(field_name)
            continue
        if char == "}":
            raise ValueError("Unexpected '}' in template.")
        if char == "*":
            flush_literal()
            pieces.append(".*")
            idx += 1
            continue

        literal_buffer.append(char)
        idx += 1

    flush_literal()
    pieces.append("$")
    return "".join(pieces), defaults


def parse_capture_pattern(pattern_text: str) -> CapturePatternConfig:
    """Parse capture input as simple template or advanced regex."""
    text = pattern_text.strip()
    if not text:
        return CapturePatternConfig(
            mode="off", regex_pattern="", regex=None, defaults={}
        )

    if text.startswith("re:"):
        regex_pattern = text[3:].strip()
        if not regex_pattern:
            raise ValueError("Regex mode requires a pattern after 're:'.")
        regex = re.compile(regex_pattern)
        return CapturePatternConfig(
            mode="regex", regex_pattern=regex_pattern, regex=regex, defaults={}
        )

    if "(?P<" in text:
        regex = re.compile(text)
        return CapturePatternConfig(
            mode="regex", regex_pattern=text, regex=regex, defaults={}
        )

    regex_pattern, defaults = _template_to_regex(text)
    regex = re.compile(regex_pattern)
    return CapturePatternConfig(
        mode="template", regex_pattern=regex_pattern, regex=regex, defaults=defaults
    )


def extract_captures(
    stem: str,
    regex: Optional[Pattern[str]],
    defaults: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
    """Extract captures from a filename stem, supporting named and positional groups."""
    if regex is None:
        return {}
    match = regex.search(stem)
    if not match:
        return None

    capture_defaults = defaults or {}
    captures = match.groupdict()
    if captures:
        for key, default_value in capture_defaults.items():
            if captures.get(key) in (None, ""):
                captures[key] = default_value
        return captures

    groups = match.groups()
    if groups:
        return {f"group_{idx + 1}": value for idx, value in enumerate(groups)}

    return {"match": match.group(0)}


def render_batch_thumbnail(row, model_func, full_thumbnail_size=(468, 312)):
    """Render a row thumbnail pixmap, including fitted curve when available."""
    try:
        data = read_measurement_csv(row["file"])
        time_col = "TIME" if "TIME" in data.columns else data.columns[0]
        x_col = row.get("x_channel") or ("CH3" if "CH3" in data.columns else data.columns[0])
        y_col = row.get("y_channel") or ("CH2" if "CH2" in data.columns else data.columns[0])
        time_data = data[time_col].to_numpy(dtype=float, copy=True) * 1e3
        y_data = data[y_col].to_numpy(dtype=float, copy=True)
        x_data = data[x_col].to_numpy(dtype=float, copy=True)

        target_width = max(24, int(full_thumbnail_size[0]))
        target_height = max(24, int(full_thumbnail_size[1]))
        render_dpi = 180
        fig = Figure(
            figsize=(target_width / render_dpi, target_height / render_dpi),
            dpi=render_dpi,
        )
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16)
        ax = fig.add_subplot(111)
        ax.plot(time_data, y_data, linewidth=1.25, color="C0")
        ax.plot(time_data, x_data, linewidth=1.25, alpha=0.45, color="C1")

        params = row.get("params")
        if params:
            fitted_y = model_func(x_data, *params)
            ax.plot(time_data, fitted_y, linewidth=1.25, color=FIT_CURVE_COLOR)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.15)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=render_dpi)
        buf.seek(0)
        data_bytes = buf.getvalue()
        buf.close()

        pixmap = QPixmap()
        pixmap.loadFromData(data_bytes, "PNG")
        return pixmap
    except Exception:
        pixmap = QPixmap(full_thumbnail_size[0], full_thumbnail_size[1])
        pixmap.fill(Qt.GlobalColor.white)
        return pixmap


class FitWorker(QObject):
    finished = pyqtSignal(object, object, float)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, x_data, y_data, p0, bounds, model_func):
        super().__init__()
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data = np.asarray(y_data, dtype=float)
        self.p0 = np.asarray(p0, dtype=float)
        self.bounds = bounds
        self.model_func = model_func
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    @pyqtSlot()
    def run(self):
        try:
            if self.cancel_requested:
                self.cancelled.emit()
                return

            popt, pcov = curve_fit(
                self.model_func,
                self.x_data,
                self.y_data,
                p0=self.p0,
                bounds=self.bounds,
                method="trf",
                maxfev=2000,
            )
            if self.cancel_requested:
                self.cancelled.emit()
                return

            fitted = self.model_func(self.x_data, *popt)
            r2 = compute_r2(self.y_data, fitted)
            r2 = float(r2) if r2 is not None else float("nan")
            self.finished.emit(popt, pcov, r2)
        except Exception as exc:
            if self.cancel_requested:
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))


class BatchFitWorker(QObject):
    progress = pyqtSignal(int, int, object)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        file_paths,
        p0,
        bounds,
        regex_pattern,
        capture_defaults,
        model_func,
        x_channel,
        y_channel,
        fit_start_pct,
        fit_end_pct,
    ):
        super().__init__()
        self.file_paths = list(file_paths)
        self.p0 = np.asarray(p0, dtype=float)
        self.bounds = bounds
        self.regex = re.compile(regex_pattern) if regex_pattern else None
        self.capture_defaults = dict(capture_defaults or {})
        self.model_func = model_func
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.fit_start_pct = fit_start_pct
        self.fit_end_pct = fit_end_pct
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    def _fit_single_file(self, source_index, file_path):
        if self.cancel_requested:
            return None
        row = {
            "_source_index": source_index,
            "file": file_path,
            "captures": {},
            "params": None,
            "r2": None,
            "error": None,
            "x_channel": self.x_channel,
            "y_channel": self.y_channel,
        }

        captures = extract_captures(
            stem_for_file_ref(file_path),
            self.regex,
            self.capture_defaults,
        )
        if captures:
            row["captures"] = captures

        data = read_measurement_csv(file_path)
        x_data = data[self.x_channel].to_numpy(dtype=float, copy=True)
        y_data = data[self.y_channel].to_numpy(dtype=float, copy=True)
        n = len(x_data)
        start = int(np.floor((self.fit_start_pct / 100.0) * max(0, n - 1)))
        end = int(np.ceil((self.fit_end_pct / 100.0) * max(0, n - 1))) + 1
        start = max(0, min(n - 1, start)) if n else 0
        end = max(start + 1, min(n, end)) if n else 0
        fit_slice = slice(start, end)
        x_fit = x_data[fit_slice]
        y_fit = y_data[fit_slice]

        if self.cancel_requested:
            return None

        popt, _pcov = curve_fit(
            self.model_func,
            x_fit,
            y_fit,
            p0=self.p0,
            bounds=self.bounds,
            method="trf",
            maxfev=1000,
        )

        if self.cancel_requested:
            return None

        fitted = self.model_func(x_fit, *popt)
        r2_val = compute_r2(y_fit, fitted)
        row["params"] = [float(x) for x in popt]
        row["r2"] = float(r2_val) if r2_val is not None else None
        return row

    @pyqtSlot()
    def run(self):
        results = [None] * len(self.file_paths)
        try:
            total = len(self.file_paths)
            if total == 0:
                self.finished.emit([])
                return

            ideal = int(QThread.idealThreadCount())
            if ideal <= 0:
                ideal = 4
            max_workers = max(1, min(8, ideal))

            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._fit_single_file, idx, file_path): idx
                    for idx, file_path in enumerate(self.file_paths)
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    if self.cancel_requested:
                        for pending in futures:
                            pending.cancel()
                        self.cancelled.emit()
                        return

                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {
                            "_source_index": idx,
                            "file": self.file_paths[idx],
                            "captures": {},
                            "params": None,
                            "r2": None,
                            "error": str(exc),
                            "x_channel": self.x_channel,
                            "y_channel": self.y_channel,
                        }
                    if row is None:
                        self.cancelled.emit()
                        return

                    results[idx] = row
                    completed += 1
                    self.progress.emit(completed, total, row)

            if self.cancel_requested:
                self.cancelled.emit()
                return

            self.finished.emit([row for row in results if row is not None])
        except Exception as exc:
            if self.cancel_requested:
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))


class ThumbnailRenderWorker(QObject):
    progress = pyqtSignal(int, int, object)
    finished = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(
        self,
        batch_results,
        model_func,
        full_thumbnail_size=(468, 312),
        row_indices=None,
    ):
        super().__init__()
        self.batch_results = batch_results
        self.model_func = model_func
        self.full_thumbnail_size = full_thumbnail_size
        if row_indices is None:
            self.row_indices = list(range(len(batch_results)))
        else:
            self.row_indices = sorted(
                {
                    int(idx)
                    for idx in row_indices
                    if 0 <= int(idx) < len(batch_results)
                }
            )
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    @pyqtSlot()
    def run(self):
        try:
            total = len(self.row_indices)
            for done_idx, row_idx in enumerate(self.row_indices):
                if self.cancel_requested:
                    self.cancelled.emit()
                    return

                row = self.batch_results[row_idx]
                if row.get("plot_full") is not None:
                    self.progress.emit(done_idx + 1, total, row_idx)
                    continue

                pixmap = self.render_thumbnail(row)
                row["plot_full"] = pixmap
                self.progress.emit(done_idx + 1, total, row_idx)

            if self.cancel_requested:
                self.cancelled.emit()
                return

            self.finished.emit()
        except Exception:
            self.finished.emit()

    def render_thumbnail(self, row):
        """Render a plot to a QPixmap for embedding in table."""
        return render_batch_thumbnail(
            row,
            self.model_func,
            full_thumbnail_size=self.full_thumbnail_size,
        )


class ManualFitGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_spec = ACTIVE_MODEL
        self.param_specs = list(self.model_spec.params)
        self.param_spinboxes = {}
        self.param_sliders = {}

        self.setWindowTitle(f"Curve Fitting ({self.model_spec.display_name})")
        self.setGeometry(100, 100, 900, 900)

        self.data_files = []
        self.current_file_idx = 0
        self.current_data = None
        self.channels = {
            "CH2": "MI output voltage",
            "CH3": "Sig Gen / TTL",
            "CH4": "TTL / trigger",
            "TIME": "Time",
        }
        self.x_channel = "CH3"
        self.y_channel = "CH2"
        self.fit_region_start_pct = 0.0
        self.fit_region_end_pct = 100.0
        self.extrapolate_unfitted = True
        self._fit_window_bounds_ms = (None, None)
        self.fit_region_selector = None
        self._suppress_fit_region_selector = False
        self._fit_region_refresh_pending = False
        self.last_popt = None
        self.last_pcov = None
        self.last_fit_r2 = None
        self.last_full_r2 = None
        self._last_r2_fit = None
        self._last_r2_full = None
        self.fit_thread = None
        self.fit_worker = None
        self.batch_thread = None
        self.batch_worker = None
        self.batch_fit_in_progress = False
        self.batch_results = []
        self.batch_files = []
        self.batch_capture_keys = []
        self.batch_match_count = 0
        self.batch_unmatched_files = []
        self.analysis_csv_records = []
        self.analysis_csv_path = None
        self.analysis_records = []
        self.analysis_columns = []
        self.analysis_numeric_data = {}
        self.analysis_param_columns = []
        self.max_thumbnails = 8
        self.thumb_cols = 1
        self.batch_row_height = 64
        self.batch_row_height_min = 40
        self.batch_row_height_max = 320
        self.batch_thumbnail_aspect = 1.5
        self.batch_thumbnail_supersample = 5.0
        self._batch_row_height_sync = False
        self._batch_progress_done = 0
        self.regex_timer = QTimer()
        self.regex_timer.setSingleShot(True)
        self.regex_timer.timeout.connect(self._do_prepare_batch_preview)
        self.thumb_thread = None
        self.thumb_worker = None
        self.thumb_render_in_progress = False
        self._pending_thumbnail_rows = set()

        # Fit model defaults and bounds come from model metadata.
        self.defaults = list(self.model_spec.defaults)
        self.bounds = self.model_spec.bounds

        # Optimization: timer for debouncing updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.do_full_update)
        self.slider_active = False

        # Cache for plot data
        self.cached_time_data = None
        self.channel_cache = {}
        self._display_target_points = 3000
        self._plot_has_residual_axis = False

        # Current directory
        self.current_dir = "./AFG_measurements/"

        # Create central widget
        central_widget = QWidget()
        central_widget.setObjectName("root")
        self.setCentralWidget(central_widget)
        self._enforce_light_mode()
        self._apply_compact_ui_defaults()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # Shared controls for all modes (manual + batch).
        self.create_file_frame(main_layout)

        # Shared parameters/statistics for all modes.
        params_stats_layout = QHBoxLayout()
        params_stats_layout.setSpacing(6)
        self.create_parameters_frame(params_stats_layout)
        self.create_stats_frame(params_stats_layout)
        params_stats_layout.setStretch(0, 3)
        params_stats_layout.setStretch(1, 2)
        main_layout.addLayout(params_stats_layout)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.manual_tab = QWidget()
        self.batch_tab = QWidget()
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.manual_tab, "Plot")
        self.tabs.addTab(self.batch_tab, "Batch Processing")
        self.tabs.addTab(self.analysis_tab, "Batch Analysis")

        manual_layout = QVBoxLayout(self.manual_tab)
        manual_layout.setContentsMargins(6, 6, 6, 6)
        manual_layout.setSpacing(4)

        # Manual mode: interactive plot only (controls are shared above tabs).
        self.create_plot_frame(manual_layout)

        batch_layout = QVBoxLayout(self.batch_tab)
        batch_layout.setContentsMargins(6, 6, 6, 6)
        batch_layout.setSpacing(6)
        self.create_batch_controls_frame(batch_layout)
        self.create_batch_results_frame(batch_layout)

        analysis_layout = QVBoxLayout(self.analysis_tab)
        analysis_layout.setContentsMargins(6, 6, 6, 6)
        analysis_layout.setSpacing(6)
        self.create_batch_analysis_frame(analysis_layout)
        analysis_layout.addStretch()

        # Load files
        self.load_files()

    def _enforce_light_mode(self):
        """Force a light Qt palette regardless of system appearance."""
        app = QApplication.instance()
        if app is None:
            return

        app.setStyle("Fusion")

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f5f7fa"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#111827"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f8fafc"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#111827"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#111827"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#111827"))
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#dbeafe"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#111827"))

        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#9ca3af")
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.ButtonText,
            QColor("#9ca3af"),
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.WindowText,
            QColor("#9ca3af"),
        )

        app.setPalette(palette)

    def _apply_compact_ui_defaults(self):
        """Apply a clean, compact light UI stylesheet."""
        self.setStyleSheet(
            """
            QMainWindow, QWidget#root, QTabWidget > QWidget {
                background: #f5f7fa;
                color: #111827;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #e3e8ef;
                border-radius: 8px;
                margin-top: 0px;
                padding-top: 2px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0px;
                margin: 0px;
                color: transparent;
                max-height: 0px;
            }
            QPushButton {
                min-height: 22px;
                padding: 2px 8px;
                background: #ffffff;
                color: #111827;
                border-radius: 8px;
                border: 1px solid #d3dae3;
            }
            QPushButton:hover {
                background: #f3f6f9;
                border-color: #c7d0dc;
            }
            QPushButton:pressed {
                background: #eaf0f5;
            }
            QPushButton:checked {
                background: #dbeafe;
                border-color: #93c5fd;
            }
            QPushButton:disabled {
                color: #9ca3af;
                background: #f5f7fa;
                border-color: #e4e9ef;
            }
            QPushButton[primary="true"] {
                background: #2563eb;
                color: white;
                border-color: #1d4ed8;
            }
            QPushButton[primary="true"]:hover {
                background: #1d4ed8;
            }
            QPushButton[primary="true"]:pressed {
                background: #1e40af;
            }
            QLineEdit, QComboBox, QDoubleSpinBox {
                min-height: 22px;
                background: #ffffff;
                color: #111827;
                border: 1px solid #d3dae3;
                border-radius: 6px;
                padding: 1px 6px;
            }
            QLineEdit:read-only {
                background: #f3f6f9;
                color: #4b5563;
            }
            QComboBox::drop-down {
                border: none;
                width: 16px;
            }
            QTextEdit {
                padding: 3px;
                background: #ffffff;
                color: #111827;
                border: 1px solid #d3dae3;
                border-radius: 6px;
            }
            QTabWidget::pane {
                border: 1px solid #e3e8ef;
                border-radius: 8px;
                background: #ffffff;
                top: -1px;
            }
            QTabBar::tab {
                background: #eef2f6;
                border: 1px solid #d7dde6;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 4px 10px;
                margin-right: 2px;
                color: #4b5563;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #111827;
            }
            QTableWidget {
                background: #ffffff;
                border: 1px solid #d3dae3;
                border-radius: 6px;
                gridline-color: #e7ecf2;
                alternate-background-color: #f8fafc;
            }
            QHeaderView::section {
                background: #f3f6f9;
                color: #374151;
                border: 1px solid #e2e8f0;
                padding: 4px;
            }
            QLabel#statusLabel {
                color: #2563eb;
                font-weight: 600;
                padding: 1px 2px;
            }
            QLabel#paramRange {
                color: #6b7280;
                font-size: 10px;
                padding: 0px;
            }
            QLabel#paramInline {
                color: #374151;
                padding: 0px 2px 0px 0px;
            }
            QSlider::groove:horizontal {
                border: none;
                background: #dbe3ec;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #6b7280;
                border: 1px solid #4b5563;
                width: 12px;
                margin: -5px 0;
                border-radius: 6px;
            }
            QToolBar {
                background: #ffffff;
                border: 1px solid #e3e8ef;
                border-radius: 6px;
                spacing: 2px;
            }
            QToolButton {
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 2px;
            }
            QToolButton:hover {
                background: #f3f6f9;
                border-color: #d3dae3;
            }
            """
        )

    def _make_compact_tool_button(self, text, tooltip, handler):
        """Build a fixed-width toolbar-like button for file navigation."""
        btn = QPushButton(text)
        btn.setFixedWidth(30)
        btn.setToolTip(tooltip)
        btn.clicked.connect(handler)
        return btn

    def _format_param_bound(self, value):
        """Format min/max bounds compactly for parameter labels."""
        if np.isclose(value, round(value), atol=1e-9):
            return str(int(round(value)))
        return f"{value:.4g}"

    def _param_range_text(self, spec):
        return (
            f"[{self._format_param_bound(spec.min_value)}, "
            f"{self._format_param_bound(spec.max_value)}]"
        )

    def _create_param_label(self, spec, width):
        """Create a one-line parameter label with inline range."""
        label = QLabel(f"{spec.symbol} {self._param_range_text(spec)}:")
        label.setObjectName("paramInline")
        label.setToolTip(spec.description)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setMinimumWidth(width)
        label.setMaximumWidth(width)
        return label

    def evaluate_model(self, x_data, params):
        """Evaluate active fit model using positional parameter values."""
        return self.model_spec.function(x_data, *params)

    def _render_formula_pixmap(self):
        """Render model LaTeX formula into a transparent pixmap."""
        try:
            fig = Figure(figsize=(4, 1.4), dpi=200)
            fig.patch.set_alpha(0.0)
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.axis("off")
            ax.text(
                0.0,
                0.5,
                f"${self.model_spec.formula_latex}$",
                fontsize=28,
                va="center",
                ha="left",
            )
            buf = BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=200,
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.0,
            )
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue(), "PNG")
            buf.close()
            return pixmap
        except Exception:
            return None

    def _set_formula_label(self):
        """Populate the formula label, preferring LaTeX rendering."""
        target_height = max(1, self.formula_label.height(), 60)
        self.formula_label.setFixedHeight(target_height)
        pixmap = self._render_formula_pixmap()
        if pixmap is not None and not pixmap.isNull():
            self.formula_label.setPixmap(
                pixmap.scaledToHeight(
                    target_height, Qt.TransformationMode.SmoothTransformation
                )
            )
            self.formula_label.setToolTip(f"${self.model_spec.formula_latex}$")
            self.formula_label.setText("")
            return
        self.formula_label.setPixmap(QPixmap())
        self.formula_label.setText(self.model_spec.formula_fallback)

    def create_file_frame(self, parent_layout):
        """Create shared data/model/fit settings section used by all modes."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        # Directory row
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(4)
        dir_layout.addWidget(QLabel("Files:"))
        browse_dir_btn = self._make_compact_tool_button(
            "📁", "Browse Source (folder or ZIP)", self.browse_directory
        )
        dir_layout.addWidget(browse_dir_btn)

        self.dir_input = QLineEdit(self.current_dir)
        self.dir_input.setReadOnly(True)
        dir_layout.addWidget(self.dir_input, 2)

        # File row
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        dir_layout.addWidget(self.file_combo, 2)

        # Navigation buttons
        prev_btn = self._make_compact_tool_button("◀", "Previous File", self.prev_file)
        dir_layout.addWidget(prev_btn)

        next_btn = self._make_compact_tool_button("▶", "Next File", self.next_file)
        dir_layout.addWidget(next_btn)

        layout.addLayout(dir_layout)

        cfg_layout = QHBoxLayout()
        cfg_layout.setSpacing(4)
        cfg_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for key, spec in FIT_MODELS.items():
            self.model_combo.addItem(spec.display_name, key)
        self.model_combo.setCurrentText(self.model_spec.display_name)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        cfg_layout.addWidget(self.model_combo)

        cfg_layout.addWidget(QLabel("X:"))
        self.x_channel_combo = QComboBox()
        self.x_channel_combo.currentIndexChanged.connect(self.on_channel_selection_changed)
        cfg_layout.addWidget(self.x_channel_combo)
        cfg_layout.addWidget(QLabel("Y:"))
        self.y_channel_combo = QComboBox()
        self.y_channel_combo.currentIndexChanged.connect(self.on_channel_selection_changed)
        cfg_layout.addWidget(self.y_channel_combo)
        layout.addLayout(cfg_layout)

        region_layout = QHBoxLayout()
        region_layout.setSpacing(4)
        reset_region_btn = QPushButton("Reset Fit Window")
        reset_region_btn.clicked.connect(self.reset_fit_region)
        region_layout.addWidget(reset_region_btn)
        self.extrapolate_toggle = QCheckBox("Plot fit across full graph")
        self.extrapolate_toggle.setChecked(self.extrapolate_unfitted)
        self.extrapolate_toggle.toggled.connect(self.on_extrapolate_toggled)
        region_layout.addWidget(self.extrapolate_toggle)
        region_layout.addStretch(1)
        layout.addLayout(region_layout)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_plot_frame(self, parent_layout):
        """Create plot section."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        self.fig = Figure(figsize=(9, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax_residual = None
        self.canvas = FigureCanvas(self.fig)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.16)

        # Add matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(14, 14))
        self.toolbar.setMaximumHeight(28)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self._recreate_fit_region_selector()

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_parameters_frame(self, parent_layout):
        """Create compact parameter controls section."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(4)
        self.formula_label = QLabel(self.model_spec.formula_fallback)
        self.formula_label.setFixedHeight(60)
        self.formula_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        header_layout.addWidget(self.formula_label)
        header_layout.addStretch()
        self.show_residuals_cb = QPushButton("Residuals")
        self.show_residuals_cb.setCheckable(True)
        self.show_residuals_cb.setChecked(True)
        self.show_residuals_cb.toggled.connect(lambda: self.update_plot(fast=False))
        header_layout.addWidget(self.show_residuals_cb)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_params)
        header_layout.addWidget(reset_btn)
        self._set_formula_label()
        layout.addLayout(header_layout)

        self.param_controls_layout = QVBoxLayout()
        self.param_controls_layout.setSpacing(4)
        layout.addLayout(self.param_controls_layout)
        self.rebuild_manual_param_controls()

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_param_control(self, spec, default_val):
        """Helper to create compact parameter control with spinbox and slider."""
        layout = QHBoxLayout()
        layout.setSpacing(4)

        layout.addWidget(self._create_param_label(spec, width=130))

        # Spinbox
        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(spec.min_value)
        spinbox.setMaximum(spec.max_value)
        spinbox.setValue(default_val)
        spinbox.setSingleStep(spec.inferred_step)
        spinbox.setDecimals(spec.decimals)
        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spinbox.setKeyboardTracking(False)
        spinbox.setMinimumWidth(80)
        spinbox.setMaximumWidth(80)
        spinbox.valueChanged.connect(lambda: self.update_plot(fast=False))
        layout.addWidget(spinbox)

        # Slider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider_limits = spec.slider_limits()
        slider.setMinimum(slider_limits[0])
        slider.setMaximum(slider_limits[1])
        slider.setValue(spec.to_slider(default_val))
        slider.setFixedHeight(18)

        # Connect slider to spinbox and vice versa
        def slider_to_spinbox(value):
            spinbox.blockSignals(True)
            spinbox.setValue(spec.from_slider(value))
            spinbox.blockSignals(False)
            self.update_plot(fast=True)

        def spinbox_to_slider(value):
            slider.blockSignals(True)
            slider.setValue(spec.to_slider(value))
            slider.blockSignals(False)

        def slider_pressed():
            self.slider_active = True

        def slider_released():
            self.slider_active = False
            self.do_full_update()

        slider.valueChanged.connect(slider_to_spinbox)
        spinbox.valueChanged.connect(spinbox_to_slider)
        slider.sliderPressed.connect(slider_pressed)
        slider.sliderReleased.connect(slider_released)
        layout.addWidget(slider)

        return (layout, spinbox, slider)

    def rebuild_manual_param_controls(self):
        if not hasattr(self, "param_controls_layout"):
            return
        clear_layout(self.param_controls_layout)
        self.param_spinboxes.clear()
        self.param_sliders.clear()
        for idx, spec in enumerate(self.param_specs):
            default_val = self.defaults[idx] if idx < len(self.defaults) else spec.default
            control_layout, spinbox, slider = self.create_param_control(spec, default_val)
            self.param_spinboxes[spec.key] = spinbox
            self.param_sliders[spec.key] = slider
            self.param_controls_layout.addLayout(control_layout)

    def on_model_changed(self, _idx):
        model_key = self.model_combo.currentData()
        if not model_key:
            return
        spec = FIT_MODELS.get(model_key)
        if spec is None:
            return
        self.model_spec = spec
        self.param_specs = list(spec.params)
        self.defaults = list(spec.defaults)
        self.bounds = spec.bounds
        self.setWindowTitle(f"Curve Fitting ({self.model_spec.display_name})")
        self.rebuild_manual_param_controls()
        self._set_formula_label()
        self.update_plot(fast=False)
        if self.batch_results:
            for row in self.batch_results:
                row["plot_full"] = None
                row["plot"] = None
            self.update_batch_table()
            self.queue_visible_thumbnail_render()

    def _refresh_channel_combos(self):
        if self.current_data is None:
            return
        channel_columns = [col for col in self.current_data.columns if col.startswith("CH") or col == "TIME"]
        if not channel_columns:
            return

        self.x_channel_combo.blockSignals(True)
        self.y_channel_combo.blockSignals(True)
        self.x_channel_combo.clear()
        self.y_channel_combo.clear()
        for col in channel_columns:
            self.x_channel_combo.addItem(col, col)
            self.y_channel_combo.addItem(col, col)

        x_choice = self.x_channel if self.x_channel in channel_columns else ("CH3" if "CH3" in channel_columns else channel_columns[0])
        y_choice = self.y_channel if self.y_channel in channel_columns else ("CH2" if "CH2" in channel_columns else channel_columns[0])
        self.x_channel = x_choice
        self.y_channel = y_choice
        self.x_channel_combo.setCurrentText(x_choice)
        self.y_channel_combo.setCurrentText(y_choice)
        self.x_channel_combo.blockSignals(False)
        self.y_channel_combo.blockSignals(False)

    def on_channel_selection_changed(self, _idx):
        x_choice = self.x_channel_combo.currentData()
        y_choice = self.y_channel_combo.currentData()
        if x_choice:
            self.x_channel = x_choice
        if y_choice:
            self.y_channel = y_choice
        self.update_plot(fast=False)

    def _set_fit_region(self, start_pct, end_pct, refresh=True):
        start = float(np.clip(start_pct, 0.0, 100.0))
        end = float(np.clip(end_pct, 0.0, 100.0))
        if start > end:
            start, end = end, start

        self.fit_region_start_pct = start
        self.fit_region_end_pct = end

        if refresh:
            self.update_plot(fast=False)

    def _schedule_fit_region_refresh(self):
        """Schedule redraw after SpanSelector event completes to avoid UI lag."""
        if self._fit_region_refresh_pending:
            return
        self._fit_region_refresh_pending = True

        def _refresh():
            self._fit_region_refresh_pending = False
            self.update_plot(fast=False)

        QTimer.singleShot(0, _refresh)

    def reset_fit_region(self):
        self._set_fit_region(0.0, 100.0, refresh=True)

    def on_extrapolate_toggled(self, checked):
        self.extrapolate_unfitted = bool(checked)
        self.update_plot(fast=False)

    def get_fit_slice(self, n_points):
        start = int(np.floor((self.fit_region_start_pct / 100.0) * max(0, n_points - 1)))
        end = int(np.ceil((self.fit_region_end_pct / 100.0) * max(0, n_points - 1))) + 1
        start = max(0, min(n_points - 1, start)) if n_points else 0
        end = max(start + 1, min(n_points, end)) if n_points else 0
        return slice(start, end)

    def _fit_window_times(self, time_data, fit_slice):
        n_points = len(time_data)
        if n_points == 0:
            return None, None, 0, 0
        start_idx = int(fit_slice.start if fit_slice.start is not None else 0)
        end_idx = int(fit_slice.stop if fit_slice.stop is not None else n_points)
        start_idx = max(0, min(n_points - 1, start_idx))
        end_idx = max(start_idx + 1, min(n_points, end_idx))
        return (
            float(time_data[start_idx]),
            float(time_data[end_idx - 1]),
            start_idx,
            end_idx,
        )

    def _draw_fit_window_overlay(self, time_data, fit_slice):
        fit_start_t, fit_end_t, start_idx, end_idx = self._fit_window_times(
            time_data, fit_slice
        )
        self._fit_window_bounds_ms = (fit_start_t, fit_end_t)
        if fit_start_t is None or fit_end_t is None:
            return

        if start_idx > 0:
            self.ax.axvline(
                fit_start_t,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                zorder=4,
                label="Fit boundary",
            )
        if end_idx < len(time_data):
            self.ax.axvline(
                fit_end_t,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                zorder=4,
                label="_nolegend_",
            )

    def _split_inside_outside_fit(self, values, fit_slice):
        values = np.asarray(values, dtype=float)
        inside = np.full_like(values, np.nan, dtype=float)
        outside = values.copy()
        inside[fit_slice] = values[fit_slice]
        outside[fit_slice] = np.nan
        return inside, outside

    def _toolbar_mode_active(self):
        return bool(getattr(self.toolbar, "mode", ""))

    def _recreate_fit_region_selector(self):
        if not hasattr(self, "ax"):
            return

        if self.fit_region_selector is not None:
            try:
                self.fit_region_selector.set_active(False)
                self.fit_region_selector.disconnect_events()
            except Exception:
                pass
            self.fit_region_selector = None

        selector_kwargs = dict(
            useblit=True,
            interactive=True,
            props=dict(
                facecolor="none",
                edgecolor="#dc2626",
                linewidth=1.1,
                alpha=0.9,
            ),
        )
        try:
            self.fit_region_selector = SpanSelector(
                self.ax,
                self.on_fit_span_selected,
                "horizontal",
                drag_from_anywhere=True,
                **selector_kwargs,
            )
        except TypeError:
            self.fit_region_selector = SpanSelector(
                self.ax,
                self.on_fit_span_selected,
                "horizontal",
                **selector_kwargs,
            )

    def _sync_fit_region_selector(self):
        if self.fit_region_selector is None:
            return
        fit_start_t, fit_end_t = self._fit_window_bounds_ms
        if fit_start_t is None or fit_end_t is None:
            return
        self._suppress_fit_region_selector = True
        try:
            self.fit_region_selector.extents = (float(fit_start_t), float(fit_end_t))
        finally:
            self._suppress_fit_region_selector = False

    def on_fit_span_selected(self, x_min, x_max):
        if self._suppress_fit_region_selector:
            return
        if self.current_data is None or self.cached_time_data is None:
            return
        if self._toolbar_mode_active():
            return
        if x_min is None or x_max is None:
            return

        lo = float(min(x_min, x_max))
        hi = float(max(x_min, x_max))
        if np.isclose(lo, hi):
            return

        time_data = self.cached_time_data
        if len(time_data) < 2:
            return

        t_min = float(min(time_data[0], time_data[-1]))
        t_max = float(max(time_data[0], time_data[-1]))
        if np.isclose(t_min, t_max):
            return

        lo = float(np.clip(lo, t_min, t_max))
        hi = float(np.clip(hi, t_min, t_max))
        if hi <= lo:
            return

        start_pct = ((lo - t_min) / (t_max - t_min)) * 100.0
        end_pct = ((hi - t_min) / (t_max - t_min)) * 100.0
        min_gap = 100.0 / max(1, len(time_data) - 1)
        if (end_pct - start_pct) < min_gap:
            center = 0.5 * (start_pct + end_pct)
            start_pct = max(0.0, center - (0.5 * min_gap))
            end_pct = min(100.0, start_pct + min_gap)
            if (end_pct - start_pct) < min_gap:
                start_pct = max(0.0, end_pct - min_gap)

        self._set_fit_region(start_pct, end_pct, refresh=False)
        self._schedule_fit_region_refresh()

    def create_stats_frame(self, parent_layout):
        """Create statistics display section with auto-fit controls."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        # Auto-fit control buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.auto_fit_btn = QPushButton("Auto Fit")
        self.auto_fit_btn.setProperty("primary", True)
        self.auto_fit_btn.clicked.connect(self.auto_fit)
        btn_layout.addWidget(self.auto_fit_btn)

        self.cancel_fit_btn = QPushButton("Cancel")
        self.cancel_fit_btn.clicked.connect(self.cancel_auto_fit)
        self.cancel_fit_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_fit_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_params)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

        # Stats text display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(160)
        layout.addWidget(self.stats_text)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_batch_controls_frame(self, parent_layout):
        """Create batch-only controls (shared params/settings are above tabs)."""
        group = QGroupBox("")
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        batch_label = QLabel("Batch actions (uses shared model/params/fit window above)")
        batch_label.setStyleSheet("font-weight: 600; color: #374151; padding: 1px 2px;")
        layout.addWidget(batch_label)

        self.run_batch_btn_default_text = "Run Batch"
        self.run_batch_btn = QPushButton(self.run_batch_btn_default_text)
        self.run_batch_btn.clicked.connect(self.run_batch_fit)

        export_table_btn = QPushButton("Export CSV")
        export_table_btn.clicked.connect(self.export_batch_table)

        regex_layout = QHBoxLayout()
        regex_layout.setSpacing(4)
        regex_layout.addWidget(QLabel("Pattern:"))
        self.regex_input = QLineEdit("data_{freq}_{idx}_ALL")
        self.regex_input.setPlaceholderText("Example: data_{freq}_{idx}_ALL")
        self.regex_input.setToolTip(
            "Simple mode: use {field} placeholders (and * wildcard).\n"
            "Optional with default: use {field=default} (e.g. filename_{idx=0}).\n"
            "Manual optional affix: {field=default|prefix|suffix} "
            "(e.g. filename{ver=0|_v} or filename{ver=0|(|)}).\n"
            "Advanced regex: prefix with re: or use (?P<name>...) groups."
        )
        self.regex_input.textChanged.connect(self._on_regex_changed)
        regex_layout.addWidget(self.regex_input)
        layout.addLayout(regex_layout)

        self.batch_parse_feedback_label = QLabel(
            "Use {field} placeholders to extract columns."
        )
        self.batch_parse_feedback_label.setObjectName("statusLabel")
        layout.addWidget(self.batch_parse_feedback_label)

        self.batch_file_scope_label = QLabel("")
        self.batch_file_scope_label.setObjectName("statusLabel")
        self.update_batch_file_list()
        layout.addWidget(self.batch_file_scope_label)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(4)
        actions_row.addWidget(self.run_batch_btn)
        self.cancel_batch_btn = QPushButton("Cancel")
        self.cancel_batch_btn.clicked.connect(self.cancel_batch_fit)
        self.cancel_batch_btn.setEnabled(False)
        actions_row.addWidget(self.cancel_batch_btn)
        actions_row.addWidget(export_table_btn)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self.batch_status_label = QLabel("")
        self.batch_status_label.setObjectName("statusLabel")
        self.batch_status_label.hide()
        layout.addWidget(self.batch_status_label)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_batch_results_frame(self, parent_layout):
        """Create batch results table."""
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(0)
        self.batch_table.setRowCount(0)
        self.batch_table.cellClicked.connect(self._on_batch_table_cell_clicked)
        batch_header = self.batch_table.horizontalHeader()
        batch_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        batch_header.setStretchLastSection(True)
        batch_header.setMinimumSectionSize(60)
        v_header = self.batch_table.verticalHeader()
        v_header.setVisible(True)
        v_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        v_header.setMinimumSectionSize(self.batch_row_height_min)
        v_header.setToolTip("Drag row borders to resize all rows")
        v_header.sectionResized.connect(self._on_batch_row_resized_by_user)
        # Apply a uniform row height for plot previews.
        v_header.setDefaultSectionSize(self._current_batch_row_height())
        self.batch_table.setSortingEnabled(True)
        batch_header.setSortIndicatorShown(True)
        batch_header.setSectionsClickable(True)
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.verticalScrollBar().valueChanged.connect(
            self.queue_visible_thumbnail_render
        )
        parent_layout.addWidget(self.batch_table)

    def create_batch_analysis_frame(self, parent_layout):
        """Create interactive batch analysis plot controls."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        source_row = QHBoxLayout()
        source_row.setSpacing(4)
        source_row.addWidget(QLabel("Analysis Source:"))
        self.analysis_source_combo = QComboBox()
        self.analysis_source_combo.addItem("Completed Batch Run", "run")
        self.analysis_source_combo.addItem("Loaded Batch CSV", "csv")
        self.analysis_source_combo.currentIndexChanged.connect(
            self._on_analysis_source_changed
        )
        source_row.addWidget(self.analysis_source_combo)
        self.analysis_load_csv_btn = QPushButton("Load Batch CSV")
        self.analysis_load_csv_btn.clicked.connect(self.load_batch_analysis_csv)
        self.analysis_load_csv_btn.setEnabled(False)
        source_row.addWidget(self.analysis_load_csv_btn)
        self.analysis_status_label = QLabel("Using completed batch results (0 rows).")
        self.analysis_status_label.setObjectName("statusLabel")
        source_row.addWidget(self.analysis_status_label, 1)
        layout.addLayout(source_row)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        filter_row.addWidget(QLabel("Filters:"))
        self.analysis_exclude_errors_cb = QCheckBox("Exclude error rows")
        self.analysis_exclude_errors_cb.setChecked(True)
        self.analysis_exclude_errors_cb.toggled.connect(
            lambda: self._refresh_batch_analysis_data(preserve_selection=True)
        )
        filter_row.addWidget(self.analysis_exclude_errors_cb)
        self.analysis_min_r2_spin = QDoubleSpinBox()
        self.analysis_min_r2_spin.setRange(-1.0, 1.0)
        self.analysis_min_r2_spin.setDecimals(3)
        self.analysis_min_r2_spin.setSingleStep(0.01)
        self.analysis_min_r2_spin.setValue(-1.0)
        self.analysis_min_r2_spin.setPrefix("R² ≥ ")
        self.analysis_min_r2_spin.valueChanged.connect(
            lambda _val: self._refresh_batch_analysis_data(preserve_selection=True)
        )
        filter_row.addWidget(self.analysis_min_r2_spin)
        self.analysis_remove_outliers_cb = QCheckBox("Remove IQR outliers")
        self.analysis_remove_outliers_cb.setChecked(False)
        self.analysis_remove_outliers_cb.toggled.connect(
            self.update_batch_analysis_plot
        )
        filter_row.addWidget(self.analysis_remove_outliers_cb)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(4)
        controls_row.addWidget(QLabel("Field (X):"))
        self.analysis_x_combo = QComboBox()
        self.analysis_x_combo.currentIndexChanged.connect(self.update_batch_analysis_plot)
        controls_row.addWidget(self.analysis_x_combo, 2)
        self.analysis_clear_x_btn = QPushButton("Clear X")
        self.analysis_clear_x_btn.clicked.connect(self._clear_analysis_x_field)
        controls_row.addWidget(self.analysis_clear_x_btn)
        controls_row.addWidget(QLabel("Plot:"))
        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItem("Combined", "combined")
        self.analysis_mode_combo.addItem("One per parameter", "separate")
        self.analysis_mode_combo.currentIndexChanged.connect(
            self.update_batch_analysis_plot
        )
        controls_row.addWidget(self.analysis_mode_combo)

        self.analysis_show_points_btn = QPushButton("Points")
        self.analysis_show_points_btn.setCheckable(True)
        self.analysis_show_points_btn.setChecked(True)
        self.analysis_show_points_btn.toggled.connect(self.update_batch_analysis_plot)
        controls_row.addWidget(self.analysis_show_points_btn)

        self.analysis_show_series_line_btn = QPushButton("Series Line")
        self.analysis_show_series_line_btn.setCheckable(True)
        self.analysis_show_series_line_btn.setChecked(False)
        self.analysis_show_series_line_btn.toggled.connect(
            self.update_batch_analysis_plot
        )
        controls_row.addWidget(self.analysis_show_series_line_btn)

        self.analysis_fit_line_btn = QPushButton("Best-Fit Lines")
        self.analysis_fit_line_btn.setCheckable(True)
        self.analysis_fit_line_btn.setChecked(True)
        self.analysis_fit_line_btn.toggled.connect(self.update_batch_analysis_plot)
        controls_row.addWidget(self.analysis_fit_line_btn)

        self.analysis_legend_btn = QPushButton("Legend")
        self.analysis_legend_btn.setCheckable(True)
        self.analysis_legend_btn.setChecked(True)
        self.analysis_legend_btn.toggled.connect(self.update_batch_analysis_plot)
        controls_row.addWidget(self.analysis_legend_btn)

        self.analysis_grid_btn = QPushButton("Grid")
        self.analysis_grid_btn.setCheckable(True)
        self.analysis_grid_btn.setChecked(True)
        self.analysis_grid_btn.toggled.connect(self.update_batch_analysis_plot)
        controls_row.addWidget(self.analysis_grid_btn)

        self.analysis_refresh_btn = QPushButton("Refresh")
        self.analysis_refresh_btn.clicked.connect(
            lambda: self._refresh_batch_analysis_data(preserve_selection=True)
        )
        controls_row.addWidget(self.analysis_refresh_btn)
        layout.addLayout(controls_row)

        params_row = QHBoxLayout()
        params_row.setSpacing(4)
        params_row.addWidget(QLabel("Parameters (Y):"))
        self.analysis_param_buttons = {}
        self.analysis_params_button_layout = QHBoxLayout()
        self.analysis_params_button_layout.setSpacing(4)
        params_row.addLayout(self.analysis_params_button_layout, 1)
        param_btn_col = QVBoxLayout()
        param_btn_col.setSpacing(4)
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all_analysis_params)
        param_btn_col.addWidget(select_all_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_analysis_params)
        param_btn_col.addWidget(clear_btn)
        param_btn_col.addStretch()
        params_row.addLayout(param_btn_col)
        layout.addLayout(params_row)

        self.analysis_fig = Figure(figsize=(10, 3.2), dpi=100)
        self.analysis_fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.2)
        self.analysis_canvas = FigureCanvas(self.analysis_fig)
        layout.addWidget(self.analysis_canvas)

        group.setLayout(layout)
        parent_layout.addWidget(group)

        self._refresh_batch_analysis_data(preserve_selection=False)

    def _on_analysis_source_changed(self):
        source = self.analysis_source_combo.currentData()
        self.analysis_load_csv_btn.setEnabled(source == "csv")
        self._refresh_batch_analysis_data(preserve_selection=True)

    def load_batch_analysis_csv(self):
        """Load a previously exported batch CSV for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Batch CSV",
            str(Path.cwd()),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not file_path:
            return
        try:
            frame = read_csv(file_path, header=0)
            self.analysis_csv_records = frame.to_dict("records")
            self.analysis_csv_path = file_path
            csv_idx = self.analysis_source_combo.findData("csv")
            if csv_idx >= 0:
                self.analysis_source_combo.setCurrentIndex(csv_idx)
            self._refresh_batch_analysis_data(preserve_selection=False)
            self.stats_text.append(
                f"✓ Loaded analysis CSV: {Path(file_path).name} ({len(self.analysis_csv_records)} rows)"
            )
        except Exception as exc:
            self.stats_text.append(f"✗ Failed to load analysis CSV: {exc}")

    def _extract_analysis_records_from_batch(self):
        records = []
        for row in self.batch_results:
            record = {"File": display_name_for_file_ref(row["file"])}
            captures = row.get("captures") or {}
            for key, value in captures.items():
                record[key] = value
            params = row.get("params") or []
            for idx, spec in enumerate(self.param_specs):
                record[spec.column_name] = params[idx] if idx < len(params) else None
            record["R2"] = row.get("r2")
            record["Error"] = row.get("error") or ""
            records.append(record)
        return records

    def _extract_analysis_columns(self, records):
        columns = []
        for row in records:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)
        return columns

    def _coerce_numeric_array(self, values):
        numeric = []
        for value in values:
            if value is None:
                numeric.append(np.nan)
                continue
            text = str(value).strip()
            if text == "":
                numeric.append(np.nan)
                continue
            try:
                numeric.append(float(text))
            except Exception:
                numeric.append(np.nan)
        return np.asarray(numeric, dtype=float)

    def _default_analysis_x_field(self, numeric_columns):
        for key in self.batch_capture_keys:
            if key in numeric_columns:
                return key
        for key in numeric_columns:
            if key not in self.analysis_param_columns and key != "R2":
                return key
        return numeric_columns[0] if numeric_columns else None

    def _parse_optional_float(self, value):
        if value is None:
            return None
        try:
            number = float(str(value).strip())
        except Exception:
            return None
        return number if np.isfinite(number) else None

    def _record_passes_analysis_filters(self, row):
        if (
            hasattr(self, "analysis_exclude_errors_cb")
            and self.analysis_exclude_errors_cb.isChecked()
            and str(row.get("Error", "")).strip()
        ):
            return False

        min_r2 = (
            self.analysis_min_r2_spin.value()
            if hasattr(self, "analysis_min_r2_spin")
            else -1.0
        )
        if min_r2 > -1.0 + 1e-12:
            r2_val = self._parse_optional_float(row.get("R2"))
            if r2_val is None or r2_val < float(min_r2):
                return False
        return True

    def _iqr_inlier_mask(self, values):
        vals = np.asarray(values, dtype=float).reshape(-1)
        finite = np.isfinite(vals)
        if np.count_nonzero(finite) < 4:
            return finite
        q1, q3 = np.percentile(vals[finite], [25.0, 75.0])
        iqr = float(q3 - q1)
        if not np.isfinite(iqr) or np.isclose(iqr, 0.0):
            return finite
        lo = float(q1 - 1.5 * iqr)
        hi = float(q3 + 1.5 * iqr)
        return finite & (vals >= lo) & (vals <= hi)

    def _refresh_batch_analysis_data(self, preserve_selection):
        source = self.analysis_source_combo.currentData()
        if source == "csv":
            raw_records = list(self.analysis_csv_records)
            if raw_records:
                file_name = (
                    Path(self.analysis_csv_path).name
                    if self.analysis_csv_path
                    else "CSV"
                )
                base_status = f"Loaded CSV: {file_name}"
            else:
                base_status = "Loaded CSV"
        else:
            raw_records = self._extract_analysis_records_from_batch()
            base_status = "Using completed batch results"

        records = [
            row for row in raw_records if self._record_passes_analysis_filters(row)
        ]
        self.analysis_status_label.setText(
            f"{base_status} ({len(records)}/{len(raw_records)} rows after filters)."
        )

        self.analysis_records = records
        self.analysis_columns = self._extract_analysis_columns(records)
        self.analysis_numeric_data = {}
        for column in self.analysis_columns:
            values = [row.get(column, "") for row in records]
            as_numeric = self._coerce_numeric_array(values)
            if np.isfinite(as_numeric).sum() > 0:
                self.analysis_numeric_data[column] = as_numeric

        numeric_columns = list(self.analysis_numeric_data.keys())
        self.analysis_param_columns = [
            spec.column_name
            for spec in self.param_specs
            if spec.column_name in self.analysis_numeric_data
        ]
        if not self.analysis_param_columns:
            self.analysis_param_columns = [
                key for key in numeric_columns if key not in ("R2",)
            ]

        previous_x = (
            self.analysis_x_combo.currentData() if preserve_selection else None
        )
        previous_params = (
            set(self._selected_analysis_params()) if preserve_selection else set()
        )

        self.analysis_x_combo.blockSignals(True)
        self.analysis_x_combo.clear()
        self.analysis_x_combo.addItem("Select X Axis...", None)
        for key in numeric_columns:
            self.analysis_x_combo.addItem(key, key)
        self.analysis_x_combo.blockSignals(False)

        if preserve_selection and previous_x is None:
            chosen_x = None
        else:
            chosen_x = previous_x if previous_x in numeric_columns else None
            if chosen_x is None:
                chosen_x = self._default_analysis_x_field(numeric_columns)
        x_idx = self.analysis_x_combo.findData(chosen_x)
        if x_idx < 0:
            x_idx = self.analysis_x_combo.findData(None)
        if x_idx >= 0:
            self.analysis_x_combo.setCurrentIndex(x_idx)

        self._rebuild_analysis_param_buttons(previous_params)

        self.update_batch_analysis_plot()

    def _selected_analysis_params(self):
        return [
            key
            for key, button in self.analysis_param_buttons.items()
            if button.isChecked()
        ]

    def _rebuild_analysis_param_buttons(self, previous_params):
        while self.analysis_params_button_layout.count():
            item = self.analysis_params_button_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.analysis_param_buttons = {}
        for key in self.analysis_param_columns:
            button = QPushButton(key)
            button.setCheckable(True)
            button.setChecked(not previous_params or key in previous_params)
            button.toggled.connect(self.update_batch_analysis_plot)
            self.analysis_params_button_layout.addWidget(button)
            self.analysis_param_buttons[key] = button
        self.analysis_params_button_layout.addStretch()

    def _select_all_analysis_params(self):
        for button in self.analysis_param_buttons.values():
            button.setChecked(True)
        self.update_batch_analysis_plot()

    def _clear_analysis_params(self):
        for button in self.analysis_param_buttons.values():
            button.setChecked(False)
        self.update_batch_analysis_plot()

    def _clear_analysis_x_field(self):
        idx = self.analysis_x_combo.findData(None)
        if idx >= 0:
            self.analysis_x_combo.setCurrentIndex(idx)
        self.update_batch_analysis_plot()

    def _show_analysis_message(self, message):
        self.analysis_fig.clear()
        ax = self.analysis_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_axis_off()
        self.analysis_canvas.draw_idle()

    def _linear_fit(self, x_data, y_data):
        if x_data.size < 2:
            return None
        if np.isclose(float(np.ptp(x_data)), 0.0):
            return None
        try:
            slope, intercept = np.polyfit(x_data, y_data, 1)
            return float(slope), float(intercept)
        except Exception:
            return None

    def update_batch_analysis_plot(self):
        """Plot parameter variation against selected field."""
        if not hasattr(self, "analysis_fig"):
            return
        if not self.analysis_numeric_data:
            self._show_analysis_message("No numeric data available for analysis.")
            return

        x_field = self.analysis_x_combo.currentData()
        selected_params = self._selected_analysis_params()
        if x_field not in self.analysis_numeric_data:
            self._show_analysis_message("Select an X field to plot.")
            return
        if not selected_params:
            self._show_analysis_message("Select at least one parameter to plot.")
            return

        x_values = self.analysis_numeric_data[x_field]
        mode = self.analysis_mode_combo.currentData()
        show_points = self.analysis_show_points_btn.isChecked()
        show_series_line = self.analysis_show_series_line_btn.isChecked()
        show_fit_lines = self.analysis_fit_line_btn.isChecked()
        show_legend = self.analysis_legend_btn.isChecked()
        show_grid = self.analysis_grid_btn.isChecked()

        if not (show_points or show_series_line or show_fit_lines):
            self._show_analysis_message("Enable at least one plot layer (Points/Line/Fit).")
            return

        self.analysis_fig.clear()
        if mode == "separate" and len(selected_params) > 1:
            axes = self.analysis_fig.subplots(len(selected_params), 1, sharex=True)
            axes = list(np.atleast_1d(axes))
        else:
            axes = [self.analysis_fig.add_subplot(111)]

        plotted_any = False
        for idx, param_name in enumerate(selected_params):
            y_values = self.analysis_numeric_data.get(param_name)
            if y_values is None:
                continue
            mask = np.isfinite(x_values) & np.isfinite(y_values)
            if np.count_nonzero(mask) == 0:
                continue

            plotted_any = True
            x_plot = x_values[mask]
            y_plot = y_values[mask]
            if (
                hasattr(self, "analysis_remove_outliers_cb")
                and self.analysis_remove_outliers_cb.isChecked()
            ):
                inlier_mask = self._iqr_inlier_mask(y_plot)
                x_plot = x_plot[inlier_mask]
                y_plot = y_plot[inlier_mask]
                if x_plot.size == 0:
                    continue
            order = np.argsort(x_plot)
            x_sorted = x_plot[order]
            y_sorted = y_plot[order]
            color = f"C{idx % 10}"
            target_ax = axes[idx] if len(axes) > 1 else axes[0]

            if show_points:
                scatter_label = param_name if not show_series_line else "_nolegend_"
                target_ax.scatter(
                    x_sorted, y_sorted, s=26, color=color, label=scatter_label
                )
            if show_series_line:
                target_ax.plot(
                    x_sorted,
                    y_sorted,
                    linewidth=1.4,
                    alpha=0.85,
                    color=color,
                    label=param_name,
                )

            if show_fit_lines:
                fit = self._linear_fit(x_sorted, y_sorted)
                if fit is not None:
                    slope, intercept = fit
                    x_line = np.linspace(float(np.min(x_sorted)), float(np.max(x_sorted)), 200)
                    y_line = slope * x_line + intercept
                    fit_label = (
                        f"{param_name} fit" if len(axes) == 1 else "Best fit"
                    )
                    target_ax.plot(
                        x_line,
                        y_line,
                        linestyle="--",
                        linewidth=1.6,
                        color=color,
                        label=fit_label,
                    )

            if len(axes) > 1:
                target_ax.set_ylabel(param_name)
                target_ax.grid(show_grid, alpha=0.25)
                if show_legend:
                    target_ax.legend(loc="best", fontsize=8)

        if not plotted_any:
            self._show_analysis_message(
                "No finite X/Y pairs available for the selected fields."
            )
            return

        if len(axes) == 1:
            axes[0].set_ylabel("Parameter Value")
            if show_legend:
                axes[0].legend(loc="best", fontsize=8)
            axes[0].grid(show_grid, alpha=0.3)

        axes[-1].set_xlabel(x_field)
        self.analysis_fig.tight_layout()
        self.analysis_canvas.draw_idle()

    def _current_batch_row_height(self):
        return max(
            self.batch_row_height_min,
            min(self.batch_row_height_max, int(self.batch_row_height)),
        )

    def _current_batch_thumbnail_size(self):
        row_height = self._current_batch_row_height()
        thumb_height = max(24, row_height - 8)
        thumb_width = max(36, int(round(thumb_height * self.batch_thumbnail_aspect)))
        return (thumb_width, thumb_height)

    def _full_batch_thumbnail_size(self):
        full_height = max(
            24,
            int(round((self.batch_row_height_max - 8) * self.batch_thumbnail_supersample)),
        )
        full_width = max(36, int(round(full_height * self.batch_thumbnail_aspect)))
        return (full_width, full_height)

    def _apply_batch_row_heights(self):
        if not hasattr(self, "batch_table"):
            return
        if self._batch_row_height_sync:
            return

        row_height = self._current_batch_row_height()
        self._batch_row_height_sync = True
        try:
            self.batch_table.verticalHeader().setDefaultSectionSize(row_height)
            if self.batch_table.columnCount() > 0:
                thumb_width, _ = self._current_batch_thumbnail_size()
                self.batch_table.setColumnWidth(0, thumb_width + 18)
            for row_idx in range(self.batch_table.rowCount()):
                self.batch_table.setRowHeight(row_idx, row_height)
        finally:
            self._batch_row_height_sync = False

    def _scaled_batch_plot(self, row):
        source = row.get("plot_full") or row.get("plot")
        if source is None:
            return None
        target_width, target_height = self._current_batch_thumbnail_size()
        return source.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _find_table_row_by_file(self, file_path):
        """Find table row index by file path stored in item user data."""
        if self.batch_table.columnCount() == 0:
            return None
        for row_idx in range(self.batch_table.rowCount()):
            item = self.batch_table.item(row_idx, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == file_path:
                return row_idx
        return None

    def _on_batch_row_resized_by_user(self, _logical_index, _old_size, new_size):
        if self._batch_row_height_sync:
            return
        self.batch_row_height = max(
            self.batch_row_height_min,
            min(self.batch_row_height_max, int(new_size)),
        )
        self._apply_batch_row_heights()
        for row in self.batch_results:
            row_idx = self._find_table_row_by_file(row["file"])
            if row_idx is not None:
                self._update_batch_plot_cell(row_idx, row)
        self.queue_visible_thumbnail_render()

    def _sync_batch_files_from_shared(self, sync_pattern=True):
        """Mirror batch files from shared file list (default: all files in folder)."""
        self.batch_files = list(self.data_files)

        if (
            sync_pattern
            and hasattr(self, "regex_input")
            and self.batch_files
        ):
            first_name = display_name_for_file_ref(self.batch_files[0])
            if self.regex_input.text() != first_name:
                self.regex_input.blockSignals(True)
                self.regex_input.setText(first_name)
                self.regex_input.blockSignals(False)

        self.update_batch_file_list()
        self.prepare_batch_preview()
        self._expand_file_column_for_selected_files()

    def load_files(self):
        """Load CSV sources from a directory root or a ZIP archive root."""
        source_path = Path(self.current_dir).expanduser()

        files = []
        if source_path.is_dir():
            csv_files = sorted(str(path) for path in source_path.glob("*.csv"))
            zip_members = []
            for zip_path in sorted(source_path.glob("*.zip")):
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        for member in sorted(zf.namelist()):
                            if member.lower().endswith(".csv") and not member.endswith("/"):
                                zip_members.append(f"{zip_path}::{member}")
                except Exception:
                    continue
            files = csv_files + zip_members
        elif source_path.is_file() and source_path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(source_path) as zf:
                    files = [
                        f"{source_path}::{member}"
                        for member in sorted(zf.namelist())
                        if member.lower().endswith(".csv") and not member.endswith("/")
                    ]
            except Exception as exc:
                self.stats_text.setText(f"Failed to read ZIP root: {exc}")
                files = []
        elif source_path.is_file() and source_path.suffix.lower() == ".csv":
            files = [str(source_path)]

        self.data_files = files

        self.file_combo.clear()
        if not self.data_files:
            self._sync_batch_files_from_shared(sync_pattern=False)
            self.stats_text.setText("No CSV files found in selected source.")
            return

        # Populate combo box
        for file in self.data_files:
            self.file_combo.addItem(display_name_for_file_ref(file), file)

        self._sync_batch_files_from_shared(sync_pattern=True)

        # Load first file
        if self.data_files:
            self.load_file(0)

    def load_file(self, idx):
        """Load a specific file."""
        if idx < 0 or idx >= len(self.data_files):
            return

        self.current_file_idx = idx
        self.file_combo.setCurrentIndex(idx)
        self.last_popt = None
        self.last_pcov = None
        self.last_fit_r2 = None
        self.last_full_r2 = None
        self._last_r2_fit = None
        self._last_r2_full = None

        try:
            file_path = self.data_files[idx]
            self.current_data = read_measurement_csv(file_path)
            # Cache data for faster updates
            time_src = "TIME" if "TIME" in self.current_data.columns else self.current_data.columns[0]
            self.cached_time_data = self.current_data[time_src].to_numpy(dtype=float, copy=True) * 1e3
            self.channel_cache = {}
            for col in self.current_data.columns:
                try:
                    self.channel_cache[col] = self.current_data[col].to_numpy(
                        dtype=float, copy=True
                    )
                except Exception:
                    continue
            self._refresh_channel_combos()
            self.update_plot(fast=False)
        except Exception as e:
            self.stats_text.setText(f"Error loading file: {e}")

    def on_file_changed(self, idx):
        """Handle file selection change."""
        if idx >= 0:
            self.load_file(idx)

    def prev_file(self):
        """Load previous file."""
        if self.current_file_idx > 0:
            self.load_file(self.current_file_idx - 1)

    def next_file(self):
        """Load next file."""
        if self.current_file_idx < len(self.data_files) - 1:
            self.load_file(self.current_file_idx + 1)

    def get_current_params(self):
        """Get current parameter values."""
        return [self.param_spinboxes[spec.key].value() for spec in self.param_specs]

    def get_batch_params(self):
        """Get batch fit initial parameters from shared controls."""
        return self.get_current_params()

    def reset_params(self):
        """Reset parameters to defaults."""
        for idx, spec in enumerate(self.param_specs):
            self.param_spinboxes[spec.key].setValue(self.defaults[idx])

    def do_full_update(self):
        """Perform a complete update including stats."""
        self.update_plot(fast=False)

    def browse_directory(self):
        """Browse for a data source root (directory, ZIP, or CSV) in one dialog."""
        current_path = Path(self.current_dir).expanduser()
        start_dir = (
            current_path.parent if current_path.is_file() else current_path
        )
        if not start_dir.exists():
            start_dir = Path.cwd()

        dialog = QFileDialog(self, "Select Data Source", str(start_dir))
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setNameFilters(
            [
                "Data Sources (*.zip *.csv)",
                "ZIP Archives (*.zip)",
                "CSV Files (*.csv)",
                "All Files (*.*)",
            ]
        )

        if not dialog.exec():
            return

        selected = dialog.selectedFiles()
        if not selected:
            return

        chosen_path = Path(selected[0]).expanduser()
        if not chosen_path.exists():
            self.stats_text.append(f"Selected path does not exist: {chosen_path}")
            return

        if chosen_path.is_dir() or chosen_path.suffix.lower() in {".zip", ".csv"}:
            self.current_dir = str(chosen_path)
            self.dir_input.setText(self.current_dir)
            self.load_files()
            return

        self.stats_text.append("Select a directory, ZIP archive, or CSV file.")

    def auto_fit(self):
        """Start auto-fit in a worker thread to keep GUI responsive."""
        if self.current_data is None:
            self.stats_text.append("No data loaded!")
            return

        if self.fit_thread is not None:
            self.stats_text.append("Auto-fit is already running.")
            return

        current_params = self.get_current_params()
        x_data = self._get_channel_data(self.x_channel)
        y_data = self._get_channel_data(self.y_channel)
        fit_slice = self.get_fit_slice(len(x_data))
        x_data = x_data[fit_slice]
        y_data = y_data[fit_slice]

        self.fit_thread = QThread(self)
        self.fit_worker = FitWorker(
            x_data,
            y_data,
            current_params,
            self.bounds,
            self.model_spec.function,
        )
        self.fit_worker.moveToThread(self.fit_thread)

        self.fit_thread.started.connect(self.fit_worker.run)
        self.fit_worker.finished.connect(self.on_fit_finished)
        self.fit_worker.failed.connect(self.on_fit_failed)
        self.fit_worker.cancelled.connect(self.on_fit_cancelled)

        self.auto_fit_btn.setEnabled(False)
        self.cancel_fit_btn.setEnabled(True)
        self.auto_fit_btn.setText("Fitting...")
        self.stats_text.append("\nAuto-fit started...")
        self.fit_thread.start()

    def on_fit_finished(self, popt, pcov, r2):
        """Handle successful fit completion."""
        self.last_popt = np.asarray(popt, dtype=float)
        self.last_pcov = np.asarray(pcov, dtype=float)
        self.last_fit_r2 = float(r2)
        self.last_full_r2 = None
        if self.current_data is not None:
            try:
                x_all = self._get_channel_data(self.x_channel)
                y_all = self._get_channel_data(self.y_channel)
                fitted_all = self.evaluate_model(x_all, self.last_popt)
                self.last_full_r2 = self._safe_r2(y_all, fitted_all)
            except Exception:
                self.last_full_r2 = None

        for idx, spec in enumerate(self.param_specs):
            self.param_spinboxes[spec.key].setValue(self.last_popt[idx])
        self.defaults = list(self.last_popt)

        full_r2_text = f"{self.last_full_r2:.6f}" if self.last_full_r2 is not None else "N/A"
        self.stats_text.append(
            f"✓ Auto-fit successful! R² (full trace) = {full_r2_text}"
        )
        summary = ", ".join(
            f"{spec.symbol}={self.last_popt[idx]:.4f}"
            for idx, spec in enumerate(self.param_specs)
        )
        self.stats_text.append(summary)
        self.update_plot()
        self.cleanup_fit_thread()

    def on_fit_failed(self, error_text):
        """Handle fit failures."""
        self.stats_text.append(f"✗ Auto-fit failed: {error_text}")
        self.stats_text.append(
            "Try manually adjusting parameters first, then retry auto-fit."
        )
        self.cleanup_fit_thread()

    def on_fit_cancelled(self):
        """Handle fit cancellation."""
        self.stats_text.append("Auto-fit cancelled.")
        self.cleanup_fit_thread()

    def cancel_auto_fit(self):
        """Request cancellation of an in-flight auto-fit."""
        if self.fit_worker is not None:
            self.fit_worker.request_cancel()
            self.stats_text.append("Auto-fit cancellation requested...")

    def cleanup_fit_thread(self):
        """Tear down worker/thread and restore button state."""
        if self.fit_thread is not None:
            self.fit_thread.quit()
            self.fit_thread.wait()
            self.fit_thread.deleteLater()
        if self.fit_worker is not None:
            self.fit_worker.deleteLater()
        self.fit_thread = None
        self.fit_worker = None
        self.auto_fit_btn.setEnabled(True)
        self.auto_fit_btn.setText("Auto Fit")
        self.cancel_fit_btn.setEnabled(False)

    def cleanup_batch_thread(self):
        if self.batch_thread is not None:
            self.batch_thread.quit()
            self.batch_thread.wait()
            self.batch_thread.deleteLater()
        if self.batch_worker is not None:
            self.batch_worker.deleteLater()
        self.batch_thread = None
        self.batch_worker = None
        self.batch_fit_in_progress = False
        self._batch_progress_done = 0
        self.run_batch_btn.setEnabled(True)
        self.run_batch_btn.setText(self.run_batch_btn_default_text)
        self.cancel_batch_btn.setEnabled(False)
        self.batch_status_label.hide()

    def _get_channel_data(self, channel_name):
        if channel_name in self.channel_cache:
            return self.channel_cache[channel_name]
        if self.current_data is None:
            raise ValueError("No data loaded.")
        if channel_name not in self.current_data.columns:
            raise KeyError(f"Channel '{channel_name}' not found in data.")
        values = self.current_data[channel_name].to_numpy(dtype=float, copy=True)
        self.channel_cache[channel_name] = values
        return values

    def _display_indices(self, n_points):
        if n_points <= 0:
            return np.asarray([], dtype=int)
        target = max(1000, int(self._display_target_points))
        stride = max(1, int(np.ceil(n_points / float(target))))
        return np.arange(0, n_points, stride, dtype=int)

    def _fit_mask(self, n_points, fit_slice):
        mask = np.zeros(int(n_points), dtype=bool)
        if n_points <= 0:
            return mask
        start = int(fit_slice.start) if fit_slice.start is not None else 0
        stop = int(fit_slice.stop) if fit_slice.stop is not None else n_points
        start = max(0, min(n_points, start))
        stop = max(start, min(n_points, stop))
        mask[start:stop] = True
        return mask

    def _ensure_plot_axes(self, show_residuals):
        if show_residuals == self._plot_has_residual_axis and hasattr(self, "ax"):
            return
        self.fig.clear()
        if show_residuals:
            grid = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            self.ax = self.fig.add_subplot(grid[0])
            self.ax_residual = self.fig.add_subplot(grid[1], sharex=self.ax)
            self.ax.tick_params(labelbottom=False)
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax_residual = None
        self._plot_has_residual_axis = bool(show_residuals)
        self._recreate_fit_region_selector()

    def _finite_min_max(self, *arrays):
        y_min = None
        y_max = None
        for arr in arrays:
            if arr is None:
                continue
            values = np.asarray(arr, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            cur_min = float(np.min(finite))
            cur_max = float(np.max(finite))
            y_min = cur_min if y_min is None else min(y_min, cur_min)
            y_max = cur_max if y_max is None else max(y_max, cur_max)

        if y_min is None or y_max is None:
            return (-1.0, 1.0)
        if np.isclose(y_min, y_max):
            pad = 1.0 if np.isclose(y_min, 0.0) else max(1e-6, abs(y_min) * 0.05)
            return (y_min - pad, y_max + pad)

        pad = (y_max - y_min) * 0.05
        if pad <= 0.0:
            pad = 1.0
        return (y_min - pad, y_max + pad)

    def _safe_r2(self, y_true, y_pred):
        return compute_r2(y_true, y_pred)

    def _apply_unique_legend(self, axis, loc="lower right"):
        handles, labels = axis.get_legend_handles_labels()
        unique_handles = []
        unique_labels = []
        seen = set()
        for handle, label in zip(handles, labels):
            if not label or label.startswith("_") or label in seen:
                continue
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
        if unique_handles:
            axis.legend(unique_handles, unique_labels, loc=loc)

    def update_plot(self, fast=False):
        """Update plot with current parameters.

        Args:
            fast: If True, skip expensive operations for smooth slider interaction
        """
        if self.current_data is None:
            return

        # Debounce full updates during slider movement
        if fast and not self.slider_active:
            # Use timer to batch rapid updates
            self.update_timer.stop()
            self.update_timer.start(50)  # 50ms debounce
            return

        try:
            params = self.get_current_params()

            x_data = self._get_channel_data(self.x_channel)
            y_data = self._get_channel_data(self.y_channel)
            n_points = len(x_data)
            if n_points == 0:
                return

            time_data = self.cached_time_data
            if time_data is None or len(time_data) != n_points:
                time_data = np.arange(n_points, dtype=float)

            fit_slice = self.get_fit_slice(n_points)
            fit_mask_full = self._fit_mask(n_points, fit_slice)
            display_idx = self._display_indices(n_points)
            if display_idx.size == 0:
                return

            time_display = time_data[display_idx]
            x_display = x_data[display_idx]
            y_display = y_data[display_idx]
            fit_mask_display = fit_mask_full[display_idx]

            fitted_display_full = self.evaluate_model(x_display, params)
            if self.extrapolate_unfitted:
                fitted_display = fitted_display_full
            else:
                fitted_display = np.where(fit_mask_display, fitted_display_full, np.nan)

            residuals_display_full = y_display - fitted_display_full
            if self.extrapolate_unfitted:
                residuals_display = residuals_display_full
            else:
                residuals_display = np.where(
                    fit_mask_display, residuals_display_full, np.nan
                )

            fit_start_t, fit_end_t, _, _ = self._fit_window_times(time_data, fit_slice)
            self._fit_window_bounds_ms = (fit_start_t, fit_end_t)
            show_residuals = self.show_residuals_cb.isChecked()

            if (
                fast
                and hasattr(self, "_plot_lines")
                and show_residuals == self._plot_has_residual_axis
            ):
                # Fast update: only update line data
                if "fitted" in self._plot_lines:
                    self._plot_lines["fitted"].set_ydata(fitted_display)
                if "residuals" in self._plot_lines and show_residuals:
                    self._plot_lines["residuals"].set_ydata(residuals_display)
                y_min, y_max = self._finite_min_max(y_display, fitted_display)
                self.ax.set_ylim(y_min, y_max)
                self.ax.set_xlim(time_data[0], time_data[-1])
                if show_residuals and self.ax_residual is not None:
                    r_min, r_max = self._finite_min_max(residuals_display)
                    self.ax_residual.set_ylim(r_min, r_max)
                self.canvas.draw_idle()
                return

            self._ensure_plot_axes(show_residuals)
            self.ax.clear()
            if self.ax_residual is not None:
                self.ax_residual.clear()

            self._plot_lines = {}
            self._draw_fit_window_overlay(time_data, fit_slice)
            y_color = "C0"
            x_color = "C1"
            outside_alpha = 0.30
            y_inside = np.where(fit_mask_display, y_display, np.nan)
            y_outside = np.where(fit_mask_display, np.nan, y_display)
            x_inside = np.where(fit_mask_display, x_display, np.nan)
            x_outside = np.where(fit_mask_display, np.nan, x_display)
            self.ax.plot(
                time_display,
                y_outside,
                color=y_color,
                linewidth=2,
                alpha=outside_alpha,
                label="_nolegend_",
            )
            self.ax.plot(
                time_display,
                y_inside,
                label=f"{self.y_channel} ({self.channels.get(self.y_channel,'Y')})",
                color=y_color,
                linewidth=2,
            )
            self.ax.plot(
                time_display,
                x_outside,
                color=x_color,
                alpha=outside_alpha,
                label="_nolegend_",
            )
            self.ax.plot(
                time_display,
                x_inside,
                label=f"{self.x_channel} ({self.channels.get(self.x_channel,'X')})",
                color=x_color,
                alpha=0.85,
            )

            (fitted_line,) = self.ax.plot(
                time_display,
                fitted_display,
                label="Fitted",
                color=FIT_CURVE_COLOR,
                linewidth=2,
            )
            self._plot_lines["fitted"] = fitted_line

            if show_residuals and self.ax_residual is not None:
                (residuals_line,) = self.ax_residual.plot(
                    time_display,
                    residuals_display,
                    label="Residuals",
                    color="black",
                    linestyle=":",
                    linewidth=1.4,
                )
                self._plot_lines["residuals"] = residuals_line
                self.ax_residual.axhline(
                    0.0, color="#6b7280", linewidth=1.0, alpha=0.6, linestyle="--"
                )

            # Calculate R² scores (skip during fast updates for smoothness)
            if not fast:
                x_fit = x_data[fit_slice]
                y_fit = y_data[fit_slice]
                fitted_fit = self.evaluate_model(x_fit, params)
                fitted_full = self.evaluate_model(x_data, params)
                fit_r2_value = self._safe_r2(y_fit, fitted_fit)
                full_r2_value = self._safe_r2(y_data, fitted_full)
                self._last_r2_fit = fit_r2_value
                self._last_r2_full = full_r2_value
            else:
                fit_r2_value = self._last_r2_fit
                full_r2_value = self._last_r2_full

            y_min, y_max = self._finite_min_max(y_display, fitted_display)
            self.ax.set_ylim(y_min, y_max)
            self._apply_unique_legend(self.ax, loc="lower right")
            self.ax.set_xlabel("" if show_residuals else "Time (ms)")
            self.ax.set_ylabel("Voltage (V)")
            self.ax.set_xlim(time_data[0], time_data[-1])
            self.ax.grid(True, alpha=0.3)
            if show_residuals and self.ax_residual is not None:
                r_min, r_max = self._finite_min_max(residuals_display)
                self.ax_residual.set_ylim(r_min, r_max)
                self.ax_residual.set_ylabel("Residual")
                self.ax_residual.set_xlabel("Time (ms)")
                self.ax_residual.grid(True, alpha=0.25)
                self._apply_unique_legend(self.ax_residual, loc="upper right")
            self._sync_fit_region_selector()
            self.ax.text(
                0.01,
                0.98,
                "Drag across the plot to set fit window",
                transform=self.ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="#7f1d1d",
            )

            self.canvas.draw_idle()

            # Update stats text
            if self.last_popt is not None and self.last_pcov is not None:
                sigma = np.sqrt(np.maximum(np.diag(self.last_pcov), 0.0))
                fit_r2_text = f"{fit_r2_value:.6f}" if fit_r2_value is not None else "N/A"
                full_r2_text = (
                    f"{full_r2_value:.6f}" if full_r2_value is not None else "N/A"
                )
                stats = (
                    f"R² (fit / full) = {fit_r2_text:.2f} / {full_r2_text:.2f}\n"
                )
                param_lines = [
                    f"{spec.symbol} = {self.last_popt[idx]:.6f} ± {sigma[idx]:.6f}"
                    for idx, spec in enumerate(self.param_specs)
                ]
                stats += "\n".join(param_lines)
                self.stats_text.setText(stats)
            else:
                self.stats_text.setText(
                    "No auto-fit results yet.\n\nClick 'Auto Fit' to optimize parameters."
                )
        except Exception as e:
            self.stats_text.setText(f"Error updating stats: {e}")

    def save_params(self):
        """Save current parameters as new defaults."""
        self.defaults = self.get_current_params()

    def select_batch_files(self):
        """Compatibility helper: batch now always uses the shared file list."""
        self._sync_batch_files_from_shared(sync_pattern=True)

    def run_batch_fit(self):
        """Run batch fitting using the shared file list."""
        if self.batch_thread is not None:
            self.stats_text.append("Batch fit is already running.")
            return
        self._sync_batch_files_from_shared(sync_pattern=False)
        if not self.batch_files:
            self.stats_text.append("No files available from the shared folder list.")
            return

        capture_config = self._resolve_batch_capture_config(show_errors=True)
        if capture_config is None:
            return

        current_params = self.get_batch_params()
        self._stop_thumbnail_render()
        self._batch_progress_done = 0
        self.batch_fit_in_progress = True

        existing_by_file = {row["file"]: row for row in self.batch_results}
        self.batch_results = []
        for source_index, file_path in enumerate(self.batch_files):
            existing = existing_by_file.get(file_path, {})
            captures = extract_captures(
                stem_for_file_ref(file_path),
                capture_config.regex,
                capture_config.defaults,
            )
            self.batch_results.append(
                {
                    "_source_index": source_index,
                    "file": file_path,
                    "captures": dict(captures or {}),
                    "params": None,
                    "r2": None,
                    "error": None,
                    "x_channel": self.x_channel,
                    "y_channel": self.y_channel,
                    "plot_full": existing.get("plot_full"),
                    "plot": existing.get("plot"),
                }
            )
        self.update_batch_table()

        self.batch_thread = QThread(self)
        self.batch_worker = BatchFitWorker(
            self.batch_files,
            current_params,
            self.bounds,
            capture_config.regex_pattern,
            capture_config.defaults,
            self.model_spec.function,
            self.x_channel,
            self.y_channel,
            self.fit_region_start_pct,
            self.fit_region_end_pct,
        )
        self.batch_worker.moveToThread(self.batch_thread)

        self.batch_thread.started.connect(self.batch_worker.run)
        self.batch_worker.progress.connect(self.on_batch_progress)
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.failed.connect(self.on_batch_failed)
        self.batch_worker.cancelled.connect(self.on_batch_cancelled)

        self.run_batch_btn.setEnabled(False)
        self.cancel_batch_btn.setEnabled(True)
        total = len(self.batch_files)
        self.run_batch_btn.setText(f"Run Batch (0/{total})")
        self.batch_status_label.hide()
        self.batch_thread.start()

    def on_batch_progress(self, completed, total, row):
        """Update progress label while batch is running."""
        self._batch_progress_done = int(completed)
        self.run_batch_btn.setText(f"Run Batch ({self._batch_progress_done}/{total})")
        row_index = row.get("_source_index")
        if row_index is None:
            for idx, existing_row in enumerate(self.batch_results):
                if existing_row.get("file") == row.get("file"):
                    row_index = idx
                    break
        if row_index is not None and 0 <= row_index < len(self.batch_results):
            existing = self.batch_results[row_index]
            if existing.get("plot_full") is not None and row.get("plot_full") is None:
                row["plot_full"] = existing["plot_full"]
            elif existing.get("plot") is not None and row.get("plot") is None:
                row["plot"] = existing["plot"]
            self.batch_results[row_index] = row
            table_row_idx = self._find_table_row_by_file(row["file"])
            if table_row_idx is not None:
                self.update_batch_table_row(table_row_idx, row)

    def on_batch_finished(self, results):
        """Populate table and thumbnails after batch fit finishes."""
        previous_by_file = {row["file"]: row for row in self.batch_results}
        ordered_results = sorted(
            list(results), key=lambda row: int(row.get("_source_index", 0))
        )
        self.batch_results = ordered_results
        for row in self.batch_results:
            existing = previous_by_file.get(row["file"])
            if existing and existing.get("plot_full") is not None:
                row["plot_full"] = existing["plot_full"]
            elif existing and existing.get("plot") is not None:
                row["plot"] = existing["plot"]
        self.update_batch_table()
        self._refresh_batch_analysis_if_run()
        self.stats_text.append("✓ Batch fit completed.")
        self.cleanup_batch_thread()
        self.queue_visible_thumbnail_render()

    def on_batch_failed(self, error_text):
        self.stats_text.append(f"✗ Batch fit failed: {error_text}")
        self.cleanup_batch_thread()

    def on_batch_cancelled(self):
        self.stats_text.append("Batch fit cancelled.")
        self.cleanup_batch_thread()

    def cancel_batch_fit(self):
        """Request cancellation of an in-flight batch fit."""
        if self.batch_worker is not None:
            self.batch_worker.request_cancel()
            self.stats_text.append("Batch cancellation requested...")

    def update_batch_table(self):
        """Refresh batch results table with captures and fit params."""
        if not self.batch_results:
            self.batch_table.setRowCount(0)
            self.batch_table.setColumnCount(0)
            return

        sorting_enabled = self.batch_table.isSortingEnabled()
        if sorting_enabled:
            self.batch_table.setSortingEnabled(False)
        try:
            columns = (
                ["Plot"]
                + ["File"]
                + self.batch_capture_keys
                + [spec.column_name for spec in self.param_specs]
                + ["R2", "Error"]
            )
            self.batch_table.setColumnCount(len(columns))
            self.batch_table.setHorizontalHeaderLabels(columns)
            self.batch_table.setRowCount(len(self.batch_results))
            self._apply_batch_row_heights()

            for row_idx, row in enumerate(self.batch_results):
                self.update_batch_table_row(row_idx, row, suspend_sorting=False)
        finally:
            if sorting_enabled:
                self.batch_table.setSortingEnabled(True)
        self.queue_visible_thumbnail_render()

    def update_batch_table_row(self, row_idx, row, suspend_sorting=True):
        """Update a single batch row in the results table."""
        sorting_enabled = suspend_sorting and self.batch_table.isSortingEnabled()
        if sorting_enabled:
            self.batch_table.setSortingEnabled(False)
        try:
            # Plot column (index 0)
            self._update_batch_plot_cell(row_idx, row)

            # File name column (index 1)
            file_name = display_name_for_file_ref(row["file"])
            file_item = NumericSortTableWidgetItem(file_name)
            file_item.setData(Qt.ItemDataRole.UserRole, row["file"])
            self.batch_table.setItem(row_idx, 1, file_item)

            # Capture columns (start at index 2)
            for col_idx, key in enumerate(self.batch_capture_keys, start=2):
                value = row.get("captures", {}).get(key, "")
                self.batch_table.setItem(
                    row_idx, col_idx, NumericSortTableWidgetItem(str(value))
                )

            # Parameter columns (start at 2 + len(batch_capture_keys))
            param_start = 2 + len(self.batch_capture_keys)
            params = row.get("params")
            for offset in range(len(self.param_specs)):
                if params and offset < len(params):
                    cell_text = f"{params[offset]:.6f}"
                else:
                    cell_text = ""
                self.batch_table.setItem(
                    row_idx,
                    param_start + offset,
                    NumericSortTableWidgetItem(cell_text),
                )
            r2_val = row.get("r2")
            if r2_val is not None:
                self.batch_table.setItem(
                    row_idx,
                    param_start + len(self.param_specs),
                    NumericSortTableWidgetItem(f"{r2_val:.6f}"),
                )
            else:
                self.batch_table.setItem(
                    row_idx,
                    param_start + len(self.param_specs),
                    NumericSortTableWidgetItem(""),
                )
            error_text = row.get("error") or ""
            self.batch_table.setItem(
                row_idx,
                param_start + len(self.param_specs) + 1,
                NumericSortTableWidgetItem(error_text),
            )
            self._apply_batch_row_error_background(row_idx, bool(error_text))
        finally:
            if sorting_enabled:
                self.batch_table.setSortingEnabled(True)

    def _apply_batch_row_error_background(self, row_idx, is_error):
        """Tint errored rows pale red; force white for non-error rows."""
        if row_idx < 0 or row_idx >= self.batch_table.rowCount():
            return
        color = QColor("#fee2e2") if is_error else QColor("#ffffff")
        for col_idx in range(self.batch_table.columnCount()):
            item = self.batch_table.item(row_idx, col_idx)
            if item is not None:
                item.setBackground(color)

    def _update_batch_plot_cell(self, row_idx, row):
        """Update only the plot thumbnail cell for a batch row."""
        thumb_item = QTableWidgetItem()
        pixmap = self._scaled_batch_plot(row)
        if pixmap is not None:
            thumb_item.setData(Qt.ItemDataRole.DecorationRole, pixmap)
        else:
            thumb_item.setData(Qt.ItemDataRole.DecorationRole, None)
        thumb_item.setData(Qt.ItemDataRole.UserRole, row["file"])  # Store file path
        self.batch_table.setItem(row_idx, 0, thumb_item)

    def _on_batch_table_cell_clicked(self, row_idx, col_idx):
        """Load selected batch row into the shared Plot tab."""
        if row_idx < 0:
            return

        clicked_item = self.batch_table.item(row_idx, col_idx)
        file_path = clicked_item.data(Qt.ItemDataRole.UserRole) if clicked_item else None
        if not file_path:
            for fallback_col in (1, 0):
                file_item = self.batch_table.item(row_idx, fallback_col)
                file_path = (
                    file_item.data(Qt.ItemDataRole.UserRole)
                    if file_item is not None
                    else None
                )
                if file_path:
                    break
        if not file_path:
            return

        if file_path not in self.data_files:
            self.data_files.append(file_path)
            self.file_combo.addItem(display_name_for_file_ref(file_path), file_path)
            self._sync_batch_files_from_shared(sync_pattern=False)

        file_idx = self.data_files.index(file_path)
        self.load_file(file_idx)
        self.tabs.setCurrentWidget(self.manual_tab)

    def _expand_file_column_for_selected_files(self):
        """Expand file column width to show the longest selected file name."""
        if not self.batch_files or self.batch_table.columnCount() < 2:
            return

        font_metrics = self.batch_table.fontMetrics()
        longest_width = 0
        for file_path in self.batch_files:
            file_name = display_name_for_file_ref(file_path)
            longest_width = max(longest_width, font_metrics.horizontalAdvance(file_name))

        # Account for text padding and small header/sort margin.
        target_width = longest_width + 36
        current_width = self.batch_table.columnWidth(1)
        if target_width > current_width:
            self.batch_table.setColumnWidth(1, target_width)

    def _visible_batch_row_indices(self):
        if not hasattr(self, "batch_table") or self.batch_table.rowCount() == 0:
            return []
        viewport = self.batch_table.viewport().rect()
        model = self.batch_table.model()
        visible = []
        for row_idx in range(self.batch_table.rowCount()):
            rect = self.batch_table.visualRect(model.index(row_idx, 0))
            if rect.isValid() and rect.intersects(viewport):
                visible.append(row_idx)
        return visible

    def queue_visible_thumbnail_render(self, *_args):
        if not self.batch_results or self.batch_fit_in_progress:
            return
        row_indices = self._visible_batch_row_indices()
        if not row_indices:
            row_indices = list(range(min(len(self.batch_results), 10)))
        self._start_thumbnail_render(row_indices=row_indices)

    def _start_thumbnail_render(self, row_indices=None):
        """Start background thread to render missing thumbnails."""
        if not self.batch_results:
            return

        if row_indices is None:
            candidate_rows = list(range(len(self.batch_results)))
        else:
            candidate_rows = sorted(
                {
                    int(idx)
                    for idx in row_indices
                    if 0 <= int(idx) < len(self.batch_results)
                }
            )
        candidate_rows = [
            idx
            for idx in candidate_rows
            if self.batch_results[idx].get("plot_full") is None
        ]
        if not candidate_rows:
            return

        if self.thumb_render_in_progress:
            self._pending_thumbnail_rows.update(candidate_rows)
            return

        self.thumb_render_in_progress = True
        self.thumb_thread = QThread(self)
        self.thumb_worker = ThumbnailRenderWorker(
            self.batch_results,
            self.model_spec.function,
            full_thumbnail_size=self._full_batch_thumbnail_size(),
            row_indices=candidate_rows,
        )
        self.thumb_worker.moveToThread(self.thumb_thread)

        self.thumb_thread.started.connect(self.thumb_worker.run)
        self.thumb_worker.progress.connect(self._on_thumbnail_rendered)
        self.thumb_worker.finished.connect(self._on_thumbnails_finished)
        self.thumb_worker.cancelled.connect(self._on_thumbnails_finished)
        self.thumb_thread.start()

    def _stop_thumbnail_render(self):
        """Stop thumbnail worker/thread if active."""
        if self.thumb_worker is not None:
            self.thumb_worker.request_cancel()
        if self.thumb_thread is not None:
            self.thumb_thread.quit()
            self.thumb_thread.wait(2000)
            self.thumb_thread.deleteLater()
        self.thumb_thread = None
        self.thumb_worker = None
        self.thumb_render_in_progress = False
        self._pending_thumbnail_rows.clear()

    def _on_thumbnail_rendered(self, idx, total, row_idx):
        """Update table cell when thumbnail is rendered."""
        if row_idx < len(self.batch_results):
            row = self.batch_results[row_idx]
            table_row_idx = self._find_table_row_by_file(row["file"])
            if table_row_idx is not None:
                self.update_batch_table_row(table_row_idx, row)

    def _on_thumbnails_finished(self):
        """Clean up thumbnail worker when finished."""
        if self.thumb_thread:
            self.thumb_thread.quit()
            self.thumb_thread.wait()
            self.thumb_thread.deleteLater()
        self.thumb_thread = None
        self.thumb_worker = None
        self.thumb_render_in_progress = False
        if self._pending_thumbnail_rows:
            queued = sorted(self._pending_thumbnail_rows)
            self._pending_thumbnail_rows.clear()
            self._start_thumbnail_render(row_indices=queued)

    def _batch_export_default_filename(self):
        pattern_text = self.regex_input.text().strip() if hasattr(self, "regex_input") else ""
        base = pattern_text or "batch_fit_results"
        if base.startswith("re:"):
            base = base[3:].strip() or "batch_fit_results"
        base = base.replace("*", "any")
        base = re.sub(r"\{([^{}]+)\}", r"\1", base)
        base = re.sub(r"\s+", "_", base)
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
        base = re.sub(r"_+", "_", base).strip("._-")
        if not base:
            base = "batch_fit_results"
        if len(base) > 100:
            base = base[:100].rstrip("._-")
        return f"{base}.csv"

    def export_batch_table(self):
        """Export batch table to CSV."""
        if not self.batch_results:
            self.stats_text.append("No batch results to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Batch Table",
            str(Path.cwd() / self._batch_export_default_filename()),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not file_path:
            return

        columns = (
            ["File"]
            + self.batch_capture_keys
            + [spec.column_name for spec in self.param_specs]
            + ["R2", "Error"]
        )

        try:
            with open(file_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                for row in self.batch_results:
                    file_name = display_name_for_file_ref(row["file"])
                    captures = row.get("captures", {})
                    params = row.get("params") or [""] * len(self.param_specs)
                    r2_val = row.get("r2")
                    error_text = row.get("error") or ""
                    writer.writerow(
                        [file_name]
                        + [captures.get(key, "") for key in self.batch_capture_keys]
                        + [
                            f"{val:.6f}" if isinstance(val, float) else val
                            for val in params
                        ]
                        + [f"{r2_val:.6f}" if r2_val is not None else ""]
                        + [error_text]
                    )
            self.stats_text.append(f"✓ Exported batch table to {file_path}")
        except Exception as exc:
            self.stats_text.append(f"✗ Export failed: {exc}")

    def update_batch_file_list(self):
        """Refresh shared-scope batch file status text."""
        count = len(self.batch_files)
        if hasattr(self, "batch_file_scope_label"):
            self.batch_file_scope_label.setText(
                f"Using all files in current folder/list: {count}"
            )

    def _set_batch_parse_feedback(self, message, is_error=False, tooltip=""):
        self.batch_parse_feedback_label.setText(message)
        self.batch_parse_feedback_label.setToolTip(tooltip)
        if is_error:
            self.batch_parse_feedback_label.setStyleSheet(
                "color: #b91c1c; font-weight: 600; padding: 1px 2px;"
            )
        else:
            self.batch_parse_feedback_label.setStyleSheet("")

    def _resolve_batch_capture_config(self, show_errors):
        pattern_text = self.regex_input.text().strip()
        try:
            return parse_capture_pattern(pattern_text)
        except Exception as exc:
            if show_errors:
                self._set_batch_parse_feedback(f"Error: {exc}", is_error=True)
                self.batch_status_label.setText(f"Error: {exc}")
                self.batch_status_label.show()
            return None

    def _update_batch_capture_feedback(self, config):
        if config.mode == "off":
            self._set_batch_parse_feedback(
                "Add {field} placeholders to extract filename columns."
            )
            return

        field_text = (
            ", ".join(self.batch_capture_keys) if self.batch_capture_keys else "none"
        )
        self._set_batch_parse_feedback(f"Fields: {field_text}")

    def _on_regex_changed(self):
        """Debounce filename pattern changes to avoid excessive updates."""
        self.regex_timer.stop()
        self.regex_timer.start(300)  # 300ms debounce

    def prepare_batch_preview(self):
        """Populate preview results before running batch fit."""
        self.regex_timer.stop()
        self._do_prepare_batch_preview()

    def _do_prepare_batch_preview(self):
        """Actually perform the batch preview update."""
        if not self.batch_files:
            self._stop_thumbnail_render()
            self.batch_match_count = 0
            self.batch_unmatched_files = []
            self.batch_capture_keys = []
            self.batch_results = []
            config = self._resolve_batch_capture_config(show_errors=True)
            if config is None:
                self.update_batch_table()
                self._refresh_batch_analysis_if_run()
                return
            self.batch_status_label.hide()
            self._update_batch_capture_feedback(config)
            self.update_batch_table()
            self._refresh_batch_analysis_if_run()
            return

        capture_config = self._resolve_batch_capture_config(show_errors=True)
        if capture_config is None:
            return

        self.batch_status_label.hide()

        existing_file_order = [row["file"] for row in self.batch_results]
        files_unchanged = existing_file_order == self.batch_files and bool(
            self.batch_results
        )

        self.batch_capture_keys = []
        self.batch_match_count = 0
        self.batch_unmatched_files = []

        if files_unchanged:
            for source_index, row in enumerate(self.batch_results):
                row["_source_index"] = source_index
                row["x_channel"] = self.x_channel
                row["y_channel"] = self.y_channel
                extracted = extract_captures(
                    stem_for_file_ref(row["file"]),
                    capture_config.regex,
                    capture_config.defaults,
                )
                captures = {}
                if extracted is None:
                    self.batch_unmatched_files.append(display_name_for_file_ref(row["file"]))
                else:
                    captures = extracted
                    if capture_config.mode != "off":
                        self.batch_match_count += 1
                    for key in captures.keys():
                        if key not in self.batch_capture_keys:
                            self.batch_capture_keys.append(key)
                row["captures"] = captures
        else:
            self._stop_thumbnail_render()

            # Build a map of existing results by file path to preserve params/r2/error/plot.
            existing_results = {row["file"]: row for row in self.batch_results}
            rebuilt_results = []

            for source_index, file_path in enumerate(self.batch_files):
                captures = {}
                extracted = extract_captures(
                    stem_for_file_ref(file_path),
                    capture_config.regex,
                    capture_config.defaults,
                )
                if extracted is None:
                    self.batch_unmatched_files.append(display_name_for_file_ref(file_path))
                else:
                    captures = extracted
                    if capture_config.mode != "off":
                        self.batch_match_count += 1
                    for key in captures.keys():
                        if key not in self.batch_capture_keys:
                            self.batch_capture_keys.append(key)

                existing = existing_results.get(file_path)
                existing_plot_full = None
                if existing:
                    existing_plot_full = (
                        existing.get("plot_full")
                        or existing.get("plot")
                        or existing.get("thumbnail")
                    )
                rebuilt_results.append(
                    {
                        "_source_index": source_index,
                        "file": file_path,
                        "captures": captures,
                        "params": existing["params"] if existing else None,
                        "r2": existing["r2"] if existing else None,
                        "error": existing["error"] if existing else None,
                        "x_channel": self.x_channel,
                        "y_channel": self.y_channel,
                        "plot_full": existing_plot_full,
                    }
                )

            self.batch_results = rebuilt_results

        self._update_batch_capture_feedback(capture_config)
        self.update_batch_table()
        self._refresh_batch_analysis_if_run()
        if any(
            row.get("plot_full") is None and row.get("plot") is None
            for row in self.batch_results
        ):
            self.queue_visible_thumbnail_render()

    def _refresh_batch_analysis_if_run(self):
        if not hasattr(self, "analysis_source_combo"):
            return
        if self.analysis_source_combo.currentData() == "run":
            self._refresh_batch_analysis_data(preserve_selection=True)

    def closeEvent(self, event):
        """Ensure worker thread is stopped before closing."""
        if self.fit_worker is not None:
            self.fit_worker.request_cancel()
        if self.fit_thread is not None:
            self.fit_thread.quit()
            self.fit_thread.wait(2000)
        if self.batch_worker is not None:
            self.batch_worker.request_cancel()
        if self.batch_thread is not None:
            self.batch_thread.quit()
            self.batch_thread.wait(2000)
        if self.thumb_worker is not None:
            self.thumb_worker.request_cancel()
        if self.thumb_thread is not None:
            self.thumb_thread.quit()
            self.thumb_thread.wait(2000)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ManualFitGUI()
    window.show()
    sys.exit(app.exec())
