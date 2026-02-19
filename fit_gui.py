#!/usr/bin/env python3
"""
Manual Curve Fitting GUI for MI Model
Allows manual adjustment of parameters for failed automatic fits.
"""

import sys
import glob
import re
import csv
from io import BytesIO
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
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QShortcut, QKeySequence, QPalette, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# use Qt5Agg backend for better performance
from matplotlib.pyplot import switch_backend


def mi_model(x, a, b, phi, d):
    """MI model: Voltage = |A*sin(B*Sig gen Output + phi_0) + D|^2"""
    return np.abs(a * np.sin(b * x + np.pi * phi) + d) ** 2


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
    )
}
ACTIVE_MODEL = FIT_MODELS["mi_abs_squared"]

# Initialize backend after model definitions.
switch_backend("Qt5Agg")


@dataclass(frozen=True)
class CapturePatternConfig:
    mode: str
    regex_pattern: str
    regex: Optional[Pattern[str]]


_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _template_to_regex(template_text: str) -> str:
    """Convert a simple filename template into a regex with named captures."""
    if not template_text:
        raise ValueError("Template is empty.")

    pieces = ["^"]
    seen_fields = set()
    idx = 0
    length = len(template_text)

    while idx < length:
        char = template_text[idx]
        if char == "{":
            end = template_text.find("}", idx + 1)
            if end < 0:
                raise ValueError("Missing closing '}' in template.")
            field_name = template_text[idx + 1 : end].strip()
            if not _FIELD_NAME_RE.fullmatch(field_name):
                raise ValueError(
                    f"Invalid field name '{field_name}'. Use letters, numbers, underscore."
                )
            if field_name in seen_fields:
                raise ValueError(f"Duplicate field name '{field_name}'.")
            pieces.append(f"(?P<{field_name}>.+?)")
            seen_fields.add(field_name)
            idx = end + 1
            continue
        if char == "}":
            raise ValueError("Unexpected '}' in template.")
        if char == "*":
            pieces.append(".*")
            idx += 1
            continue

        literal_start = idx
        while idx < length and template_text[idx] not in "{}*":
            idx += 1
        pieces.append(re.escape(template_text[literal_start:idx]))

    pieces.append("$")
    return "".join(pieces)


def parse_capture_pattern(pattern_text: str) -> CapturePatternConfig:
    """Parse capture input as simple template or advanced regex."""
    text = pattern_text.strip()
    if not text:
        return CapturePatternConfig(mode="off", regex_pattern="", regex=None)

    if text.startswith("re:"):
        regex_pattern = text[3:].strip()
        if not regex_pattern:
            raise ValueError("Regex mode requires a pattern after 're:'.")
        regex = re.compile(regex_pattern)
        return CapturePatternConfig(
            mode="regex", regex_pattern=regex_pattern, regex=regex
        )

    if "(?P<" in text:
        regex = re.compile(text)
        return CapturePatternConfig(mode="regex", regex_pattern=text, regex=regex)

    regex_pattern = _template_to_regex(text)
    regex = re.compile(regex_pattern)
    return CapturePatternConfig(
        mode="template", regex_pattern=regex_pattern, regex=regex
    )


def extract_captures(
    stem: str, regex: Optional[Pattern[str]]
) -> Optional[Dict[str, str]]:
    """Extract captures from a filename stem, supporting named and positional groups."""
    if regex is None:
        return {}
    match = regex.search(stem)
    if not match:
        return None

    captures = match.groupdict()
    if captures:
        return captures

    groups = match.groups()
    if groups:
        return {f"group_{idx + 1}": value for idx, value in enumerate(groups)}

    return {"match": match.group(0)}


def render_batch_thumbnail(row, model_func, full_thumbnail_size=(468, 312)):
    """Render a row thumbnail pixmap, including fitted curve when available."""
    try:
        data = read_csv(row["file"], skiprows=13, header=0)
        time_data = data["TIME"].to_numpy() * 1e3
        ch2_data = data["CH2"].to_numpy()
        ch3_data = data["CH3"].to_numpy()

        target_width = max(24, int(full_thumbnail_size[0]))
        target_height = max(24, int(full_thumbnail_size[1]))
        render_dpi = 180
        fig = Figure(
            figsize=(target_width / render_dpi, target_height / render_dpi),
            dpi=render_dpi,
        )
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16)
        ax = fig.add_subplot(111)
        ax.plot(time_data, ch2_data, linewidth=1.25, color="C0")
        ax.plot(time_data, ch3_data, linewidth=1.25, alpha=0.45, color="C1")

        params = row.get("params")
        if params:
            fitted_ch2 = model_func(ch3_data, *params)
            ax.plot(time_data, fitted_ch2, linewidth=1.25, color="C2")

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
            p_curr = self.p0
            popt = None
            pcov = None

            # Run in short chunks so cancellation can be checked between chunks.
            for _ in range(100):
                if self.cancel_requested:
                    self.cancelled.emit()
                    return

                popt, pcov = curve_fit(
                    self.model_func,
                    self.x_data,
                    self.y_data,
                    p0=p_curr,
                    bounds=self.bounds,
                    method="trf",
                    maxfev=200,
                )

                if np.allclose(popt, p_curr, rtol=1e-7, atol=1e-9):
                    break
                p_curr = popt

            if self.cancel_requested:
                self.cancelled.emit()
                return

            fitted = self.model_func(self.x_data, *popt)
            r2 = float(r2_score(self.y_data, fitted))
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
        model_func,
        full_thumbnail_size=(468, 312),
    ):
        super().__init__()
        self.file_paths = list(file_paths)
        self.p0 = np.asarray(p0, dtype=float)
        self.bounds = bounds
        self.regex = re.compile(regex_pattern) if regex_pattern else None
        self.model_func = model_func
        self.full_thumbnail_size = full_thumbnail_size
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    @pyqtSlot()
    def run(self):
        results = []
        try:
            total = len(self.file_paths)
            for idx, file_path in enumerate(self.file_paths):
                if self.cancel_requested:
                    self.cancelled.emit()
                    return

                row = {
                    "file": file_path,
                    "captures": {},
                    "params": None,
                    "r2": None,
                    "error": None,
                }

                captures = extract_captures(Path(file_path).stem, self.regex)
                if captures:
                    row["captures"] = captures

                try:
                    data = read_csv(file_path, skiprows=13, header=0)
                    x_data = data["CH3"].to_numpy(dtype=float, copy=True)
                    y_data = data["CH2"].to_numpy(dtype=float, copy=True)

                    popt, _ = curve_fit(
                        self.model_func,
                        x_data,
                        y_data,
                        p0=self.p0,
                        bounds=self.bounds,
                        method="trf",
                        maxfev=500,
                    )

                    fitted = self.model_func(x_data, *popt)
                    row["params"] = [float(x) for x in popt]
                    row["r2"] = float(r2_score(y_data, fitted))
                except Exception as exc:
                    row["error"] = str(exc)

                # Build/update thumbnail immediately so the table updates per-file.
                row["plot_full"] = render_batch_thumbnail(
                    row,
                    self.model_func,
                    full_thumbnail_size=self.full_thumbnail_size,
                )

                results.append(row)
                self.progress.emit(idx + 1, total, row)

            if self.cancel_requested:
                self.cancelled.emit()
                return

            self.finished.emit(results)
        except Exception as exc:
            if self.cancel_requested:
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))


class ThumbnailRenderWorker(QObject):
    progress = pyqtSignal(int, int, object)
    finished = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, batch_results, model_func, full_thumbnail_size=(468, 312)):
        super().__init__()
        self.batch_results = batch_results
        self.model_func = model_func
        self.full_thumbnail_size = full_thumbnail_size
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    @pyqtSlot()
    def run(self):
        try:
            total = len(self.batch_results)
            for idx, row in enumerate(self.batch_results):
                if self.cancel_requested:
                    self.cancelled.emit()
                    return

                pixmap = self.render_thumbnail(row)
                row["plot_full"] = pixmap
                self.progress.emit(idx + 1, total, idx)

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
        self.batch_param_spinboxes = {}

        self.setWindowTitle(f"Curve Fitting ({self.model_spec.display_name})")
        self.setGeometry(100, 100, 900, 640)

        self.data_files = []
        self.current_file_idx = 0
        self.current_data = None
        self.channels = {"CH2": "MI output voltage", "CH3": "Sig Gen Output"}
        self.last_popt = None
        self.last_pcov = None
        self.last_fit_r2 = None
        self.fit_thread = None
        self.fit_worker = None
        self.batch_thread = None
        self.batch_worker = None
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
        self.regex_timer = QTimer()
        self.regex_timer.setSingleShot(True)
        self.regex_timer.timeout.connect(self._do_prepare_batch_preview)
        self.thumb_thread = None
        self.thumb_worker = None
        self.thumb_render_in_progress = False
        self._preview_file_path = None

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
        self.cached_ch2_data = None
        self.cached_ch3_data = None

        # Current directory
        self.current_dir = "./measurements_18-02-26/shaking_functions/"

        # Create central widget
        central_widget = QWidget()
        central_widget.setObjectName("root")
        self.setCentralWidget(central_widget)
        self._enforce_light_mode()
        self._apply_compact_ui_defaults()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.manual_tab = QWidget()
        self.batch_tab = QWidget()
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.manual_tab, "Manual Fit")
        self.tabs.addTab(self.batch_tab, "Batch Fit")
        self.tabs.addTab(self.analysis_tab, "Batch Analysis")

        manual_layout = QVBoxLayout(self.manual_tab)
        manual_layout.setContentsMargins(6, 6, 6, 6)
        manual_layout.setSpacing(4)

        # File selection (compact)
        self.create_file_frame(manual_layout)

        # Plot frame
        self.create_plot_frame(manual_layout)

        # Parameters and Stats side by side
        params_stats_layout = QHBoxLayout()
        params_stats_layout.setSpacing(6)
        self.create_parameters_frame(params_stats_layout)
        self.create_stats_frame(params_stats_layout)
        params_stats_layout.setStretch(0, 3)
        params_stats_layout.setStretch(1, 2)
        manual_layout.addLayout(params_stats_layout)

        batch_layout = QVBoxLayout(self.batch_tab)
        batch_layout.setContentsMargins(6, 6, 6, 6)
        batch_layout.setSpacing(6)
        self.create_batch_controls_frame(batch_layout)
        self.create_batch_results_frame(batch_layout)
        self.create_thumbnails_frame(batch_layout)

        analysis_layout = QVBoxLayout(self.analysis_tab)
        analysis_layout.setContentsMargins(6, 6, 6, 6)
        analysis_layout.setSpacing(6)
        self.create_batch_analysis_frame(analysis_layout)
        analysis_layout.addStretch()

        self.preview_close_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.preview_close_shortcut.activated.connect(self._close_preview_panel)

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
        """Create compact file selection section."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        # Directory row
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(4)
        browse_dir_btn = self._make_compact_tool_button(
            "📁", "Browse Directory", self.browse_directory
        )
        dir_layout.addWidget(browse_dir_btn)

        self.dir_input = QLineEdit(self.current_dir)
        self.dir_input.setReadOnly(True)
        dir_layout.addWidget(self.dir_input, 2)

        # File row
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        dir_layout.addWidget(self.file_combo, 2)

        # Browse file button
        browse_file_btn = self._make_compact_tool_button(
            "📄", "Browse File", self.browse_file
        )
        dir_layout.addWidget(browse_file_btn)

        # Navigation buttons
        prev_btn = self._make_compact_tool_button("◀", "Previous File", self.prev_file)
        dir_layout.addWidget(prev_btn)

        next_btn = self._make_compact_tool_button("▶", "Next File", self.next_file)
        dir_layout.addWidget(next_btn)

        layout.addLayout(dir_layout)

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
        self.canvas = FigureCanvas(self.fig)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.16)

        # Add matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(14, 14))
        self.toolbar.setMaximumHeight(28)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

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

        self.param_spinboxes.clear()
        self.param_sliders.clear()
        for idx, spec in enumerate(self.param_specs):
            control_layout, spinbox, slider = self.create_param_control(
                spec,
                self.defaults[idx],
            )
            self.param_spinboxes[spec.key] = spinbox
            self.param_sliders[spec.key] = slider
            layout.addLayout(control_layout)

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

    def create_batch_param_control(self, spec, default_val):
        """Create batch parameter row with label + spinbox (no slider)."""
        layout = QHBoxLayout()
        layout.setSpacing(4)

        layout.addWidget(self._create_param_label(spec, width=130))

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
        spinbox.setToolTip(spec.description)
        layout.addWidget(spinbox)
        layout.addStretch()

        return layout, spinbox

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

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_params)
        btn_layout.addWidget(save_btn)

        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_fit)
        btn_layout.addWidget(export_btn)

        layout.addLayout(btn_layout)

        # Stats text display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(160)
        layout.addWidget(self.stats_text)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_batch_controls_frame(self, parent_layout):
        """Create batch controls and selected file display."""
        group = QGroupBox("")
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        split_layout = QHBoxLayout()
        split_layout.setSpacing(12)

        params_layout = QVBoxLayout()
        params_layout.setSpacing(4)
        sync_btn = QPushButton("Sync from Manual")
        sync_btn.setToolTip("Copy parameters from Manual Fit tab")
        sync_btn.clicked.connect(self.sync_batch_params_from_manual)

        params_header_row = QHBoxLayout()
        params_header_row.setSpacing(4)
        params_header_row.addWidget(QLabel("Batch Parameters"))
        params_header_row.addStretch()
        params_header_row.addWidget(sync_btn)
        params_layout.addLayout(params_header_row)

        self.batch_param_spinboxes.clear()
        for idx, spec in enumerate(self.param_specs):
            param_row, spinbox = self.create_batch_param_control(
                spec, self.defaults[idx]
            )
            params_layout.addLayout(param_row)
            self.batch_param_spinboxes[spec.key] = spinbox

        file_layout = QVBoxLayout()
        file_layout.setSpacing(4)

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
            "Advanced regex: prefix with re: or use (?P<name>...) groups."
        )
        self.regex_input.textChanged.connect(self._on_regex_changed)
        regex_layout.addWidget(self.regex_input)
        file_layout.addLayout(regex_layout)

        self.batch_parse_feedback_label = QLabel(
            "Use {field} placeholders to extract columns."
        )
        self.batch_parse_feedback_label.setObjectName("statusLabel")
        file_layout.addWidget(self.batch_parse_feedback_label)

        self.select_batch_btn = QPushButton("")
        self.select_batch_btn.clicked.connect(self.select_batch_files)
        self.update_batch_file_list()
        file_layout.addWidget(self.select_batch_btn)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(4)
        actions_row.addWidget(self.run_batch_btn)
        actions_row.addWidget(export_table_btn)
        file_layout.addLayout(actions_row)

        self.batch_status_label = QLabel("")
        self.batch_status_label.setObjectName("statusLabel")
        self.batch_status_label.hide()
        file_layout.addWidget(self.batch_status_label)

        split_layout.addLayout(params_layout, 3)
        split_layout.addLayout(file_layout, 2)
        layout.addLayout(split_layout)

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
        self.analysis_fit_line_btn = QPushButton("Best-Fit Lines")
        self.analysis_fit_line_btn.setCheckable(True)
        self.analysis_fit_line_btn.setChecked(True)
        self.analysis_fit_line_btn.toggled.connect(self.update_batch_analysis_plot)
        controls_row.addWidget(self.analysis_fit_line_btn)
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
            record = {"File": Path(row["file"]).name}
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

    def _refresh_batch_analysis_data(self, preserve_selection):
        source = self.analysis_source_combo.currentData()
        if source == "csv":
            records = list(self.analysis_csv_records)
            if records:
                file_name = (
                    Path(self.analysis_csv_path).name
                    if self.analysis_csv_path
                    else "CSV"
                )
                self.analysis_status_label.setText(
                    f"Loaded CSV: {file_name} ({len(records)} rows)."
                )
            else:
                self.analysis_status_label.setText("Loaded CSV: no rows.")
        else:
            records = self._extract_analysis_records_from_batch()
            self.analysis_status_label.setText(
                f"Using completed batch results ({len(records)} rows)."
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
        show_fit_lines = self.analysis_fit_line_btn.isChecked()

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
            order = np.argsort(x_plot)
            x_sorted = x_plot[order]
            y_sorted = y_plot[order]
            color = f"C{idx % 10}"
            target_ax = axes[idx] if len(axes) > 1 else axes[0]
            target_ax.scatter(x_sorted, y_sorted, s=26, color=color, label=param_name)

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
                target_ax.grid(True, alpha=0.25)
                target_ax.legend(loc="best", fontsize=8)

        if not plotted_any:
            self._show_analysis_message(
                "No finite X/Y pairs available for the selected fields."
            )
            return

        if len(axes) == 1:
            axes[0].set_ylabel("Parameter Value")
            axes[0].legend(loc="best", fontsize=8)
            axes[0].grid(True, alpha=0.3)

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

    def create_thumbnails_frame(self, parent_layout):
        """Create click-to-open preview panel for selected batch row."""
        self.preview_group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        header_layout = QHBoxLayout()
        self.preview_title_label = QLabel("Click a table row to preview")
        header_layout.addWidget(self.preview_title_label)
        header_layout.addStretch()
        self.preview_close_btn = QPushButton("Close (Esc)")
        self.preview_close_btn.clicked.connect(self._close_preview_panel)
        header_layout.addWidget(self.preview_close_btn)
        layout.addLayout(header_layout)

        self.preview_fig = Figure(figsize=(10, 4), dpi=100)
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.14)
        self.preview_canvas = FigureCanvas(self.preview_fig)
        layout.addWidget(self.preview_canvas)

        self.preview_group.setLayout(layout)
        self.preview_group.hide()
        parent_layout.addWidget(self.preview_group)

    def load_files(self):
        """Load all CSV files from directory."""
        dir_path = self.current_dir
        self.data_files = sorted(glob.glob(dir_path + "*.csv"))

        if not self.data_files:
            self.stats_text.setText("No CSV files found in directory!")
            return

        # Populate combo box
        self.file_combo.clear()
        for file in self.data_files:
            self.file_combo.addItem(Path(file).name, file)

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

        try:
            file_path = self.data_files[idx]
            self.current_data = read_csv(file_path, skiprows=13, header=0)
            # Cache data for faster updates
            self.cached_time_data = self.current_data["TIME"].to_numpy() * 1e3
            self.cached_ch2_data = self.current_data["CH2"].to_numpy()
            self.cached_ch3_data = self.current_data["CH3"].to_numpy()
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
        """Get batch fit initial parameters."""
        return [
            self.batch_param_spinboxes[spec.key].value() for spec in self.param_specs
        ]

    def sync_batch_params_from_manual(self):
        """Copy parameters from manual tab to batch tab."""
        for spec in self.param_specs:
            self.batch_param_spinboxes[spec.key].setValue(
                self.param_spinboxes[spec.key].value()
            )

    def reset_params(self):
        """Reset parameters to defaults."""
        for idx, spec in enumerate(self.param_specs):
            self.param_spinboxes[spec.key].setValue(self.defaults[idx])

    def do_full_update(self):
        """Perform a complete update including stats."""
        self.update_plot(fast=False)

    def browse_directory(self):
        """Browse for a directory containing CSV files."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory", str(Path.cwd())
        )
        if dir_path:
            self.current_dir = dir_path + "/"
            self.dir_input.setText(self.current_dir)
            self.load_files()

    def browse_file(self):
        """Browse for a single CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            str(Path.cwd()),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if file_path:
            # Add this file to the list if not already there
            if file_path not in self.data_files:
                self.data_files.append(file_path)
                self.file_combo.addItem(Path(file_path).name, file_path)

            # Select this file
            idx = self.data_files.index(file_path)
            self.load_file(idx)

    def auto_fit(self):
        """Start auto-fit in a worker thread to keep GUI responsive."""
        if self.current_data is None:
            self.stats_text.append("No data loaded!")
            return

        if self.fit_thread is not None:
            self.stats_text.append("Auto-fit is already running.")
            return

        data = self.current_data
        current_params = self.get_current_params()
        x_data = data["CH3"].to_numpy(dtype=float, copy=True)
        y_data = data["CH2"].to_numpy(dtype=float, copy=True)

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
        self.auto_fit_btn.setText("Fitting...")
        self.stats_text.append("\nAuto-fit started...")
        self.fit_thread.start()

    def on_fit_finished(self, popt, pcov, r2):
        """Handle successful fit completion."""
        self.last_popt = np.asarray(popt, dtype=float)
        self.last_pcov = np.asarray(pcov, dtype=float)
        self.last_fit_r2 = float(r2)

        for idx, spec in enumerate(self.param_specs):
            self.param_spinboxes[spec.key].setValue(self.last_popt[idx])
        self.defaults = list(self.last_popt)

        self.stats_text.append(f"✓ Auto-fit successful! R² = {self.last_fit_r2:.6f}")
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

    def cleanup_batch_thread(self):
        if self.batch_thread is not None:
            self.batch_thread.quit()
            self.batch_thread.wait()
            self.batch_thread.deleteLater()
        if self.batch_worker is not None:
            self.batch_worker.deleteLater()
        self.batch_thread = None
        self.batch_worker = None
        self.run_batch_btn.setEnabled(True)
        self.run_batch_btn.setText(self.run_batch_btn_default_text)
        self.batch_status_label.hide()

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

            # Use cached data for faster computation
            time_data = self.cached_time_data
            ch2_data = self.cached_ch2_data
            ch3_data = self.cached_ch3_data

            # Clear plot only when necessary
            if not fast or not hasattr(self, "_plot_lines"):
                self.ax.clear()
                self._plot_lines = {}

            # Calculate fitted curve
            fitted_ch2 = self.evaluate_model(ch3_data, params)

            if fast and hasattr(self, "_plot_lines"):
                # Fast update: only update line data
                if "fitted" in self._plot_lines:
                    self._plot_lines["fitted"].set_ydata(fitted_ch2)
                y_min = float(np.min(fitted_ch2))
                y_max = float(np.max(fitted_ch2))
                y_min = min(y_min, float(np.min(ch2_data)), float(np.min(ch3_data)))
                y_max = max(y_max, float(np.max(ch2_data)), float(np.max(ch3_data)))
                if (
                    "residuals" in self._plot_lines
                    and self.show_residuals_cb.isChecked()
                ):
                    residuals = ch2_data - fitted_ch2
                    self._plot_lines["residuals"].set_visible(True)
                    self._plot_lines["residuals"].set_ydata(residuals)
                    res_min = float(np.min(residuals))
                    res_max = float(np.max(residuals))
                    y_min = min(y_min, res_min)
                    y_max = max(y_max, res_max)
                elif (
                    "residuals" in self._plot_lines
                    and not self.show_residuals_cb.isChecked()
                ):
                    self._plot_lines["residuals"].set_visible(False)
                pad = (y_max - y_min) * 0.05
                if pad == 0.0:
                    pad = 1.0
                self.ax.set_ylim(y_min - pad, y_max + pad)
                self.ax.set_xlim(time_data[0], time_data[-1])
                self.canvas.draw_idle()
                return
            else:
                # Full update: redraw everything
                self.ax.plot(
                    time_data, ch2_data, label=self.channels["CH2"], linewidth=2
                )
                self.ax.plot(
                    time_data,
                    ch3_data,
                    label=self.channels["CH3"],
                    color="gray",
                    alpha=0.5,
                )

                (fitted_line,) = self.ax.plot(
                    time_data,
                    fitted_ch2,
                    label="Fitted CH2",
                    linewidth=2,
                )
                self._plot_lines["fitted"] = fitted_line

                # Show residuals if enabled
                if self.show_residuals_cb.isChecked():
                    residuals = ch2_data - fitted_ch2
                    (residuals_line,) = self.ax.plot(
                        time_data,
                        residuals,
                        label="Residuals",
                        linestyle=":",
                        linewidth=1.5,
                    )
                    self._plot_lines["residuals"] = residuals_line

            # Calculate R² score (skip during fast updates for smoothness)
            if not fast:
                r2 = r2_score(ch2_data, fitted_ch2)
                self._last_r2 = r2
            else:
                r2 = getattr(self, "_last_r2", 0.0)

            self.ax.legend(loc="lower right")
            self.ax.set_xlabel("Time (ms)")
            self.ax.set_ylabel("Voltage (V)")
            self.ax.set_xlim(time_data[0], time_data[-1])
            self.ax.grid(True, alpha=0.3)

            self.canvas.draw()

            # Update stats text
            if self.last_popt is not None and self.last_pcov is not None:
                sigma = np.sqrt(np.maximum(np.diag(self.last_pcov), 0.0))
                fit_r2_text = (
                    f"{self.last_fit_r2:.6f}" if self.last_fit_r2 is not None else "N/A"
                )
                stats = f"R² = {fit_r2_text}\n\n"
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

    def export_fit(self):
        """Export fitted parameters to a text file."""
        try:
            params = self.get_current_params()
            filename = (
                Path(self.data_files[self.current_file_idx]).stem + "_fit_params.txt"
            )

            with open(filename, "w") as f:
                f.write(f"File: {Path(self.data_files[self.current_file_idx]).name}\n")
                f.write(f"Model: {self.model_spec.display_name}\n")
                f.write(f"Formula: {self.model_spec.formula_fallback}\n")
                for idx, spec in enumerate(self.param_specs):
                    f.write(f"{spec.symbol} ({spec.description}): {params[idx]:.6f}\n")

                if self.current_data is not None:
                    fitted_ch2 = self.evaluate_model(self.current_data["CH3"], params)
                    r2 = r2_score(self.current_data["CH2"], fitted_ch2)
                    f.write(f"R² Score: {r2:.6f}\n")

        except Exception:
            pass  # Silent fail

    def select_batch_files(self):
        """Select multiple CSV files for batch processing."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV Files",
            str(Path.cwd()),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not file_paths:
            return
        self.batch_files = file_paths
        self.update_batch_file_list()
        self.prepare_batch_preview()
        self._expand_file_column_for_selected_files()

    def run_batch_fit(self):
        """Run batch fitting on selected files."""
        if not self.batch_files:
            self.stats_text.append("No batch files selected!")
            return
        if self.batch_thread is not None:
            self.stats_text.append("Batch fit is already running.")
            return

        capture_config = self._resolve_batch_capture_config(show_errors=True)
        if capture_config is None:
            return

        current_params = self.get_batch_params()

        self.prepare_batch_preview()
        self._stop_thumbnail_render()

        self.batch_thread = QThread(self)
        self.batch_worker = BatchFitWorker(
            self.batch_files,
            current_params,
            self.bounds,
            capture_config.regex_pattern,
            self.model_spec.function,
            full_thumbnail_size=self._full_batch_thumbnail_size(),
        )
        self.batch_worker.moveToThread(self.batch_thread)

        self.batch_thread.started.connect(self.batch_worker.run)
        self.batch_worker.progress.connect(self.on_batch_progress)
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.failed.connect(self.on_batch_failed)
        self.batch_worker.cancelled.connect(self.on_batch_cancelled)

        self.run_batch_btn.setEnabled(False)
        total = len(self.batch_files)
        self.run_batch_btn.setText(f"Run Batch (0/{total})")
        self.batch_status_label.hide()
        self.batch_thread.start()

    def on_batch_progress(self, idx, total, row):
        """Update progress label while batch is running."""
        self.run_batch_btn.setText(f"Run Batch ({idx}/{total})")
        row_index = idx - 1
        if row_index < len(self.batch_results):
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
        self.batch_results = list(results)
        for row in self.batch_results:
            existing = previous_by_file.get(row["file"])
            if existing and existing.get("plot_full") is not None:
                row["plot_full"] = existing["plot_full"]
            elif existing and existing.get("plot") is not None:
                row["plot"] = existing["plot"]
        self.update_batch_table()
        self._refresh_batch_analysis_if_run()
        if self.batch_results and any(
            row.get("plot_full") is None and row.get("plot") is None
            for row in self.batch_results
        ):
            self._stop_thumbnail_render()
            self._start_thumbnail_render()
        self.stats_text.append("✓ Batch fit completed.")
        self.cleanup_batch_thread()

    def on_batch_failed(self, error_text):
        self.stats_text.append(f"✗ Batch fit failed: {error_text}")
        self.cleanup_batch_thread()

    def on_batch_cancelled(self):
        self.stats_text.append("Batch fit cancelled.")
        self.cleanup_batch_thread()

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

    def update_batch_table_row(self, row_idx, row, suspend_sorting=True):
        """Update a single batch row in the results table."""
        sorting_enabled = suspend_sorting and self.batch_table.isSortingEnabled()
        if sorting_enabled:
            self.batch_table.setSortingEnabled(False)
        try:
            # Plot column (index 0)
            self._update_batch_plot_cell(row_idx, row)

            # File name column (index 1)
            file_name = Path(row["file"]).name
            file_item = QTableWidgetItem(file_name)
            file_item.setData(Qt.ItemDataRole.UserRole, row["file"])
            self.batch_table.setItem(row_idx, 1, file_item)

            # Capture columns (start at index 2)
            for col_idx, key in enumerate(self.batch_capture_keys, start=2):
                value = row.get("captures", {}).get(key, "")
                self.batch_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

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
                    QTableWidgetItem(cell_text),
                )
            r2_val = row.get("r2")
            if r2_val is not None:
                self.batch_table.setItem(
                    row_idx,
                    param_start + len(self.param_specs),
                    QTableWidgetItem(f"{r2_val:.6f}"),
                )
            else:
                self.batch_table.setItem(
                    row_idx,
                    param_start + len(self.param_specs),
                    QTableWidgetItem(""),
                )
            error_text = row.get("error") or ""
            self.batch_table.setItem(
                row_idx,
                param_start + len(self.param_specs) + 1,
                QTableWidgetItem(error_text),
            )
            self._apply_batch_row_error_background(row_idx, bool(error_text))
        finally:
            if sorting_enabled:
                self.batch_table.setSortingEnabled(True)

    def _apply_batch_row_error_background(self, row_idx, is_error):
        """Tint errored rows pale red; clear tint for non-error rows."""
        if row_idx < 0 or row_idx >= self.batch_table.rowCount():
            return
        color = QColor("#fee2e2") if is_error else QColor()
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
        """Open selected row in the preview panel on click."""
        if col_idx != 0:
            return
        if row_idx < 0 or row_idx >= len(self.batch_results):
            return
        clicked_item = self.batch_table.item(row_idx, col_idx)
        file_path = clicked_item.data(Qt.ItemDataRole.UserRole) if clicked_item else None
        if not file_path:
            file_item = self.batch_table.item(row_idx, 0)
            file_path = file_item.data(Qt.ItemDataRole.UserRole) if file_item else None
        if not file_path:
            return
        row = next((entry for entry in self.batch_results if entry["file"] == file_path), None)
        if row is None:
            return
        self._open_preview_panel(row)

    def _expand_file_column_for_selected_files(self):
        """Expand file column width to show the longest selected file name."""
        if not self.batch_files or self.batch_table.columnCount() < 2:
            return

        font_metrics = self.batch_table.fontMetrics()
        longest_width = 0
        for file_path in self.batch_files:
            file_name = Path(file_path).name
            longest_width = max(longest_width, font_metrics.horizontalAdvance(file_name))

        # Account for text padding and small header/sort margin.
        target_width = longest_width + 36
        current_width = self.batch_table.columnWidth(1)
        if target_width > current_width:
            self.batch_table.setColumnWidth(1, target_width)

    def _close_preview_panel(self):
        """Hide preview panel."""
        self._preview_file_path = None
        if hasattr(self, "preview_group"):
            self.preview_group.hide()

    def _open_preview_panel(self, row):
        """Render full-size plot in the embedded preview panel."""
        file_path = row.get("file")
        if not file_path:
            return

        self._preview_file_path = file_path
        self.preview_title_label.setText(f"Preview: {Path(file_path).name}")
        self.preview_ax.clear()

        try:
            data = read_csv(file_path, skiprows=13, header=0)
            time_data = data["TIME"].to_numpy() * 1e3
            ch2_data = data["CH2"].to_numpy()
            ch3_data = data["CH3"].to_numpy()

            self.preview_ax.plot(
                time_data, ch2_data, label="CH2 (MI output)", linewidth=2
            )
            self.preview_ax.plot(
                time_data, ch3_data, label="CH3 (Sig Gen)", linewidth=2, alpha=0.5
            )

            params = row.get("params")
            if params:
                fitted_ch2 = self.evaluate_model(ch3_data, params)
                self.preview_ax.plot(time_data, fitted_ch2, label="Fitted", linewidth=2)

            r2_val = row.get("r2")
            error_text = row.get("error")
            if error_text:
                self.preview_ax.set_title(
                    f"Error: {error_text[:80]}", color="red", fontsize=12
                )
            elif r2_val is not None:
                color = "green" if r2_val > 0.95 else "orange"
                self.preview_ax.set_title(
                    f"R² = {r2_val:.6f}", color=color, fontsize=12
                )

            self.preview_ax.legend(loc="best", fontsize=10)
            self.preview_ax.set_xlabel("Time (ms)", fontsize=11)
            self.preview_ax.set_ylabel("Voltage (V)", fontsize=11)
            self.preview_ax.grid(True, alpha=0.3)
        except Exception as exc:
            self.preview_ax.text(
                0.5, 0.5, f"Plot error: {str(exc)}", ha="center", va="center"
            )
            self.preview_ax.set_axis_off()

        self.preview_canvas.draw_idle()
        self.preview_group.show()

    def _start_thumbnail_render(self):
        """Start background thread to render all thumbnails."""
        if self.thumb_render_in_progress or not self.batch_results:
            return

        self.thumb_render_in_progress = True
        self.thumb_thread = QThread(self)
        self.thumb_worker = ThumbnailRenderWorker(
            self.batch_results,
            self.model_spec.function,
            full_thumbnail_size=self._full_batch_thumbnail_size(),
        )
        self.thumb_worker.moveToThread(self.thumb_thread)

        self.thumb_thread.started.connect(self.thumb_worker.run)
        self.thumb_worker.progress.connect(self._on_thumbnail_rendered)
        self.thumb_worker.finished.connect(self._on_thumbnails_finished)
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

    def export_batch_table(self):
        """Export batch table to CSV."""
        if not self.batch_results:
            self.stats_text.append("No batch results to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Batch Table",
            str(Path.cwd() / "batch_fit_results.csv"),
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
                    file_name = Path(row["file"]).name
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
        """Refresh select button label with selected file count."""
        count = len(self.batch_files)
        self.select_batch_btn.setText(f"Select Files ({count} files)")

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
            self.batch_match_count = 0
            self.batch_unmatched_files = []
            config = self._resolve_batch_capture_config(show_errors=True)
            if config is None:
                self._close_preview_panel()
                self._refresh_batch_analysis_if_run()
                return
            self.batch_status_label.hide()
            self._update_batch_capture_feedback(config)
            self._close_preview_panel()
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
            for row in self.batch_results:
                extracted = extract_captures(
                    Path(row["file"]).stem, capture_config.regex
                )
                captures = {}
                if extracted is None:
                    self.batch_unmatched_files.append(Path(row["file"]).name)
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

            for file_path in self.batch_files:
                captures = {}
                extracted = extract_captures(Path(file_path).stem, capture_config.regex)
                if extracted is None:
                    self.batch_unmatched_files.append(Path(file_path).name)
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
                        "file": file_path,
                        "captures": captures,
                        "params": existing["params"] if existing else None,
                        "r2": existing["r2"] if existing else None,
                        "error": existing["error"] if existing else None,
                        "plot_full": existing_plot_full,
                    }
                )

            self.batch_results = rebuilt_results

        self._update_batch_capture_feedback(capture_config)
        self.update_batch_table()
        self._refresh_batch_analysis_if_run()
        if self._preview_file_path and not any(
            row["file"] == self._preview_file_path for row in self.batch_results
        ):
            self._close_preview_panel()
        if any(
            row.get("plot_full") is None and row.get("plot") is None
            for row in self.batch_results
        ):
            self._start_thumbnail_render()

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
