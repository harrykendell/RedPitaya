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
from typing import Callable, Sequence, Tuple
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
    return np.abs(a * np.sin(b * x + np.pi*phi) + d) ** 2


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
        formula_latex=r"\left|A\sin(Bx+\pi\times\phi)+D\right|^2",
        formula_fallback="|A·sin(B·x + φ) + D|²",
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

    def __init__(self, file_paths, p0, bounds, regex_pattern, model_func):
        super().__init__()
        self.file_paths = list(file_paths)
        self.p0 = np.asarray(p0, dtype=float)
        self.bounds = bounds
        self.regex = re.compile(regex_pattern) if regex_pattern else None
        self.model_func = model_func
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

                if self.regex:
                    match = self.regex.search(Path(file_path).stem)
                    if match:
                        row["captures"] = match.groupdict()

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

    def __init__(self, batch_results, model_func, thumbnail_size=(72, 48)):
        super().__init__()
        self.batch_results = batch_results
        self.model_func = model_func
        self.thumbnail_size = thumbnail_size
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
                row["plot"] = pixmap
                self.progress.emit(idx + 1, total, idx)

            if self.cancel_requested:
                self.cancelled.emit()
                return

            self.finished.emit()
        except Exception:
            self.finished.emit()

    def render_thumbnail(self, row):
        """Render a plot to a QPixmap for embedding in table."""
        try:
            data = read_csv(row["file"], skiprows=13, header=0)
            time_data = data["TIME"].to_numpy() * 1e3
            ch2_data = data["CH2"].to_numpy()
            ch3_data = data["CH3"].to_numpy()

            fig = Figure(figsize=(1.8, 1.0), dpi=80)
            fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16)
            ax = fig.add_subplot(111)
            ax.plot(time_data, ch2_data, linewidth=1, color="C0")
            ax.plot(time_data, ch3_data, linewidth=1, alpha=0.4, color="C1")

            params = row.get("params")
            if params:
                fitted_ch2 = self.model_func(ch3_data, *params)
                ax.plot(time_data, fitted_ch2, linewidth=1, color="C2")

            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.15)

            # Render to pixmap via matplotlib PNG export
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
            buf.seek(0)
            data_bytes = buf.getvalue()
            buf.close()

            pixmap = QPixmap()
            pixmap.loadFromData(data_bytes, "PNG")
            return pixmap.scaledToHeight(
                self.thumbnail_size[1], Qt.TransformationMode.SmoothTransformation
            )
        except Exception:
            pixmap = QPixmap(self.thumbnail_size[0], self.thumbnail_size[1])
            pixmap.fill(Qt.GlobalColor.white)
            return pixmap


class ManualFitGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_spec = ACTIVE_MODEL
        self.param_specs = list(self.model_spec.params)
        self.param_spinboxes = {}
        self.param_sliders = {}
        self.batch_param_spinboxes = {}

        self.setWindowTitle(f"Manual {self.model_spec.display_name} Curve Fitting")
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
        self.max_thumbnails = 8
        self.thumb_cols = 1
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
        self.tabs.addTab(self.manual_tab, "Manual Fit")
        self.tabs.addTab(self.batch_tab, "Batch Fit")

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
        self.show_residuals_cb.toggled.connect(
            lambda: self.update_plot(fast=False)
        )
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
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        regex_layout = QHBoxLayout()
        regex_layout.setSpacing(4)
        regex_layout.addWidget(QLabel("Regex:"))
        self.regex_input = QLineEdit(r"data_(?P<freq>\d+Hz)_(?P<idx>\d{3})_ALL")
        self.regex_input.setToolTip("Named groups become table columns")
        self.regex_input.textChanged.connect(self._on_regex_changed)
        regex_layout.addWidget(self.regex_input)
        layout.addLayout(regex_layout)

        params_layout = QHBoxLayout()
        params_layout.setSpacing(4)
        self.batch_param_spinboxes.clear()
        for idx, spec in enumerate(self.param_specs):
            params_layout.addWidget(self._create_param_label(spec, width=120))
            spinbox = QDoubleSpinBox()
            spinbox.setMinimum(spec.min_value)
            spinbox.setMaximum(spec.max_value)
            spinbox.setValue(self.defaults[idx])
            spinbox.setSingleStep(spec.inferred_step)
            spinbox.setDecimals(spec.decimals)
            spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spinbox.setMaximumWidth(90)
            spinbox.setToolTip(spec.description)
            params_layout.addWidget(spinbox)
            self.batch_param_spinboxes[spec.key] = spinbox

        sync_btn = QPushButton("Sync from Manual")
        sync_btn.setToolTip("Copy parameters from Manual Fit tab")
        sync_btn.clicked.connect(self.sync_batch_params_from_manual)
        params_layout.addWidget(sync_btn)

        self.run_batch_btn = QPushButton("Run Batch")
        self.run_batch_btn.clicked.connect(self.run_batch_fit)

        export_table_btn = QPushButton("Export CSV")
        export_table_btn.clicked.connect(self.export_batch_table)

        self.batch_status_label = QLabel("")
        self.batch_status_label.setObjectName("statusLabel")
        self.batch_status_label.hide()
        params_layout.addWidget(self.batch_status_label)
        params_layout.addStretch()
        layout.addLayout(params_layout)

        files_layout = QHBoxLayout()
        files_layout.setSpacing(4)
        self.batch_files_label = QLabel("Selected Files:")
        files_layout.addWidget(self.batch_files_label, 1)
        self.batch_count_label = QLabel("0 files")
        files_layout.addWidget(self.batch_count_label)
        select_batch_btn = QPushButton("Select Files")
        select_batch_btn.clicked.connect(self.select_batch_files)
        files_layout.addWidget(select_batch_btn)
        files_layout.addWidget(self.run_batch_btn)
        files_layout.addWidget(export_table_btn)
        layout.addLayout(files_layout)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_batch_results_frame(self, parent_layout):
        """Create batch results table."""
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(0)
        self.batch_table.setRowCount(0)
        self.batch_table.cellClicked.connect(self._on_batch_table_cell_clicked)
        self.batch_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.batch_table.verticalHeader().setVisible(False)
        # Apply a uniform row height for thumbnail previews.
        self.batch_table.verticalHeader().setDefaultSectionSize(64)
        self.batch_table.setAlternatingRowColors(True)
        parent_layout.addWidget(self.batch_table)

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
        return [self.batch_param_spinboxes[spec.key].value() for spec in self.param_specs]

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
                    f.write(
                        f"{spec.symbol} ({spec.description}): {params[idx]:.6f}\n"
                    )

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
        self.batch_count_label.setText(f"{len(self.batch_files)} files")
        self.update_batch_file_list()
        self.prepare_batch_preview()

    def run_batch_fit(self):
        """Run batch fitting on selected files."""
        if not self.batch_files:
            self.stats_text.append("No batch files selected!")
            return
        if self.batch_thread is not None:
            self.stats_text.append("Batch fit is already running.")
            return

        regex_pattern = self.regex_input.text().strip()
        if regex_pattern:
            try:
                re.compile(regex_pattern)
            except re.error as exc:
                self.batch_status_label.setText(f"Regex error: {exc}")
                self.batch_status_label.show()
                return
        current_params = self.get_batch_params()

        self.prepare_batch_preview()

        self.batch_thread = QThread(self)
        self.batch_worker = BatchFitWorker(
            self.batch_files,
            current_params,
            self.bounds,
            regex_pattern,
            self.model_spec.function,
        )
        self.batch_worker.moveToThread(self.batch_thread)

        self.batch_thread.started.connect(self.batch_worker.run)
        self.batch_worker.progress.connect(self.on_batch_progress)
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.failed.connect(self.on_batch_failed)
        self.batch_worker.cancelled.connect(self.on_batch_cancelled)

        self.run_batch_btn.setEnabled(False)
        self.batch_status_label.setText("⏳ Batch fit in progress...")
        self.batch_status_label.show()
        self.batch_thread.start()

    def on_batch_progress(self, idx, total, row):
        """Update progress label while batch is running."""
        self.batch_status_label.setText(f"⏳ Batch fit {idx}/{total}")
        row_index = idx - 1
        if row_index < len(self.batch_results):
            existing = self.batch_results[row_index]
            if existing.get("thumbnail") is not None and row.get("plot") is None:
                row["plot"] = existing["thumbnail"]
            self.batch_results[row_index] = row
            self.update_batch_table_row(row_index, row)

    def on_batch_finished(self, results):
        """Populate table and thumbnails after batch fit finishes."""
        previous_by_file = {row["file"]: row for row in self.batch_results}
        self.batch_results = list(results)
        for row in self.batch_results:
            existing = previous_by_file.get(row["file"])
            if existing and existing.get("thumbnail") is not None:
                row["plot"] = existing["thumbnail"]
        self.update_batch_table()
        if any(row.get("plot") is None for row in self.batch_results):
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

        columns = (
            ["Thumbnail"]
            + ["File"]
            + self.batch_capture_keys
            + [spec.column_name for spec in self.param_specs]
            + ["R2", "Error"]
        )
        self.batch_table.setColumnCount(len(columns))
        self.batch_table.setHorizontalHeaderLabels(columns)
        self.batch_table.setRowCount(len(self.batch_results))

        for row_idx, row in enumerate(self.batch_results):
            self.update_batch_table_row(row_idx, row)

    def update_batch_table_row(self, row_idx, row):
        """Update a single batch row in the results table."""
        # Thumbnail column (index 0) - add icon if available
        thumb_item = QTableWidgetItem()
        pixmap = row.get("plot")
        if pixmap:
            thumb_item.setData(Qt.ItemDataRole.DecorationRole, pixmap)
        thumb_item.setData(Qt.ItemDataRole.UserRole, row["file"])  # Store file path
        self.batch_table.setItem(row_idx, 0, thumb_item)

        # File name column (index 1)
        file_name = Path(row["file"]).name
        self.batch_table.setItem(row_idx, 1, QTableWidgetItem(file_name))

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

    def _on_batch_table_cell_clicked(self, row_idx, col_idx):
        """Open selected row in the preview panel on click."""
        if col_idx not in (0, 1):
            return
        if row_idx < 0 or row_idx >= len(self.batch_results):
            return
        self._open_preview_panel(self.batch_results[row_idx])

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
            self.update_batch_table_row(row_idx, row)

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
        """Refresh selected file list display."""
        if not self.batch_files:
            self.batch_files_label.setText("Selected Files: ")
        else:
            first_name = Path(self.batch_files[0]).name
            count_str = (
                f" +{len(self.batch_files) - 1} more"
                if len(self.batch_files) > 1
                else ""
            )
            self.batch_files_label.setText(f"Selected Files: {first_name}{count_str}")

    def _on_regex_changed(self):
        """Debounce regex changes to avoid excessive updates."""
        self.regex_timer.stop()
        self.regex_timer.start(300)  # 300ms debounce

    def prepare_batch_preview(self):
        """Populate preview results before running batch fit."""
        self.regex_timer.stop()
        self._do_prepare_batch_preview()

    def _do_prepare_batch_preview(self):
        """Actually perform the batch preview update."""
        if not self.batch_files:
            self._close_preview_panel()
            return

        regex_pattern = self.regex_input.text().strip()
        if regex_pattern:
            try:
                regex = re.compile(regex_pattern)
            except re.error as exc:
                self.batch_status_label.setText(f"Regex error: {exc}")
                self.batch_status_label.show()
                return
        else:
            regex = None

        self.batch_status_label.hide()

        self._stop_thumbnail_render()

        # Build a map of existing results by file path to preserve params/r2/error
        existing_results = {row["file"]: row for row in self.batch_results}

        self.batch_results = []
        self.batch_capture_keys = []
        for file_path in self.batch_files:
            captures = {}
            if regex:
                match = regex.search(Path(file_path).stem)
                if match:
                    captures = match.groupdict()
                    for key in captures.keys():
                        if key not in self.batch_capture_keys:
                            self.batch_capture_keys.append(key)

            # Preserve existing params/r2/error if file was already processed
            existing = existing_results.get(file_path)
            self.batch_results.append(
                {
                    "file": file_path,
                    "captures": captures,
                    "params": existing["params"] if existing else None,
                    "r2": existing["r2"] if existing else None,
                    "error": existing["error"] if existing else None,
                    "thumbnail": existing["thumbnail"] if existing else None,
                }
            )

        self.update_batch_table()
        if self._preview_file_path and not any(
            row["file"] == self._preview_file_path for row in self.batch_results
        ):
            self._close_preview_panel()
        if any(row.get("plot") is None for row in self.batch_results):
            self._start_thumbnail_render()

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
