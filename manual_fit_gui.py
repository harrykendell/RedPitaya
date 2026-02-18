#!/usr/bin/env python3
"""
Manual Curve Fitting GUI for MI Model
Allows manual adjustment of parameters for failed automatic fits.
"""

import sys
import glob
import re
import csv
import numpy as np
from pathlib import Path
from pandas import read_csv
from sklearn.metrics import r2_score
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
    QCheckBox,
    QFileDialog,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGridLayout,
    QScrollArea,
    QTabWidget,
    QListWidget,
    QDialog,
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# use Qt5Agg backend for better performance
from matplotlib.pyplot import switch_backend

switch_backend("Qt5Agg")


def mi_model(x, a, b, phi, d):
    """MI model: Voltage = |A*sin(B*Sig gen Output + phi_0) + D|^2"""
    return np.abs(a * np.sin(b * x + phi) + d) ** 2


class FitWorker(QObject):
    finished = pyqtSignal(object, object, float)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, x_data, y_data, p0, bounds):
        super().__init__()
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data = np.asarray(y_data, dtype=float)
        self.p0 = np.asarray(p0, dtype=float)
        self.bounds = bounds
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
                    mi_model,
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

            fitted = mi_model(self.x_data, *popt)
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

    def __init__(self, file_paths, p0, bounds, regex_pattern):
        super().__init__()
        self.file_paths = list(file_paths)
        self.p0 = np.asarray(p0, dtype=float)
        self.bounds = bounds
        self.regex = re.compile(regex_pattern) if regex_pattern else None
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
                        mi_model,
                        x_data,
                        y_data,
                        p0=self.p0,
                        bounds=self.bounds,
                        method="trf",
                        maxfev=500,
                    )

                    fitted = mi_model(x_data, *popt)
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

    def __init__(self, batch_results, thumbnail_size=(80, 60)):
        super().__init__()
        self.batch_results = batch_results
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
                row["thumbnail"] = pixmap
                self.progress.emit(idx + 1, total, idx)

            if self.cancel_requested:
                self.cancelled.emit()
                return

            self.finished.emit()
        except Exception:
            self.finished.emit()

    def render_thumbnail(self, row):
        """Render a plot to a QPixmap for embedding in table."""
        from io import BytesIO

        try:
            data = read_csv(row["file"], skiprows=13, header=0)
            time_data = data["TIME"].to_numpy() * 1e3
            ch2_data = data["CH2"].to_numpy()
            ch3_data = data["CH3"].to_numpy()

            fig = Figure(figsize=(2.0, 1.2), dpi=80)
            fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
            ax = fig.add_subplot(111)

            ax.plot(time_data, ch2_data, linewidth=1, color="C0")
            ax.plot(time_data, ch3_data, linewidth=1, alpha=0.4, color="C1")

            params = row.get("params")
            if params:
                fitted_ch2 = mi_model(ch3_data, *params)
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
        self.setWindowTitle("Manual MI Model Curve Fitting")
        self.setGeometry(100, 100, 600, 700)

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
        self._hover_dialog = None

        # Default parameter values
        self.defaults = [0.74545, -0.2175, 0.5263, 1.7019]
        self.bounds = ([0, -2, -2 * np.pi, -10], [100, 2, 2 * np.pi, 10])

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
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.manual_tab = QWidget()
        self.batch_tab = QWidget()
        self.tabs.addTab(self.manual_tab, "Manual Fit")
        self.tabs.addTab(self.batch_tab, "Batch Fit")

        manual_layout = QVBoxLayout(self.manual_tab)
        manual_layout.setContentsMargins(10, 10, 10, 10)
        manual_layout.setSpacing(5)

        # File selection (compact)
        self.create_file_frame(manual_layout)

        # Plot frame
        self.create_plot_frame(manual_layout)

        # Parameters and Stats side by side
        params_stats_layout = QHBoxLayout()
        self.create_parameters_frame(params_stats_layout)
        self.create_stats_frame(params_stats_layout)
        manual_layout.addLayout(params_stats_layout)

        batch_layout = QVBoxLayout(self.batch_tab)
        batch_layout.setContentsMargins(10, 10, 10, 10)
        batch_layout.setSpacing(8)
        self.create_batch_controls_frame(batch_layout)
        self.create_batch_results_frame(batch_layout)
        self.create_thumbnails_frame(batch_layout)

        # Load files
        self.load_files()

    def create_file_frame(self, parent_layout):
        """Create compact file selection section."""
        group = QGroupBox("")
        layout = QVBoxLayout()

        # Directory row
        dir_layout = QHBoxLayout()
        browse_dir_btn = QPushButton("📁")
        browse_dir_btn.setMaximumWidth(40)
        browse_dir_btn.setToolTip("Browse Directory")
        browse_dir_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(browse_dir_btn)

        self.dir_input = QLineEdit(self.current_dir)
        self.dir_input.setReadOnly(True)
        self.dir_input.setStyleSheet("QLineEdit { background-color: #f0f0f0; }")
        dir_layout.addWidget(self.dir_input)

        # File row
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        dir_layout.addWidget(self.file_combo)

        # Browse file button
        browse_file_btn = QPushButton("📄")
        browse_file_btn.setMaximumWidth(40)
        browse_file_btn.setToolTip("Browse File")
        browse_file_btn.clicked.connect(self.browse_file)
        dir_layout.addWidget(browse_file_btn)

        # Navigation buttons
        prev_btn = QPushButton("◀")
        prev_btn.setMaximumWidth(40)
        prev_btn.setToolTip("Previous File")
        prev_btn.clicked.connect(self.prev_file)
        dir_layout.addWidget(prev_btn)

        next_btn = QPushButton("▶")
        next_btn.setMaximumWidth(40)
        next_btn.setToolTip("Next File")
        next_btn.clicked.connect(self.next_file)
        dir_layout.addWidget(next_btn)

        layout.addLayout(dir_layout)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_plot_frame(self, parent_layout):
        """Create plot section."""
        group = QGroupBox("")
        layout = QVBoxLayout()

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.16)

        # Add matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_parameters_frame(self, parent_layout):
        """Create compact parameter controls section."""
        group = QGroupBox("")
        layout = QVBoxLayout()

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("|A·sin(B·x + φ) + D|²"))
        header_layout.addStretch()
        self.show_residuals_cb = QCheckBox("Show Residuals")
        self.show_residuals_cb.setChecked(True)
        self.show_residuals_cb.stateChanged.connect(
            lambda: self.update_plot(fast=False)
        )
        header_layout.addWidget(self.show_residuals_cb)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_params)
        header_layout.addWidget(reset_btn)
        layout.addLayout(header_layout)

        # Parameter A (MI Amplitude)
        a_layout = self.create_param_control(
            "A:",
            0.0,
            10.0,
            self.defaults[0],
            0.001,
            slider_range=(0, 10000),
        )
        self.a_spinbox = a_layout[1]
        self.a_slider = a_layout[2]
        layout.addLayout(a_layout[0])

        # Parameter B (Voltage to Freq to Phase)
        b_layout = self.create_param_control(
            "B:",
            -2.0,
            2.0,
            self.defaults[1],
            0.0001,
            slider_range=(-2000, 2000),
        )
        self.b_spinbox = b_layout[1]
        self.b_slider = b_layout[2]
        layout.addLayout(b_layout[0])

        # Parameter phi (MI Phase)
        phi_layout = self.create_param_control(
            "φ:",
            -2 * np.pi,
            2 * np.pi,
            self.defaults[2],
            0.001,
            slider_range=(-6283, 6283),
        )
        self.phi_spinbox = phi_layout[1]
        self.phi_slider = phi_layout[2]
        layout.addLayout(phi_layout[0])

        # Parameter D (MI Offset)
        d_layout = self.create_param_control(
            "D:",
            -10.0,
            10.0,
            self.defaults[3],
            0.001,
            slider_range=(-10000, 10000),
        )
        self.d_spinbox = d_layout[1]
        self.d_slider = d_layout[2]
        layout.addLayout(d_layout[0])

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_param_control(
        self, label_text, min_val, max_val, default_val, step, slider_range
    ):
        """Helper to create compact parameter control with spinbox and slider."""
        layout = QHBoxLayout()

        label = QLabel(label_text)
        label.setMinimumWidth(30)
        label.setMaximumWidth(30)
        layout.addWidget(label)

        # Spinbox
        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(default_val)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(4)
        spinbox.setMinimumWidth(90)
        spinbox.setMaximumWidth(90)
        spinbox.valueChanged.connect(lambda: self.update_plot(fast=False))
        layout.addWidget(spinbox)

        # Slider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(slider_range[0])
        slider.setMaximum(slider_range[1])
        slider.setValue(int(default_val * 1000))

        # Connect slider to spinbox and vice versa
        def slider_to_spinbox(value):
            spinbox.blockSignals(True)
            spinbox.setValue(value / 1000)
            spinbox.blockSignals(False)
            self.update_plot(fast=True)

        def spinbox_to_slider(value):
            slider.blockSignals(True)
            slider.setValue(int(value * 1000))
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

        # Auto-fit control buttons
        btn_layout = QHBoxLayout()

        self.auto_fit_btn = QPushButton("🔄 Auto Fit")
        self.auto_fit_btn.setStyleSheet(
            "font-weight: bold; background-color: #4CAF50; color: white;"
        )
        self.auto_fit_btn.clicked.connect(self.auto_fit)
        btn_layout.addWidget(self.auto_fit_btn)

        save_btn = QPushButton("Save Params")
        save_btn.clicked.connect(self.save_params)
        btn_layout.addWidget(save_btn)

        export_btn = QPushButton("Export Fit")
        export_btn.clicked.connect(self.export_fit)
        btn_layout.addWidget(export_btn)

        layout.addLayout(btn_layout)

        # Stats text display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        layout.addWidget(self.stats_text)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_batch_controls_frame(self, parent_layout):
        """Create batch controls and selected file display."""
        group = QGroupBox("Batch Controls")
        layout = QVBoxLayout()

        regex_layout = QHBoxLayout()
        regex_layout.addWidget(QLabel("Regex:"))
        self.regex_input = QLineEdit(r"data_(?P<freq>\d+Hz)_(?P<idx>\d{3})_ALL")
        self.regex_input.setToolTip("Named groups become table columns")
        self.regex_input.textChanged.connect(self._on_regex_changed)
        regex_layout.addWidget(self.regex_input)
        layout.addLayout(regex_layout)

        # Batch fit initial parameters
        params_label = QLabel("Initial Fit Parameters:")
        params_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(params_label)

        params_layout = QHBoxLayout()

        # A parameter
        params_layout.addWidget(QLabel("A:"))
        self.batch_a_spinbox = QDoubleSpinBox()
        self.batch_a_spinbox.setMinimum(0.0)
        self.batch_a_spinbox.setMaximum(100.0)
        self.batch_a_spinbox.setValue(self.defaults[0])
        self.batch_a_spinbox.setSingleStep(0.001)
        self.batch_a_spinbox.setDecimals(4)
        params_layout.addWidget(self.batch_a_spinbox)

        # B parameter
        params_layout.addWidget(QLabel("B:"))
        self.batch_b_spinbox = QDoubleSpinBox()
        self.batch_b_spinbox.setMinimum(-2.0)
        self.batch_b_spinbox.setMaximum(2.0)
        self.batch_b_spinbox.setValue(self.defaults[1])
        self.batch_b_spinbox.setSingleStep(0.0001)
        self.batch_b_spinbox.setDecimals(4)
        params_layout.addWidget(self.batch_b_spinbox)

        # phi parameter
        params_layout.addWidget(QLabel("φ:"))
        self.batch_phi_spinbox = QDoubleSpinBox()
        self.batch_phi_spinbox.setMinimum(-2 * np.pi)
        self.batch_phi_spinbox.setMaximum(2 * np.pi)
        self.batch_phi_spinbox.setValue(self.defaults[2])
        self.batch_phi_spinbox.setSingleStep(0.001)
        self.batch_phi_spinbox.setDecimals(4)
        params_layout.addWidget(self.batch_phi_spinbox)

        # D parameter
        params_layout.addWidget(QLabel("D:"))
        self.batch_d_spinbox = QDoubleSpinBox()
        self.batch_d_spinbox.setMinimum(-10.0)
        self.batch_d_spinbox.setMaximum(10.0)
        self.batch_d_spinbox.setValue(self.defaults[3])
        self.batch_d_spinbox.setSingleStep(0.001)
        self.batch_d_spinbox.setDecimals(4)
        params_layout.addWidget(self.batch_d_spinbox)

        sync_btn = QPushButton("Sync from Manual")
        sync_btn.setToolTip("Copy parameters from Manual Fit tab")
        sync_btn.clicked.connect(self.sync_batch_params_from_manual)
        params_layout.addWidget(sync_btn)

        params_layout.addStretch()
        layout.addLayout(params_layout)

        btn_layout = QHBoxLayout()
        select_batch_btn = QPushButton("Select Files")
        select_batch_btn.clicked.connect(self.select_batch_files)
        btn_layout.addWidget(select_batch_btn)

        self.batch_count_label = QLabel("0 files")
        btn_layout.addWidget(self.batch_count_label)

        self.run_batch_btn = QPushButton("Run Batch Fit")
        self.run_batch_btn.clicked.connect(self.run_batch_fit)
        btn_layout.addWidget(self.run_batch_btn)

        export_table_btn = QPushButton("Export Table CSV")
        export_table_btn.clicked.connect(self.export_batch_table)
        btn_layout.addWidget(export_table_btn)

        self.batch_status_label = QLabel("")
        self.batch_status_label.setStyleSheet(
            "font-weight: bold; color: #1976D2; padding: 5px;"
        )
        self.batch_status_label.hide()
        btn_layout.addWidget(self.batch_status_label)
        layout.addLayout(btn_layout)

        self.batch_files_label = QLabel("Selected Files: ")
        layout.addWidget(self.batch_files_label)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_batch_results_frame(self, parent_layout):
        """Create batch results table."""
        group = QGroupBox("Batch Results")
        layout = QVBoxLayout()

        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(0)
        self.batch_table.setRowCount(0)
        self.batch_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.batch_table.setRowHeight(80)  # Accommodate thumbnail height
        self.batch_table.mouseMoveEvent = self._batch_table_mouse_move
        self.batch_table.leaveEvent = self._batch_table_leave
        layout.addWidget(self.batch_table)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_thumbnails_frame(self, parent_layout):
        """Create thumbnail plot grid (hidden - thumbnails embedded in table)."""
        group = QGroupBox("Batch Thumbnails")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("(Thumbnails displayed in table on hover)"))
        group.setLayout(layout)
        group.hide()
        parent_layout.addWidget(group)

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
        return [
            self.a_spinbox.value(),
            self.b_spinbox.value(),
            self.phi_spinbox.value(),
            self.d_spinbox.value(),
        ]

    def get_batch_params(self):
        """Get batch fit initial parameters."""
        return [
            self.batch_a_spinbox.value(),
            self.batch_b_spinbox.value(),
            self.batch_phi_spinbox.value(),
            self.batch_d_spinbox.value(),
        ]

    def sync_batch_params_from_manual(self):
        """Copy parameters from manual tab to batch tab."""
        params = self.get_current_params()
        self.batch_a_spinbox.setValue(params[0])
        self.batch_b_spinbox.setValue(params[1])
        self.batch_phi_spinbox.setValue(params[2])
        self.batch_d_spinbox.setValue(params[3])

    def reset_params(self):
        """Reset parameters to defaults."""
        self.a_spinbox.setValue(self.defaults[0])
        self.b_spinbox.setValue(self.defaults[1])
        self.phi_spinbox.setValue(self.defaults[2])
        self.d_spinbox.setValue(self.defaults[3])

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
        self.fit_worker = FitWorker(x_data, y_data, current_params, self.bounds)
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

        self.a_spinbox.setValue(self.last_popt[0])
        self.b_spinbox.setValue(self.last_popt[1])
        self.phi_spinbox.setValue(self.last_popt[2])
        self.d_spinbox.setValue(self.last_popt[3])
        self.defaults = list(self.last_popt)

        self.stats_text.append(f"✓ Auto-fit successful! R² = {self.last_fit_r2:.6f}")
        self.stats_text.append(
            f"A={self.last_popt[0]:.4f}, B={self.last_popt[1]:.4f}, φ={self.last_popt[2]:.4f}, D={self.last_popt[3]:.4f}"
        )
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
            fitted_ch2 = mi_model(ch3_data, *params)

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
                stats += f"A = {self.last_popt[0]:.6f} ± {sigma[0]:.6f}\n"
                stats += f"B = {self.last_popt[1]:.6f} ± {sigma[1]:.6f}\n"
                stats += f"φ = {self.last_popt[2]:.6f} ± {sigma[2]:.6f}\n"
                stats += f"D = {self.last_popt[3]:.6f} ± {sigma[3]:.6f}"
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
                f.write(f"A (MI Amplitude): {params[0]:.6f}\n")
                f.write(f"B (Voltage to Phase): {params[1]:.6f}\n")
                f.write(f"φ (MI Phase): {params[2]:.6f}\n")
                f.write(f"D (MI Offset): {params[3]:.6f}\n")

                if self.current_data is not None:
                    fitted_ch2 = mi_model(self.current_data["CH3"], *params)
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
            self.batch_files, current_params, self.bounds, regex_pattern
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
            self.batch_results[row_index] = row
            self.update_batch_table_row(row_index, row)

    def on_batch_finished(self, results):
        """Populate table and thumbnails after batch fit finishes."""
        self.batch_results = list(results)
        self.update_batch_table()
        self.update_thumbnails()
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
            + [
                "A",
                "B",
                "phi",
                "D",
                "R2",
                "Error",
            ]
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
        pixmap = row.get("thumbnail")
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
        if params:
            for offset, val in enumerate(params):
                self.batch_table.setItem(
                    row_idx,
                    param_start + offset,
                    QTableWidgetItem(f"{val:.6f}"),
                )
        else:
            for offset in range(4):
                self.batch_table.setItem(
                    row_idx,
                    param_start + offset,
                    QTableWidgetItem(""),
                )
        r2_val = row.get("r2")
        if r2_val is not None:
            self.batch_table.setItem(
                row_idx,
                param_start + 4,
                QTableWidgetItem(f"{r2_val:.6f}"),
            )
        else:
            self.batch_table.setItem(
                row_idx,
                param_start + 4,
                QTableWidgetItem(""),
            )
        error_text = row.get("error") or ""
        self.batch_table.setItem(
            row_idx,
            param_start + 5,
            QTableWidgetItem(error_text),
        )

    def _batch_table_mouse_move(self, event):
        """Handle mouse hover over batch table to show popup."""
        item = self.batch_table.itemAt(event.pos())
        if item and item.column() == 0:  # Thumbnail column
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path:
                # Find the row's batch result
                for row in self.batch_results:
                    if row["file"] == file_path:
                        self._show_hover_plot(row, event.globalPos())
                        break

    def _batch_table_leave(self, event):
        """Handle mouse leaving batch table."""
        pass

    def _show_hover_plot(self, row, pos):
        """Show full-size plot in a popup dialog on hover."""
        if hasattr(self, "_hover_dialog") and self._hover_dialog:
            self._hover_dialog.close()

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Plot: {Path(row['file']).name}")
        dialog.setGeometry(pos.x(), pos.y(), 1000, 600)
        layout = QVBoxLayout(dialog)

        fig = Figure(figsize=(12, 6), dpi=100)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.1)
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        try:
            data = read_csv(row["file"], skiprows=13, header=0)
            time_data = data["TIME"].to_numpy() * 1e3
            ch2_data = data["CH2"].to_numpy()
            ch3_data = data["CH3"].to_numpy()

            ax.plot(time_data, ch2_data, label="CH2 (MI output)", linewidth=2)
            ax.plot(time_data, ch3_data, label="CH3 (Sig Gen)", linewidth=2, alpha=0.5)

            params = row.get("params")
            if params:
                fitted_ch2 = mi_model(ch3_data, *params)
                ax.plot(time_data, fitted_ch2, label="Fitted", linewidth=2)

            r2_val = row.get("r2")
            error_text = row.get("error")
            if error_text:
                ax.set_title(f"Error: {error_text[:80]}", color="red", fontsize=12)
            elif r2_val is not None:
                color = "green" if r2_val > 0.95 else "orange"
                ax.set_title(f"R² = {r2_val:.6f}", color=color, fontsize=12)

            ax.legend(loc="best", fontsize=10)
            ax.set_xlabel("Time (ms)", fontsize=11)
            ax.set_ylabel("Voltage (V)", fontsize=11)
            ax.grid(True, alpha=0.3)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Plot error: {str(exc)}", ha="center", va="center")
            ax.set_axis_off()

        self._hover_dialog = dialog
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()

    def _start_thumbnail_render(self):
        """Start background thread to render all thumbnails."""
        if self.thumb_render_in_progress or not self.batch_results:
            return

        self.thumb_render_in_progress = True
        self.thumb_thread = QThread(self)
        self.thumb_worker = ThumbnailRenderWorker(self.batch_results)
        self.thumb_worker.moveToThread(self.thumb_thread)

        self.thumb_thread.started.connect(self.thumb_worker.run)
        self.thumb_worker.progress.connect(self._on_thumbnail_rendered)
        self.thumb_worker.finished.connect(self._on_thumbnails_finished)
        self.thumb_thread.start()

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
            + [
                "A",
                "B",
                "phi",
                "D",
                "R2",
                "Error",
            ]
        )

        try:
            with open(file_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                for row in self.batch_results:
                    file_name = Path(row["file"]).name
                    captures = row.get("captures", {})
                    params = row.get("params") or ["", "", "", ""]
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

        # Build a map of existing results by file path to preserve params/r2/error
        existing_results = {row["file"]: row for row in self.batch_results}
        is_first_preview = len(existing_results) == 0

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
                }
            )

        self.update_batch_table()
        # Start background thumbnail rendering on first preview
        if is_first_preview:
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
