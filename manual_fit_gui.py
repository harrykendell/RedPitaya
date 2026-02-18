#!/usr/bin/env python3
"""
Manual Curve Fitting GUI for MI Model
Allows manual adjustment of parameters for failed automatic fits.
"""

import sys
import glob
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
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

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

        # File selection (compact)
        self.create_file_frame(main_layout)

        # Plot frame
        self.create_plot_frame(main_layout)

        # Parameters and Stats side by side
        params_stats_layout = QHBoxLayout()
        self.create_parameters_frame(params_stats_layout)
        self.create_stats_frame(params_stats_layout)
        main_layout.addLayout(params_stats_layout)

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

        # Progress status label
        self.fit_status_label = QLabel("")
        self.fit_status_label.setStyleSheet(
            "font-weight: bold; color: #FF9800; padding: 5px;"
        )
        self.fit_status_label.hide()
        btn_layout.addWidget(self.fit_status_label)

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
        self.fit_status_label.setText("⏳ Auto-fit in progress...")
        self.fit_status_label.show()
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
        self.fit_status_label.hide()

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

    def closeEvent(self, event):
        """Ensure worker thread is stopped before closing."""
        if self.fit_worker is not None:
            self.fit_worker.request_cancel()
        if self.fit_thread is not None:
            self.fit_thread.quit()
            self.fit_thread.wait(2000)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ManualFitGUI()
    window.show()
    sys.exit(app.exec())
