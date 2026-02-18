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
    QPlainTextEdit,
    QTabWidget,
    QCheckBox,
    QFileDialog,
    QLineEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QDateTime
from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


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
        self.setGeometry(100, 100, 1200, 900)

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

        # Current directory
        self.current_dir = "./measurements_18-02-26/shaking_functions/"

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # File selection
        self.create_file_frame(main_layout)

        # Plot frame
        self.create_plot_frame(main_layout)

        # Parameters frame
        self.create_parameters_frame(main_layout)

        # Control buttons
        self.create_controls_frame(main_layout)

        # Stats frame
        self.create_stats_frame(main_layout)

        # Load files
        self.load_files()

    def create_file_frame(self, parent_layout):
        """Create file selection section."""
        group = QGroupBox("File Selection")
        layout = QVBoxLayout()

        # Directory selection
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Directory:"))
        self.dir_input = QLineEdit(self.current_dir)
        self.dir_input.setReadOnly(True)
        dir_layout.addWidget(self.dir_input)

        browse_dir_btn = QPushButton("Browse Directory")
        browse_dir_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(browse_dir_btn)
        layout.addLayout(dir_layout)

        # File selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Select File:"))
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        file_layout.addWidget(self.file_combo)

        browse_file_btn = QPushButton("Browse File")
        browse_file_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_file_btn)

        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(self.prev_file)
        file_layout.addWidget(prev_btn)

        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(self.next_file)
        file_layout.addWidget(next_btn)

        layout.addLayout(file_layout)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_plot_frame(self, parent_layout):
        """Create plot section."""
        group = QGroupBox("Data and Fit")
        layout = QVBoxLayout()

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        # Add matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_parameters_frame(self, parent_layout):
        """Create parameter controls section."""
        group = QGroupBox("MI Model Parameters: |A*sin(B*x + φ) + D|²")
        layout = QVBoxLayout()

        # Parameter A (MI Amplitude)
        a_layout = self.create_param_control(
            "A (MI Amplitude):",
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
            "B (Voltage to Phase):",
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
            "φ (MI Phase):",
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
            "D (MI Offset):",
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
        """Helper to create parameter control with spinbox and slider."""
        layout = QHBoxLayout()

        label = QLabel(label_text)
        label.setMinimumWidth(200)
        layout.addWidget(label)

        # Spinbox
        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(default_val)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(4)
        spinbox.setMinimumWidth(100)
        spinbox.valueChanged.connect(self.update_plot)
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
            self.update_plot()

        def spinbox_to_slider(value):
            slider.blockSignals(True)
            slider.setValue(int(value * 1000))
            slider.blockSignals(False)

        slider.valueChanged.connect(slider_to_spinbox)
        spinbox.valueChanged.connect(spinbox_to_slider)
        layout.addWidget(slider)

        return (layout, spinbox, slider)

    def create_controls_frame(self, parent_layout):
        """Create control buttons section."""
        group = QGroupBox("Controls")
        layout = QHBoxLayout()

        self.auto_fit_btn = QPushButton("🔄 Auto Fit")
        self.auto_fit_btn.setStyleSheet(
            "font-weight: bold; background-color: #4CAF50; color: white;"
        )
        self.auto_fit_btn.clicked.connect(self.auto_fit)
        layout.addWidget(self.auto_fit_btn)

        self.cancel_fit_btn = QPushButton("Cancel Fit")
        self.cancel_fit_btn.setStyleSheet(
            "font-weight: bold; background-color: #d32f2f; color: white;"
        )
        self.cancel_fit_btn.setEnabled(False)
        self.cancel_fit_btn.clicked.connect(self.cancel_auto_fit)
        layout.addWidget(self.cancel_fit_btn)

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_params)
        layout.addWidget(reset_btn)

        self.show_residuals_cb = QCheckBox("Show Residuals")
        self.show_residuals_cb.setChecked(True)
        self.show_residuals_cb.stateChanged.connect(self.update_plot)
        layout.addWidget(self.show_residuals_cb)

        save_btn = QPushButton("Save Parameters")
        save_btn.clicked.connect(self.save_params)
        layout.addWidget(save_btn)

        export_btn = QPushButton("Export Fit")
        export_btn.clicked.connect(self.export_fit)
        layout.addWidget(export_btn)

        layout.addStretch()

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_stats_frame(self, parent_layout):
        """Create statistics display section."""
        group = QGroupBox("Fit Statistics")
        layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(100)
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
            self.update_plot()
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

    def browse_directory(self):
        """Browse for a directory containing CSV files."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory", str(Path.cwd())
        )
        if dir_path:
            self.current_dir = dir_path + "/"
            self.dir_input.setText(self.current_dir)
            self.load_files()
            self.stats_text.append(f"Loaded directory: {self.current_dir}")

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
            self.stats_text.append(f"Loaded file: {Path(file_path).name}")

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
        self.cancel_fit_btn.setEnabled(True)
        self.stats_text.append("\nAuto-fit started...")
        self.fit_thread.start()

    def cancel_auto_fit(self):
        """Request cancellation of the running auto-fit."""
        if self.fit_worker is None:
            return
        self.fit_worker.request_cancel()
        self.cancel_fit_btn.setEnabled(False)
        self.stats_text.append("Cancelling auto-fit...")

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
        self.cancel_fit_btn.setEnabled(False)

    def update_plot(self):
        """Update plot with current parameters."""
        if self.current_data is None:
            return

        try:
            data = self.current_data
            params = self.get_current_params()

            # Clear plot
            self.ax.clear()

            # Plot original data
            self.ax.plot(
                data["TIME"] * 1e3, data["CH2"], label=self.channels["CH2"], linewidth=2
            )
            self.ax.plot(
                data["TIME"] * 1e3,
                data["CH3"],
                label=self.channels["CH3"],
                color="gray",
                alpha=0.5,
            )

            # Plot fitted curve
            fitted_ch2 = mi_model(data["CH3"], *params)
            self.ax.plot(
                data["TIME"] * 1e3,
                fitted_ch2,
                label="Fitted CH2",
                linewidth=2,
                linestyle="--",
            )

            # Show residuals if enabled
            if self.show_residuals_cb.isChecked():
                residuals = data["CH2"] - fitted_ch2
                self.ax.plot(
                    data["TIME"] * 1e3,
                    residuals,
                    label="Residuals",
                    linestyle=":",
                    linewidth=1.5,
                )

            # Calculate R² score
            r2 = r2_score(data["CH2"], fitted_ch2)

            # Add parameter text to plot
            param_names = [
                "A (MI Amplitude)",
                "B (V→Phase)",
                "φ (MI Phase)",
                "D (MI Offset)",
            ]
            param_text = "\n".join(
                [f"{name}: {value:.4f}" for name, value in zip(param_names, params)]
            )
            param_text += f"\nR² Score: {r2:.4f}"

            self.ax.text(
                0.02,
                0.98,
                param_text,
                transform=self.ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            self.ax.legend(loc="upper right")
            self.ax.set_xlabel("Time (ms)")
            self.ax.set_ylabel("Voltage (V)")
            self.ax.set_title(Path(self.data_files[self.current_file_idx]).name)
            self.ax.grid(True, alpha=0.3)

            self.fig.tight_layout()
            self.canvas.draw()

            # Update stats text
            stats = f"R² Score: {r2:.6f}\n"
            stats += f"A (MI Amplitude): {params[0]:.6f}\n"
            stats += f"B (Voltage to Phase): {params[1]:.6f}\n"
            stats += f"φ (MI Phase): {params[2]:.6f}\n"
            stats += f"D (MI Offset): {params[3]:.6f}"

            if self.last_popt is not None and self.last_pcov is not None:
                popt_text = np.array2string(
                    self.last_popt, precision=6, floatmode="fixed", suppress_small=False
                )
                pcov_text = np.array2string(
                    self.last_pcov, precision=6, floatmode="fixed", suppress_small=False
                )
                sigma = np.sqrt(np.maximum(np.diag(self.last_pcov), 0.0))
                sigma_text = np.array2string(
                    sigma, precision=6, floatmode="fixed", suppress_small=False
                )
                fit_r2_text = (
                    f"{self.last_fit_r2:.6f}" if self.last_fit_r2 is not None else "N/A"
                )
                stats += "\n\nLast Auto-Fit:"
                stats += f"\nR²: {fit_r2_text}"
                stats += f"\npopt: {popt_text}"
                stats += f"\n1σ (sqrt(diag(pcov))): {sigma_text}"
                stats += f"\npcov:\n{pcov_text}"
            self.stats_text.setText(stats)

        except Exception as e:
            self.stats_text.setText(f"Error updating plot: {e}")

    def save_params(self):
        """Save current parameters as new defaults."""
        self.defaults = self.get_current_params()
        self.stats_text.append(f"\nParameters saved as defaults!")

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

            self.stats_text.append(f"Parameters exported to {filename}")

        except Exception as e:
            self.stats_text.append(f"Error exporting: {e}")

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
