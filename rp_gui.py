import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from redpitaya_scpi import scpi


class WavePreview(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.samples = np.zeros(1024)
        self.setMinimumHeight(200)

    def set_samples(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            self.samples = np.zeros(1024)
        else:
            self.samples = np.asarray(samples, dtype=float)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        w, h = self.width(), self.height()

        # Midline
        mid_y = h / 2
        painter.setPen(QPen(Qt.GlobalColor.darkGray, 1))
        painter.drawLine(0, int(mid_y), w, int(mid_y))

        if self.samples.size < 2:
            return

        data = np.clip(self.samples, -1.0, 1.0)
        x_scale = (w - 1) / (len(data) - 1)

        painter.setPen(QPen(Qt.GlobalColor.green, 2))
        last_x = 0
        last_y = int(mid_y - data[0] * (h * 0.45))
        for i, y in enumerate(data[1:], start=1):
            x = int(i * x_scale)
            yy = int(mid_y - y * (h * 0.45))
            painter.drawLine(last_x, last_y, x, yy)
            last_x, last_y = x, yy


class MainWindow(QMainWindow):
    BTN_GREEN = "QPushButton { background-color: #2e7d32; color: white; font-weight: 600; }"
    BTN_RED = "QPushButton { background-color: #b71c1c; color: white; font-weight: 600; }"
    BTN_GRAY = "QPushButton { background-color: #616161; color: white; font-weight: 600; }"
    BTN_DIM = "QPushButton { background-color: #263238; color: #b0bec5; font-weight: 600; }"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Red Pitaya Waveform Control")

        self.rp = None
        self.file_samples = np.array([], dtype=float)
        self.output_enabled = False
        self.active_mode = "NONE"
        self.active_freq = 1000.0
        self.active_amplitude = 0.0
        self.active_offset = 0.0
        self.active_base_samples = np.zeros(1000)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # Connection controls
        conn_box = QGroupBox("Connection")
        conn_layout = QGridLayout(conn_box)
        self.ip_edit = QLineEdit("192.168.0.6")
        self.chan_combo = QComboBox()
        self.chan_combo.addItems(["1", "2"])
        self.connect_toggle_btn = QPushButton("Connect")

        conn_layout.addWidget(QLabel("IP"), 0, 0)
        conn_layout.addWidget(self.ip_edit, 0, 1)
        conn_layout.addWidget(QLabel("Channel"), 0, 2)
        conn_layout.addWidget(self.chan_combo, 0, 3)
        conn_layout.addWidget(self.connect_toggle_btn, 1, 0, 1, 4)

        # Waveform controls
        wave_box = QGroupBox("Waveform")
        wave_layout = QFormLayout(wave_box)

        self.wave_combo = QComboBox()
        self.wave_combo.addItems(["SINE", "SQUARE", "FILE"])

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.1, 1e7)
        self.freq_spin.setDecimals(3)
        self.freq_spin.setValue(1000.0)

        self.volt_spin = QDoubleSpinBox()
        self.volt_spin.setRange(0.0, 10.0)
        self.volt_spin.setDecimals(3)
        self.volt_spin.setValue(1.0)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1.0, 1.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setValue(0.0)

        self.file_label = QLabel("No file loaded")
        self.load_file_btn = QPushButton("Load Waveform File")

        wave_layout.addRow("Type", self.wave_combo)
        wave_layout.addRow("Frequency (Hz)", self.freq_spin)
        wave_layout.addRow("Amplitude (V)", self.volt_spin)
        wave_layout.addRow("Offset (V)", self.offset_spin)
        wave_layout.addRow(self.load_file_btn, self.file_label)

        # Action buttons
        action_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Waveform")
        self.output_toggle_btn = QPushButton("Output OFF")
        action_row.addWidget(self.apply_btn)
        action_row.addWidget(self.output_toggle_btn)

        # Status + preview
        self.status_label = QLabel("Disconnected")
        self.pending_label = QLabel("Selected waveform: SINE")
        self.pending_preview = WavePreview()
        self.active_label = QLabel("Active waveform: NONE")
        self.active_preview = WavePreview()
        self.timebase_label = QLabel("")

        layout.addWidget(conn_box)
        layout.addWidget(wave_box)
        layout.addLayout(action_row)
        layout.addWidget(self.pending_label)
        layout.addWidget(self.pending_preview)
        layout.addWidget(self.active_label)
        layout.addWidget(self.active_preview)
        layout.addWidget(self.timebase_label)
        layout.addWidget(self.status_label)

        # Signals
        self.connect_toggle_btn.clicked.connect(self.toggle_connection)
        self.load_file_btn.clicked.connect(self.load_file)
        self.wave_combo.currentTextChanged.connect(self.refresh_pending_preview)
        self.freq_spin.valueChanged.connect(self.refresh_previews)
        self.volt_spin.valueChanged.connect(self.refresh_previews)
        self.offset_spin.valueChanged.connect(self.refresh_previews)
        self.apply_btn.clicked.connect(self.apply_waveform)
        self.output_toggle_btn.clicked.connect(self.toggle_output)

        self.refresh_previews()
        self.update_button_states()

    @property
    def chan(self) -> int:
        return int(self.chan_combo.currentText())

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def update_button_states(self) -> None:
        connected = self.rp is not None
        self.connect_toggle_btn.setEnabled(True)
        self.apply_btn.setEnabled(connected)
        self.output_toggle_btn.setEnabled(connected)

        if connected:
            self.connect_toggle_btn.setText("Disconnect")
            self.connect_toggle_btn.setStyleSheet(self.BTN_RED)
        else:
            self.connect_toggle_btn.setText("Connect")
            self.connect_toggle_btn.setStyleSheet(self.BTN_GREEN)

        if self.output_enabled and connected:
            self.output_toggle_btn.setText("Output ON")
            self.output_toggle_btn.setStyleSheet(self.BTN_GREEN)
        else:
            self.output_toggle_btn.setText("Output OFF")
            self.output_toggle_btn.setStyleSheet(self.BTN_RED if connected else self.BTN_GRAY)

    def ensure_connected(self) -> bool:
        if self.rp is None:
            QMessageBox.warning(self, "Not connected", "Connect to Red Pitaya first.")
            return False
        return True

    def connect_rp(self) -> None:
        ip = self.ip_edit.text().strip()
        if not ip:
            QMessageBox.warning(self, "Missing IP", "Please enter an IP address.")
            return

        try:
            self.rp = scpi(ip)
            idn = self.rp.idn_q().strip()
            self.rp.tx_txt("GEN:RST")
            self.output_enabled = False
            self.set_status(f"Connected: {idn}")
            self.update_button_states()
        except Exception as exc:
            self.rp = None
            self.output_enabled = False
            QMessageBox.critical(self, "Connection failed", str(exc))
            self.set_status("Connection failed")
            self.update_button_states()

    def disconnect_rp(self) -> None:
        self.rp = None
        self.output_enabled = False
        self.set_status("Disconnected")
        self.update_button_states()

    def toggle_connection(self) -> None:
        if self.rp is None:
            self.connect_rp()
        else:
            self.disconnect_rp()

    def load_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load waveform",
            str(Path.cwd()),
            "Data files (*.csv *.txt);;All files (*)",
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            raw = np.loadtxt(path, delimiter=",")
            if raw.ndim == 2:
                if raw.shape[1] == 1:
                    raw = raw[:, 0]
                else:
                    raw = raw[:, -1]
            samples = np.asarray(raw, dtype=float).flatten()
            if samples.size < 2:
                raise ValueError("File must contain at least two samples.")

            samples = np.clip(samples, -1.0, 1.0)
            self.file_samples = samples
            self.file_label.setText(f"{path.name} ({samples.size} samples)")
            self.set_status(f"Loaded waveform file: {path}")
            if self.wave_combo.currentText() == "FILE":
                self.refresh_previews()
        except Exception as exc:
            QMessageBox.critical(self, "File load failed", str(exc))

    def build_mode_base_samples(self, mode: str) -> np.ndarray:
        x = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        if mode == "SINE":
            return np.sin(x)
        if mode == "SQUARE":
            return np.where(np.sin(x) >= 0, 1.0, -1.0)
        if mode == "FILE" and self.file_samples.size:
            return self.file_samples
        return np.zeros(1000)

    def get_shared_timebase(self) -> np.ndarray:
        n_points = 1000
        preview_cycles = 4.0
        f = max(self.freq_spin.value(), 0.1)
        duration = preview_cycles / f
        return np.linspace(0.0, duration, n_points, endpoint=False)

    def render_preview_samples(
        self,
        mode: str,
        base_samples: np.ndarray,
        freq: float,
        amplitude: float,
        offset: float,
        t: np.ndarray,
    ) -> np.ndarray:
        if mode == "SINE":
            norm = np.sin(2 * np.pi * freq * t)
        elif mode == "SQUARE":
            norm = np.where(np.sin(2 * np.pi * freq * t) >= 0, 1.0, -1.0)
        elif mode == "FILE" and base_samples.size > 1:
            phase = np.mod(freq * t, 1.0)
            src_x = np.linspace(0.0, 1.0, base_samples.size, endpoint=False)
            norm = np.interp(phase, src_x, base_samples, period=1.0)
        else:
            norm = np.zeros_like(t)
        return np.clip(amplitude * norm + offset, -1.0, 1.0)

    def refresh_pending_preview(self) -> None:
        self.refresh_previews()

    def refresh_previews(self) -> None:
        pending_mode = self.wave_combo.currentText()
        pending_freq = self.freq_spin.value()
        pending_amplitude = self.volt_spin.value()
        pending_offset = self.offset_spin.value()
        t = self.get_shared_timebase()
        pending_base = self.build_mode_base_samples(pending_mode)
        pending_samples = self.render_preview_samples(
            pending_mode,
            pending_base,
            pending_freq,
            pending_amplitude,
            pending_offset,
            t,
        )
        self.pending_label.setText(
            f"Selected: {pending_mode} @ {pending_freq:.3f} Hz, A={pending_amplitude:.3f} V, Off={pending_offset:.3f} V"
        )
        self.pending_preview.set_samples(pending_samples)

        if self.output_enabled:
            active_samples = self.render_preview_samples(
                self.active_mode,
                self.active_base_samples,
                self.active_freq,
                self.active_amplitude,
                self.active_offset,
                t,
            )
            active_state = "ON"
        else:
            # Show disabled output as a flat zero trace on the active pane.
            active_samples = np.zeros_like(t)
            active_state = "OFF"
        self.active_label.setText(
            f"Active ({active_state}): {self.active_mode} @ {self.active_freq:.3f} Hz, A={self.active_amplitude:.3f} V, Off={self.active_offset:.3f} V"
        )
        self.active_preview.set_samples(active_samples)

        window_ms = (t[-1] - t[0]) * 1000 if t.size > 1 else 0.0
        self.timebase_label.setText(
            f"Shared preview time base: {window_ms:.3f} ms window, {t.size} samples"
        )

    def apply_waveform(self) -> None:
        if not self.ensure_connected():
            return

        try:
            mode = self.wave_combo.currentText()
            chan = self.chan

            if mode == "FILE":
                if self.file_samples.size == 0:
                    QMessageBox.warning(self, "No file", "Load a waveform file first.")
                    return
                # Apply amplitude/offset in software and clamp before upload.
                # Red Pitaya arbitrary data must be normalized to [-1, 1].
                tx_wave = np.clip(
                    self.volt_spin.value() * self.file_samples + self.offset_spin.value(),
                    -1.0,
                    1.0,
                )
                wave_str = ",".join(f"{v:.5f}" for v in tx_wave)
                self.rp.tx_txt(f"SOUR{chan}:TRAC:DATA:DATA {wave_str}")
                self.rp.tx_txt(f"SOUR{chan}:FUNC ARBITRARY")
                # Arbitrary samples are already scaled/offset, so keep generator gain neutral.
                self.rp.tx_txt(f"SOUR{chan}:VOLT 1.0")
                self.rp.tx_txt(f"SOUR{chan}:VOLT:OFFS 0.0")
            else:
                self.rp.tx_txt(f"SOUR{chan}:FUNC {mode}")
                self.rp.tx_txt(f"SOUR{chan}:VOLT {self.volt_spin.value()}")
                self.rp.tx_txt(f"SOUR{chan}:VOLT:OFFS {self.offset_spin.value()}")

            self.rp.tx_txt(f"SOUR{chan}:FREQ:FIX {self.freq_spin.value()}")
            self.rp.tx_txt(f"SOUR{chan}:BURS:STAT CONTINUOUS")
            self.rp.tx_txt(f"SOUR{chan}:TRig:SOUR INT")

            self.active_mode = mode
            self.active_freq = self.freq_spin.value()
            self.active_amplitude = self.volt_spin.value()
            self.active_offset = self.offset_spin.value()
            self.active_base_samples = self.build_mode_base_samples(mode)
            self.set_status(f"Applied {mode} waveform on channel {chan}")
            self.refresh_previews()
        except Exception as exc:
            QMessageBox.critical(self, "Apply failed", str(exc))
            self.set_status("Apply failed")

    def output_on(self) -> None:
        if not self.ensure_connected():
            return
        try:
            chan = self.chan
            self.rp.tx_txt(f"OUTPUT{chan}:STATE ON")
            self.rp.tx_txt(f"SOUR{chan}:TRig:INT")
            self.output_enabled = True
            self.set_status(f"Output CH{chan} ON")
            self.update_button_states()
        except Exception as exc:
            QMessageBox.critical(self, "Output ON failed", str(exc))

    def output_off(self) -> None:
        if not self.ensure_connected():
            return
        try:
            chan = self.chan
            self.rp.tx_txt(f"OUTPUT{chan}:STATE OFF")
            self.output_enabled = False
            self.set_status(f"Output CH{chan} OFF")
            self.update_button_states()
        except Exception as exc:
            QMessageBox.critical(self, "Output OFF failed", str(exc))

    def toggle_output(self) -> None:
        if not self.ensure_connected():
            return
        if self.output_enabled:
            self.output_off()
        else:
            self.output_on()


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 650)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
