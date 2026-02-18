import numpy as np
from redpitaya_scpi import scpi

IP = "192.168.0.6"
CHAN = 1
FILE = "shake_func.csv"

FREQ = 1000
VOLT = 1.0
OFFS = 0.0

print("Connecting to Red Pitaya at", IP)
rp = scpi(IP)
print("Connected to:", rp.idn_q())

# Reset generator + output state
rp.tx_txt("GEN:RST")
rp.tx_txt(f"OUTPUT{CHAN}:STATE OFF")

do_csv = False
if do_csv:
    # Load waveform (Red Pitaya expects -1..1)
    data = np.loadtxt(FILE, delimiter=",", dtype=float)
    data = np.clip(data, -1.0, 1.0)
    padded_data = np.zeros(1000)
    data = np.concatenate((padded_data, data, padded_data))

    print(f"Loaded {len(data)} samples from {FILE}")
    print(f"Data range: {data.min()} to {data.max()}")

    # Send ARB buffer
    wave_str = ",".join(f"{v:.5f}" for v in data)
    rp.tx_txt(f"SOUR{CHAN}:TRAC:DATA:DATA {wave_str}")

    # Select arbitrary waveform mode
    rp.tx_txt(f"SOUR{CHAN}:FUNC ARBITRARY")
else:
    rp.tx_txt(f"SOUR{CHAN}:FUNC SINE")  # proof of concept

# Set output params
rp.tx_txt(f"SOUR{CHAN}:FREQ:FIX {FREQ}")
rp.tx_txt(f"SOUR{CHAN}:VOLT {VOLT}")
rp.tx_txt(f"SOUR{CHAN}:VOLT:OFFS {OFFS}")

# Ensure continuous (i.e., NOT burst)
rp.tx_txt(f"SOUR{CHAN}:BURS:STAT CONTINUOUS")

# Use internal trigger source (fine for continuous)
rp.tx_txt(f"SOUR{CHAN}:TRig:SOUR INT")

# Enable output and (re)start from beginning of buffer
rp.tx_txt(f"OUTPUT{CHAN}:STATE ON")
rp.tx_txt(f"SOUR{CHAN}:TRig:INT")

rp.gen_get_settings(chan=CHAN)
