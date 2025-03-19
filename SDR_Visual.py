import numpy as np
import matplotlib.pyplot as plt
import adi
import time
import scipy.signal as signal

# Initialize PlutoSDR
sdr = adi.Pluto(uri="ip:192.168.2.1")

# Transmitter (Tx) configuration
sdr.tx_lo = int(2.4e9)  # Tx center frequency: 2.4 GHz
sdr.tx_rf_bandwidth = int(500e3)  # Tx bandwidth: 500 kHz
sdr.tx_hardwaregain_chan0 = -10  # Tx gain: -10 dB (to avoid saturation)
sdr.tx_cyclic_buffer = True  # Enable cyclic buffer for continuous Tx

# Receiver (Rx) configuration
sdr.rx_lo = int(2.4e9)  # Rx center frequency: 2.4 GHz
sdr.rx_rf_bandwidth = int(500e3)  # Rx bandwidth: 500 kHz
sdr.rx_buffer_size = int(8192)  # Number of IQ samples per buffer
sdr.gain_control_mode_chan0 = "manual"  # Manual gain control
sdr.rx_hardwaregain_chan0 = int(50)  # Rx gain: 50 dB

# Common settings
sdr.sample_rate = int(1e6)  # Sample rate: 1 MSPS (for both Tx and Rx)

# Generate IQ transmission data
fs = sdr.sample_rate
N = 8192  # Number of samples
t = np.arange(N) / fs

# Create multi-tone complex signal for I and Q components
f1, f2, f3 = 100e3, -150e3, 200e3  # Frequency components
# I(t) + jQ(t) = Aej2πft = A(cos(2πft) + jsin(2πft))
tx_iq_signal = 0.4 * np.exp(2j * np.pi * f1 * t) + 0.3 * np.exp(2j * np.pi * f2 * t) + 0.2 * np.exp(2j * np.pi * f3 * t)

# Normalize the IQ signal to avoid clipping
tx_iq_signal = tx_iq_signal / np.max(np.abs(tx_iq_signal)) * 0.8

# Transmit the IQ signal
sdr.tx(tx_iq_signal)
print(f"Transmitting IQ signal with {len(tx_iq_signal)} samples")

# Set up visualization
plt.ion()
fig, ((ax_i, ax_q), (ax_iq, ax_fft)) = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('PlutoSDR IQ Signal Analysis', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Function to calculate FFT of IQ signal
def calculate_fft(iq_signal, fs):
    N = len(iq_signal)
    # Window function to reduce spectral leakage
    win = signal.blackmanharris(N)
    iq_windowed = iq_signal * win
    
    # Calculate FFT of IQ signal
    fft_data = np.fft.fftshift(np.fft.fft(iq_windowed))
    # Calculate frequency axis (shifted to center)
    freq = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    
    # Calculate power spectrum in dB
    power_db = 20 * np.log10(np.abs(fft_data) / N + 1e-10)
    
    return freq, power_db

# Function to update plots
def update_plots():
    # Receive IQ samples
    rx_iq_signal = sdr.rx()
    
    # Extract I (real) and Q (imaginary) components
    i_component = np.real(rx_iq_signal)
    q_component = np.imag(rx_iq_signal)
    
    # Calculate FFT of received IQ signal
    freq, power_db = calculate_fft(rx_iq_signal, sdr.sample_rate)
    
    # Clear previous plots
    ax_i.clear()
    ax_q.clear()
    ax_iq.clear()
    ax_fft.clear()
    
    # Plot I component (real part)
    ax_i.plot(i_component)
    ax_i.set_title('I Component (Real)')
    ax_i.set_xlabel('Sample')
    ax_i.set_ylabel('Amplitude')
    ax_i.grid(True)
    
    # Plot Q component (imaginary part)
    ax_q.plot(q_component)
    ax_q.set_title('Q Component (Imaginary)')
    ax_q.set_xlabel('Sample')
    ax_q.set_ylabel('Amplitude')
    ax_q.grid(True)
    
    # IQ constellation plot
    ax_iq.plot(i_component, q_component, '.')
    ax_iq.set_title('IQ Constellation')
    ax_iq.set_xlabel('I')
    ax_iq.set_ylabel('Q')
    ax_iq.grid(True)
    ax_iq.set_aspect('equal')
    max_val = max(np.max(np.abs(i_component)), np.max(np.abs(q_component)))
    ax_iq.set_xlim([-max_val*1.1, max_val*1.1])
    ax_iq.set_ylim([-max_val*1.1, max_val*1.1])
    
    # FFT spectrum plot
    ax_fft.plot(freq/1e3, power_db)
    ax_fft.set_title('IQ Signal Spectrum (FFT)')
    ax_fft.set_xlabel('Frequency (kHz)')
    ax_fft.set_ylabel('Power (dB)')
    ax_fft.grid(True)
    ax_fft.set_xlim([freq.min()/1e3, freq.max()/1e3])
    
    # Refresh the plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.draw()
    plt.pause(0.05)

# Main loop
try:
    print("Receiving and visualizing IQ signals. Press Ctrl+C to stop...")
    while True:
        update_plots()
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Visualization stopped by user.")
finally:
    # Clean up
    sdr.tx_destroy_buffer()
    print("PlutoSDR resources released.")
    plt.ioff()
    plt.show()