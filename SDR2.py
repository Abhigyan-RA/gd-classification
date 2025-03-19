import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import adi
import scipy.signal as signal
import time

# Initialize PlutoSDR
sdr = adi.Pluto()

# Transmitter configuration
sdr.tx_lo = int(2.4e9)  # Tx center frequency: 2.4 GHz
sdr.tx_rf_bandwidth = int(500e3)  # Tx bandwidth: 500 kHz
sdr.tx_hardwaregain_chan0 = -10  # Tx attenuation: 10 dB
sdr.tx_cyclic_buffer = True  # Enable cyclic buffer for continuous Tx

# Receiver configuration
sdr.rx_lo = int(2.4e9)  # Rx center frequency: 2.4 GHz
sdr.rx_rf_bandwidth = int(500e3)  # Rx bandwidth: 500 kHz
sdr.rx_buffer_size = 8192  # IQ samples per buffer
sdr.gain_control_mode_chan0 = "manual"  # Manual gain control
sdr.rx_hardwaregain_chan0 = 50  # Rx gain: 50 dB

# Common settings
sdr.sample_rate = int(1e6)  # Sample rate: 1 MSPS

# Create frequency hopping IQ signal
fs = sdr.sample_rate
N = 8192
t = np.arange(N) / fs

def create_freq_hopping_iq_signal(t, fs, duration=1.0):
    samples_per_hop = int(fs * duration / 10)
    iq_signal = np.zeros(len(t), dtype=complex)
    
    # Different frequency hops (Hz)
    hop_freqs = [-200e3, -100e3, 50e3, 150e3, 200e3]
    idx = 0
    
    # Generate IQ signal with frequency hops
    while idx < len(t):
        end_idx = min(idx + samples_per_hop, len(t))
        freq = hop_freqs[int(idx / samples_per_hop) % len(hop_freqs)]
        # I + jQ = cos(2πft) + jsin(2πft) = e^j2πft
        iq_signal[idx:end_idx] = np.exp(2j * np.pi * freq * t[idx:end_idx])
        idx = end_idx
    
    # Add some IQ noise
    iq_noise = 0.05 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    iq_signal += iq_noise
    
    # Normalize IQ amplitude
    return iq_signal / np.max(np.abs(iq_signal)) * 0.8

# Generate and transmit IQ signal
tx_iq_signal = create_freq_hopping_iq_signal(t, fs)
sdr.tx(tx_iq_signal)
print(f"Transmitting frequency hopping IQ signal with {len(tx_iq_signal)} samples")

# Set up visualization
fig = plt.figure(figsize=(12, 10))
fig.suptitle("PlutoSDR IQ Signal Analysis", fontsize=16)

# Create subplot grid
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2])
ax_i = fig.add_subplot(gs[0, 0])
ax_q = fig.add_subplot(gs[0, 1])
ax_iq = fig.add_subplot(gs[1, 0])
ax_fft = fig.add_subplot(gs[1, 1])
ax_waterfall = fig.add_subplot(gs[2, :])

# Waterfall plot settings
waterfall_rows = 100
waterfall_data = np.zeros((waterfall_rows, sdr.rx_buffer_size))
waterfall_img = ax_waterfall.imshow(
    waterfall_data,
    aspect='auto',
    extent=[-sdr.sample_rate/2/1e3, sdr.sample_rate/2/1e3, 0, waterfall_rows],
    cmap='viridis',
    origin='lower',
    vmin=-100,
    vmax=0
)

# FFT frequency axis (kHz)
freq = np.fft.fftshift(np.fft.fftfreq(sdr.rx_buffer_size, 1/sdr.sample_rate)) / 1e3

# Initialize plots
i_line, = ax_i.plot([], [])
q_line, = ax_q.plot([], [])
iq_scatter = ax_iq.scatter([], [], s=2)
fft_line, = ax_fft.plot(freq, np.zeros_like(freq))

# Set up plot labels and grids
ax_i.set_title('I Component (Real)')
ax_i.set_xlabel('Sample')
ax_i.set_ylabel('Amplitude')
ax_i.grid(True)

ax_q.set_title('Q Component (Imaginary)')
ax_q.set_xlabel('Sample')
ax_q.set_ylabel('Amplitude')
ax_q.grid(True)

ax_iq.set_title('IQ Constellation')
ax_iq.set_xlabel('I')
ax_iq.set_ylabel('Q')
ax_iq.grid(True)
ax_iq.set_aspect('equal')

ax_fft.set_title('IQ Signal Spectrum (FFT)')
ax_fft.set_xlabel('Frequency (kHz)')
ax_fft.set_ylabel('Power (dB)')
ax_fft.set_xlim([freq.min(), freq.max()])
ax_fft.set_ylim([-100, 0])
ax_fft.grid(True)

ax_waterfall.set_title('Spectrum Waterfall Display')
ax_waterfall.set_xlabel('Frequency (kHz)')
ax_waterfall.set_ylabel('Time')

# Add colorbar
cbar = fig.colorbar(waterfall_img, ax=ax_waterfall)
cbar.set_label('Power (dB)')

# Tight layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Function to calculate FFT of IQ signal
def calculate_fft(iq_signal):
    win = signal.blackmanharris(len(iq_signal))
    iq_windowed = iq_signal * win
    fft_data = np.fft.fftshift(np.fft.fft(iq_windowed))
    power_db = 20 * np.log10(np.abs(fft_data) / len(iq_signal) + 1e-10)
    return power_db

# Animation update function
def update_plot(frame):
    # Receive new IQ samples
    rx_iq_signal = sdr.rx()
    
    # Extract I and Q components
    i_component = np.real(rx_iq_signal)
    q_component = np.imag(rx_iq_signal)
    
    # Calculate FFT power spectrum
    power_db = calculate_fft(rx_iq_signal)
    
    # Update I component plot
    i_line.set_data(range(len(i_component)), i_component)
    ax_i.relim()
    ax_i.autoscale_view()
    
    # Update Q component plot
    q_line.set_data(range(len(q_component)), q_component)
    ax_q.relim()
    ax_q.autoscale_view()
    
    # Update IQ constellation plot
    iq_scatter.set_offsets(np.column_stack((i_component, q_component)))
    max_val = max(np.max(np.abs(i_component)), np.max(np.abs(q_component)))
    ax_iq.set_xlim([-max_val*1.1, max_val*1.1])
    ax_iq.set_ylim([-max_val*1.1, max_val*1.1])
    
    # Update FFT plot
    fft_line.set_ydata(power_db)
    
    # Update waterfall - roll the data up one row
    waterfall_data[:-1] = waterfall_data[1:]
    waterfall_data[-1] = power_db
    waterfall_img.set_array(waterfall_data)
    
    return i_line, q_line, iq_scatter, fft_line, waterfall_img

# Create animation
ani = FuncAnimation(fig, update_plot, interval=100, blit=True)

try:
    plt.show()
except KeyboardInterrupt:
    print("Visualization stopped by user")
finally:
    # Clean up resources
    sdr.tx_destroy_buffer()
    print("PlutoSDR resources released")