import numpy as np
import adi
import time
import csv
import matplotlib.pyplot as plt
from datetime import datetime

# Parameters for transmission and reception
sample_rate = 1e6  # 1 MSPS
center_freq = 2.4e9 # 915 MHz (ISM band)
bandwidth = 500e3  # 500 kHz
tx_gain = -10  # Transmit gain in dB
rx_gain = 30   # Receive gain in dB
buffer_size = 10000  # Number of samples per buffer
total_duration = 60  # Duration in seconds (1 minute)
num_buffers = int((total_duration * sample_rate) / buffer_size)

# Signal type: QPSK (good balance of bandwidth efficiency and error performance)
# Other options could be BPSK (simpler) or 16QAM (higher data rate)
def generate_qpsk_signal(num_symbols):
    # Generate random QPSK symbols (±1±1j)
    symbols = np.random.randint(0, 4, num_symbols)
    qpsk_signal = np.zeros(num_symbols, dtype=complex)
    
    # Map to QPSK constellation
    for i in range(num_symbols):
        if symbols[i] == 0:
            qpsk_signal[i] = 1 + 1j  # 1st quadrant
        elif symbols[i] == 1:
            qpsk_signal[i] = -1 + 1j  # 2nd quadrant
        elif symbols[i] == 2:
            qpsk_signal[i] = -1 - 1j  # 3rd quadrant
        else:
            qpsk_signal[i] = 1 - 1j  # 4th quadrant
    
    return qpsk_signal

def main():
    try:
        # Initialize the PlutoSDR
        sdr = adi.Pluto()
        
        # Configure Tx
        sdr.tx_rf_bandwidth = int(bandwidth)
        sdr.tx_lo = int(center_freq)
        sdr.tx_hardwaregain_chan0 = int(tx_gain)
        sdr.tx_buffer_size = buffer_size
        sdr.sample_rate = int(sample_rate)
        
        # Configure Rx
        sdr.rx_rf_bandwidth = int(bandwidth)
        sdr.rx_lo = int(center_freq)
        sdr.rx_gain_mode_chan0 = 'manual'
        sdr.rx_hardwaregain_chan0 = int(rx_gain)
        sdr.rx_buffer_size = buffer_size
        
        print(f"PlutoSDR configured for Tx/Rx at {center_freq/1e6} MHz")
        
        # Generate QPSK signal for transmission
        tx_signal = generate_qpsk_signal(buffer_size)
        tx_signal = tx_signal * 0.5  # Scale to avoid clipping
        
        # Prepare CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"pluto_sdr_iq_data_{timestamp}.csv"
        
        with open(csv_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write header
            csvwriter.writerow(['Sample', 'I', 'Q', 'Magnitude', 'Phase'])
            
            # Start transmission and reception
            print(f"Starting capture for {total_duration} seconds...")
            total_samples = 0
            
            start_time = time.time()
            
            for i in range(num_buffers):
                # Transmit the signal
                sdr.tx(tx_signal)
                
                # Receive samples
                rx_samples = sdr.rx()
                
                # Write IQ data to CSV
                for j, sample in enumerate(rx_samples):
                    sample_idx = i * buffer_size + j
                    i_val = np.real(sample)
                    q_val = np.imag(sample)
                    magnitude = np.abs(sample)
                    phase = np.angle(sample, deg=True)
                    
                    csvwriter.writerow([sample_idx, i_val, q_val, magnitude, phase])
                
                total_samples += len(rx_samples)
                
                # Print progress every 10% of the capture
                progress = (i + 1) / num_buffers * 100
                if (i + 1) % max(1, num_buffers // 10) == 0:
                    print(f"Progress: {progress:.1f}% ({total_samples} samples captured)")
            
            end_time = time.time()
            actual_duration = end_time - start_time
            
            print(f"Capture complete!")
            print(f"Total duration: {actual_duration:.2f} seconds")
            print(f"Total samples: {total_samples}")
            print(f"Effective sample rate: {total_samples/actual_duration:.2f} samples/second")
            print(f"Data saved to: {csv_filename}")
            
            # Plot a small portion of the received signal
            plot_samples = 1000
            t = np.arange(plot_samples) / sample_rate * 1000  # time in ms
            
            plt.figure(figsize=(12, 8))
            
            # Plot I and Q components
            plt.subplot(3, 1, 1)
            plt.plot(t, np.real(rx_samples[:plot_samples]), label='I')
            plt.plot(t, np.imag(rx_samples[:plot_samples]), label='Q')
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude')
            plt.title('I/Q Components')
            plt.legend()
            
            # Plot constellation
            plt.subplot(3, 1, 2)
            plt.scatter(np.real(rx_samples[:plot_samples]), np.imag(rx_samples[:plot_samples]), 
                       s=2, alpha=0.5)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            plt.xlabel('In-phase (I)')
            plt.ylabel('Quadrature (Q)')
            plt.title('Constellation Diagram')
            plt.grid(True)
            
            # Plot magnitude
            plt.subplot(3, 1, 3)
            plt.plot(t, np.abs(rx_samples[:plot_samples]))
            plt.xlabel('Time (ms)')
            plt.ylabel('Magnitude')
            plt.title('Signal Magnitude')
            
            plt.tight_layout()
            plt.savefig(f"pluto_sdr_iq_plot_{timestamp}.png")
            plt.show()
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if 'sdr' in locals():
            print("Closing PlutoSDR connection...")
            sdr = None

if __name__ == "__main__":
    main()