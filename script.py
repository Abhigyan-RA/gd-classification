import numpy as np
import adi
import time
import matplotlib.pyplot as plt
from scipy import signal
import pickle

# --- Parameters for SDR Capture ---
sample_rate = 1e6      # 1 MSPS
center_freq = 2.4e9    # 2.4 GHz
bandwidth = 500e3      # 500 kHz
tx_gain = -10          # Transmit gain in dB
rx_gain = 30           # Receive gain in dB
buffer_size = 10000    # Number of samples per buffer
total_duration = 60    # Duration in seconds (for overall capture, if desired)

# --- Parameters for Feature Extraction & Prediction ---
feat_name = 'SPEC'     # Use 'SPEC' for spectrum in dB (matching training)
n_per_seg = 1024       # Number of samples per segment for PSD computation
window_size = 5        # Window size for moving average filter
t_seg = 20             # Segment duration per band in milliseconds

# Calculate number of samples per band
samples_per_band = int(t_seg / 1000 * sample_rate)
# Each segment consists of low and high bands => total required samples per segment:
segment_length = 2 * samples_per_band

def moving_average_filter(data, window_size=5):
    """
    Applies a moving average filter (FIR filter with uniform weights)
    to smooth the input data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def extract_dronerf_features(signal_low, signal_high, fs, n_per_seg, feat_name='SPEC', to_norm=True):
    """
    Compute PSD features for low and high bands using Welch's method,
    optionally convert to dB (if feat_name is 'SPEC'), and normalize.
    The two PSDs are concatenated to form a single feature vector.
    
    Parameters:
      signal_low: low band signal (1D numpy array)
      signal_high: high band signal (1D numpy array)
      fs: sampling rate in Hz
      n_per_seg: number of samples per segment (for Welch's method)
      feat_name: if 'PSD' returns linear PSD; if 'SPEC', converts to dB
      to_norm: if True, normalizes the final feature vector
      
    Returns:
      Feature vector (1D numpy array)
    """
    # Compute PSD for low band
    freqs_low, psd_low = signal.welch(signal_low, fs, nperseg=n_per_seg)
    # Compute PSD for high band
    freqs_high, psd_high = signal.welch(signal_high, fs, nperseg=n_per_seg)
    
    # Concatenate PSD features
    feat = np.concatenate((psd_low, psd_high))
    
    # Convert to dB if feat_name is 'SPEC'
    if feat_name == 'SPEC':
        feat = -10 * np.log10(feat + 1e-8)  # add epsilon to avoid log(0)
    
    # Normalize feature vector if required
    if to_norm and np.max(feat) != 0:
        feat = feat / np.max(feat)
    
    return feat

def load_model(model_path):
    """Load a trained classification model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def configure_sdr(center_freq=2.4e9, sample_rate=1e6, rx_buffer_size=10000, rx_gain=30):
    """Configure and return a PlutoSDR object."""
    try:
        print("Attempting to connect to SDR at ip:192.168.2.1...")
        sdr = adi.Pluto(uri="ip:192.168.2.1")
        print("Initial connection established")
        
        # Set common parameters
        sdr.sample_rate = int(sample_rate)
        
        # Configure Tx
        sdr.tx_rf_bandwidth = int(bandwidth)
        sdr.tx_lo = int(center_freq)
        sdr.tx_hardwaregain_chan0 = int(tx_gain)
        sdr.tx_buffer_size = buffer_size
        
        # Configure Rx
        sdr.rx_rf_bandwidth = int(bandwidth)
        sdr.rx_lo = int(center_freq)
        sdr.rx_gain_mode_chan0 = 'manual'
        sdr.rx_hardwaregain_chan0 = int(rx_gain)
        sdr.rx_buffer_size = rx_buffer_size
        
        print("SDR configured successfully")
        return sdr
    except Exception as e:
        print(f"Error connecting to SDR: {e}")
        return None

def main():
  
    model_path = 'drone_detector_stratified_model.pkl'
    model = load_model(model_path)
    if model is None:
        return
    
    # Configure SDR
    sdr = configure_sdr(center_freq=center_freq, sample_rate=sample_rate, rx_buffer_size=buffer_size, rx_gain=rx_gain)
    if sdr is None:
        return
    
    print("Starting continuous prediction. Press Ctrl+C to stop.")
    
    segment_buffer = np.array([])  # buffer to accumulate samples for one segment
    
    try:
        while True:
            # Acquire new samples from SDR
            rx_samples = sdr.rx()
            # Convert complex IQ samples to magnitude (or use your preferred column)
            new_data = np.abs(rx_samples)
            # Append new data to the segment buffer
            segment_buffer = np.concatenate((segment_buffer, new_data))
            
            # Check if we have enough samples to form a complete segment
            if len(segment_buffer) >= segment_length:
                # Extract one segment and update the buffer
                current_segment = segment_buffer[:segment_length]
                segment_buffer = segment_buffer[segment_length:]
                
                # Split the segment into low and high bands
                low_band = current_segment[:samples_per_band]
                high_band = current_segment[samples_per_band:segment_length]
                
                # Apply moving average filter
                filtered_low = moving_average_filter(low_band, window_size)
                filtered_high = moving_average_filter(high_band, window_size)
                
                # Extract feature vector
                feature_vector = extract_dronerf_features(filtered_low, filtered_high, sample_rate, n_per_seg, feat_name=feat_name, to_norm=True)
                feature_vector = feature_vector.reshape(1, -1)
                
                # Make prediction
                pred = model.predict(feature_vector)
                print(f"Segment Prediction: {pred[0]}")
                
            time.sleep(0.05)  # short delay to avoid hogging CPU
            
    except KeyboardInterrupt:
        print("Stopping continuous prediction.")
    
    finally:
        # Cleanup SDR connection
        sdr = None

if __name__ == "__main__":
    main()
