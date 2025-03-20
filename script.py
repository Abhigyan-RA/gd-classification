import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import adi
import time
from scipy import signal
from tqdm import tqdm

# Configuration parameters
feat_name = 'SPEC'    # Use 'SPEC' for spectrum in dB (matching training)
t_seg = 20            # Segment duration in milliseconds per band
n_per_seg = 1024      # Number of samples per segment for PSD computation
output_name = 'drones'
feat_format = 'ARR'
# For SDR testing, we assume the full received block covers both low and high bands

def compute_psd(data, fs, n_per_seg, win_type='hamming'):
    """Compute Power Spectral Density using Welch's method."""
    freqs, psd = signal.welch(data, fs, window=win_type, nperseg=n_per_seg)
    return freqs, psd

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
    f_low, psd_low = signal.welch(signal_low, fs, nperseg=n_per_seg)
    f_high, psd_high = signal.welch(signal_high, fs, nperseg=n_per_seg)
    
    feat = np.concatenate((psd_low, psd_high))
    
    if feat_name == 'SPEC':
        feat = -10 * np.log10(feat + 1e-8)  # Avoid log(0)
    
    if to_norm and np.max(feat) != 0:
        feat = feat / np.max(feat)
    
    return feat

def load_model(model_path):
    """Load a trained classification model."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def configure_sdr(center_freq=2.4e9, sample_rate=30.72e6, rx_buffer_size=1600000, rx_gain=70):
    """Configure and return a PlutoSDR object."""
    try:
        print("Attempting to connect to SDR at ip:192.168.2.1...")
        sdr = adi.Pluto(uri="ip:192.168.2.1")
        print("Initial connection established")
        
        try:
            sdr.sample_rate = int(sample_rate)
            print("Sample rate set successfully:", int(sample_rate))
        except Exception as e:
            print(f"Error setting sample rate: {e}")
        
        try:
            sdr.rx_rf_bandwidth = int(sample_rate)
            print("RF bandwidth set successfully:", int(sample_rate))
        except Exception as e:
            print(f"Error setting RF bandwidth: {e}")
        
        try:
            sdr.rx_lo = int(center_freq)
            print("Center frequency set successfully:", int(center_freq))
        except Exception as e:
            print(f"Error setting center frequency: {e}")
        
        try:
            sdr.rx_buffer_size = rx_buffer_size
            print("Buffer size set successfully:", rx_buffer_size)
        except Exception as e:
            print(f"Error setting buffer size: {e}")
        
        try:
            sdr.gain_control_mode_chan0 = 'manual'
            print("Gain control mode set to manual")
        except Exception as e:
            print(f"Error setting gain control mode: {e}")
        
        try:
            sdr.rx_hardwaregain_chan0 = rx_gain
            print("Hardware gain set successfully:", rx_gain)
        except Exception as e:
            print(f"Error setting hardware gain: {e}")
        
        print("SDR configured successfully")
        return sdr
    except Exception as e:
        print(f"Error connecting to SDR: {e}")
        return None
def moving_average_filter(data, window_size=5):
    """
    Applies a moving average filter (FIR filter with uniform weights)
    to smooth the input data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
def acquire_and_process_data(sdr, fs, t_seg, n_per_seg, batch_size=5):
    """
    Acquire data from the SDR, apply a moving average filter,
    split it into two halves to simulate low and high bands, and extract PSD-based features.
    
    Note: This function assumes that the SDR returns at least 2*n_samples samples,
    so that the first half represents the low band and the second half the high band.
    """
    n_samples = int(t_seg / 1000 * fs)
    processed_features = []
    
    for i in tqdm(range(batch_size), desc="Acquiring data batches"):
        raw_data = sdr.rx()  # Acquire raw complex IQ data from SDR
        
        total_required = 2 * n_samples
        if len(raw_data) < total_required:
            print(f"Warning: Received {len(raw_data)} samples; expected {total_required}. Padding...")
            raw_data = np.pad(raw_data, (0, total_required - len(raw_data)))
        else:
            raw_data = raw_data[:total_required]
        
        # Split into low and high band signals (take absolute value)
        low_band = np.abs(raw_data[:n_samples])
        high_band = np.abs(raw_data[n_samples:2 * n_samples])
        
        # *** Apply moving average filter for consistency with training ***
        low_band_filtered = moving_average_filter(low_band, window_size=5)
        high_band_filtered = moving_average_filter(high_band, window_size=5)
        
        # Extract combined PSD features from both bands
        features = extract_dronerf_features(low_band_filtered, high_band_filtered, fs, n_per_seg, feat_name=feat_name, to_norm=True)
        processed_features.append(features)
        
        time.sleep(0.1)
    
    return np.array(processed_features)


def predict_drone_type(model, features):
    """Predict drone presence using the trained model."""
    try:
        predictions = model.predict(features)
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def display_results(predictions):
    """Display prediction results."""
    if predictions is None:
        print("No predictions available")
        return
    unique_preds, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction Results:")
    print("-------------------")
    for pred, count in zip(unique_preds, counts):
        percentage = (count / len(predictions)) * 100
        print(f"Drone Presence: {pred} - {count} samples ({percentage:.2f}%)")
    
    most_common_idx = np.argmax(counts)
    print(f"\nMost likely condition: {'Drone Present' if unique_preds[most_common_idx]==1 else 'No Drone'}")

def main():
    model_path = 'drone_detector-2.pkl'
    fs = 40e6          # Sampling rate (should match training)
    t_seg = 20         # Segment duration (ms) per band
    batch_size = 5     # Number of test samples
    
    # Load the trained model
    model = load_model(model_path)
    if model is None:
        return
    
    # Configure the SDR device
    sdr = configure_sdr(sample_rate=fs)
    if sdr is None:
        return
    
    print("\nAcquiring data from SDR and processing...")
    features = acquire_and_process_data(sdr, fs, t_seg, n_per_seg, batch_size=batch_size)
    print(f"Processed feature shape: {features.shape}")
    
    # Make predictions
    predictions = predict_drone_type(model, features)
    display_results(predictions)
    
    # For visualization: plot PSD of a sample segment from the latest SDR acquisition
    raw_data = sdr.rx()
    total_required = 2 * int(t_seg/1000 * fs)
    if len(raw_data) < total_required:
        raw_data = np.pad(raw_data, (0, total_required - len(raw_data)))
    else:
        raw_data = raw_data[:total_required]
    low_band = np.abs(raw_data[:int(t_seg/1000 * fs)])
    high_band = np.abs(raw_data[int(t_seg/1000 * fs):2*int(t_seg/1000 * fs)])
    
    freqs_low, psd_low = compute_psd(low_band, fs, n_per_seg)
    freqs_high, psd_high = compute_psd(high_band, fs, n_per_seg)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(psd_low, label='Low Band')
    plt.semilogy(psd_high, label='High Band')
    plt.title('Power Spectral Density from SDR')
    plt.xlabel('Frequency Bin')
    plt.ylabel('PSD (dB/Hz)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
