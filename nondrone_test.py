import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal
from tqdm import tqdm

# ------------------------------
# Feature Extraction Functions
# ------------------------------
def compute_psd(data, fs, n_per_seg, win_type='hamming'):
    """Compute Power Spectral Density using Welch's method."""
    freqs, psd = signal.welch(data, fs, window=win_type, nperseg=n_per_seg)
    return freqs, psd

def extract_dronerf_features(signal_low, signal_high, fs, n_per_seg, feat_name='SPEC', to_norm=True):
    """
    Extract features from paired low and high band signals similar to DroneRFTorch.

    Parameters:
        signal_low (np.array): The low band RF signal.
        signal_high (np.array): The high band RF signal.
        fs (float): Sampling rate in Hz.
        n_per_seg (int): Number of samples per segment (used for Welch's PSD computation).
        feat_name (str): Feature type ('PSD' or 'SPEC' for spectrum in dB).
        to_norm (bool): Normalize the output feature by its maximum value.

    Returns:
        np.array: Concatenated feature vector for both bands.
    """
    f_low, psd_low = signal.welch(signal_low, fs, nperseg=n_per_seg)
    f_high, psd_high = signal.welch(signal_high, fs, nperseg=n_per_seg)
    
    feat = np.concatenate((psd_low, psd_high))

    if feat_name == 'SPEC':
        feat = -10 * np.log10(feat + 1e-8)  # Avoid log(0) issues

    if to_norm and np.max(feat) != 0:
        feat = feat / np.max(feat)

    return feat

# ------------------------------
# Synthetic Data Generation
# ------------------------------
def generate_fake_rf_signal(fs, duration_ms, carrier_freq=1e6, snr_db=20, drone_present=True):
    """
    Generate synthetic RF signals with noise.

    Parameters:
        drone_present (bool): If True, includes frequency-modulated carrier. If False, only noise.

    Returns:
        Tuple of (low band signal, high band signal)
    """
    num_samples = int(duration_ms * fs / 1000)
    t = np.arange(num_samples) / fs
    if drone_present:
        modulation_freq = 50e3
        mod_signal = np.sin(2 * np.pi * modulation_freq * t)
        base_signal_low = np.exp(1j * (2 * np.pi * carrier_freq * t + mod_signal))
        base_signal_high = np.exp(1j * (2 * np.pi * (carrier_freq + 1e6) * t + mod_signal))
    else:
        base_signal_low = np.zeros(num_samples, dtype=complex)
        base_signal_high = np.zeros(num_samples, dtype=complex)

    snr = 10 ** (snr_db / 10)
    noise_power = 1 / snr
    noise_low = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    noise_high = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

    return base_signal_low + noise_low, base_signal_high + noise_high
def moving_average_filter(data, window_size=5):
    """
    Applies a moving average filter (FIR filter with uniform weights)
    to smooth the input data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
def acquire_and_process_data_fake(fs, t_seg, n_per_seg, batch_size=5, drone_present=True):
    """
    Generate and process synthetic RF data.

    Parameters:
        drone_present (bool): True for drone activity, False for background noise.

    Returns:
        np.array: Feature vectors.
    """
    processed_features = []
    for _ in tqdm(range(batch_size), desc="Generating data batches"):
        raw_low, raw_high = generate_fake_rf_signal(fs, t_seg, drone_present=drone_present)
        raw_high = moving_average_filter(raw_high, window_size=5)
        raw_low = moving_average_filter(raw_low, window_size=5)
        features = extract_dronerf_features(np.abs(raw_low), np.abs(raw_high), fs, n_per_seg, feat_name='SPEC', to_norm=True)
        processed_features.append(features)
    return np.array(processed_features)

# ------------------------------
# Model Loading and Prediction
# ------------------------------
def load_model(model_path):
    """Load a trained classification model."""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print("Error loading model:", e)
        return None

def predict_drone_presence(model, features):
    """Predict drone presence using the trained model."""
    try:
        predictions = model.predict(features)
        print("Predictions:", predictions)
        return predictions
    except Exception as e:
        print("Prediction error:", e)
        return None

def display_results(predictions):
    """Display prediction results."""
    if predictions is None:
        print("No predictions available.")
        return
    
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction Results:")
    for label, count in zip(unique, counts):
        label_str = "Drone Present" if label == 1 else "No Drone"
        print(f"{label_str}: {count} samples ({count / len(predictions) * 100:.2f}%)")
    
    most_common = unique[np.argmax(counts)]
    print(f"Most likely condition: {'Drone Present' if most_common == 1 else 'No Drone'}")

# ------------------------------
# Main Test Script
# ------------------------------
def main():
    # Configuration parameters
    model_path = 'drone_detector-2.pkl'
    fs = 40e6          # Sampling rate: 40 MHz
    t_seg = 20         # Segment duration in milliseconds
    n_per_seg = 1024   # PSD segment size
    batch_size = 5     # Number of test samples

    simulate_drone = False  # Change to True to simulate drone presence

    # Load the trained model
    model = load_model(model_path)
    if model is None:
        return
    
    # Generate and process synthetic RF data
    print("Generating and processing synthetic RF data...")
    features = acquire_and_process_data_fake(fs, t_seg, n_per_seg, batch_size=batch_size, drone_present=simulate_drone)

    # Predict drone presence
    predictions = predict_drone_presence(model, features)

    # Display results
    display_results(predictions)

    # Visualize PSD of a sample segment
    raw_low, raw_high = generate_fake_rf_signal(fs, t_seg, drone_present=simulate_drone)
    _, psd_low = compute_psd(np.abs(raw_low), fs, n_per_seg)
    _, psd_high = compute_psd(np.abs(raw_high), fs, n_per_seg)

    plt.figure(figsize=(12, 5))
    plt.semilogy(psd_low, label='Low Band')
    plt.semilogy(psd_high, label='High Band')
    plt.title('Power Spectral Density (No Drone Present)' if not simulate_drone else 'Power Spectral Density (Drone Present)')
    plt.xlabel('Frequency Bin')
    plt.ylabel('PSD (dB/Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
