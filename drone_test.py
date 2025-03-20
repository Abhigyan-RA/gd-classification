import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal
from tqdm import tqdm

# Configuration parameters
feat_name = 'PSD'
t_seg = 20  # ms
n_per_seg = 1024
output_name = 'drones'
feat_format = 'ARR'
high_low = 'L'

def compute_psd(data, fs, n_per_seg, win_type='hamming'):
    """Compute Power Spectral Density using Welch's method"""
    fpsd, Pxx_den = signal.welch(data, fs, window=win_type, nperseg=n_per_seg)
    return fpsd, Pxx_den

def generate_fake_rf_signal(fs, duration_ms, carrier_freq=1e6, snr_db=20):
    """Generate synthetic RF signal with noise and modulation"""
    num_samples = int(duration_ms * fs / 1000)
    t = np.arange(num_samples) / fs
    
    # Create frequency-modulated signal
    modulation_freq = 50e3  # 50 kHz modulation
    mod_signal = np.sin(2 * np.pi * modulation_freq * t)
    
    # Complex signal with FM and noise
    fm_signal = np.exp(1j * (2 * np.pi * carrier_freq * t + mod_signal))
    
    # Add Gaussian noise based on SNR
    snr = 10 ** (snr_db / 10)
    noise_power = 1 / snr
    noise = np.sqrt(noise_power/2) * (np.random.randn(num_samples) + 1j*np.random.randn(num_samples))
    
    return fm_signal + noise

def acquire_and_process_data_fake(fs, t_seg, n_per_seg, batch_size=5):
    """Generate and process synthetic RF data"""
    processed_features = []
    n_samples = int(t_seg/1000 * fs)
    
    for _ in tqdm(range(batch_size), desc="Generating data batches"):
        # Generate synthetic RF signal
        raw_data = generate_fake_rf_signal(fs, t_seg)
        
        # Process magnitude (simulated I/Q to magnitude)
        data_magnitude = np.abs(raw_data)
        
        # Ensure correct length
        if len(data_magnitude) < n_samples:
            data_magnitude = np.pad(data_magnitude, (0, n_samples - len(data_magnitude)))
        
        # Compute PSD features
        _, psd = compute_psd(data_magnitude, fs, n_per_seg)
        processed_features.append(psd)
    
    return np.array(processed_features)

def load_model(model_path):
    """Load pre-trained classification model"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_drone_type(model, features):
    """Predict drone type from features"""
    try:
        result = model.predict(features)
        print(result)
        return result
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def display_results(predictions):
    """Show prediction results"""
    if predictions is None:
        print("No predictions available")
        return
    
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction Results:")
    print("-------------------")
    for pred, count in zip(unique, counts):
        print(f"Drone Type: {pred} - {count} samples ({count/len(predictions)*100:.2f}%)")
    
    most_common = unique[np.argmax(counts)]
    print(f"\nMost likely drone: {most_common}")

def main():
    # Model configuration
    model_path = 'drone_detector.pkl'
    fs = 40e6  # 40 MHz sample rate
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Generate and process synthetic data
    print("Generating and processing synthetic RF data...")
    features = acquire_and_process_data_fake(fs, t_seg, n_per_seg, batch_size=5)
    
    # Make predictions
    predictions = predict_drone_type(model, features)
    
    # Display results
    display_results(predictions)
    
    # Plot last PSD for visualization
    plt.figure(figsize=(12, 5))
    plt.plot(features[-1])
    plt.title('Synthetic Signal Power Spectral Density')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Power')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()