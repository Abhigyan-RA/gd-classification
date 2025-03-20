import numpy as np
import pickle
import adi
import time
from scipy import signal
from tqdm import tqdm

# Configuration parameters
feat_name = 'SPEC'    # Use 'SPEC' for spectrum in dB (matching training)
t_seg = 20            # Segment duration in milliseconds per band
n_per_seg = 1024      # Number of samples per segment for PSD computation
fs = 40e6             # Sampling rate (should match training)

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

def configure_sdr(center_freq=2.4e9, sample_rate=40e6, rx_buffer_size=1600000, rx_gain=70):
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

def acquire_and_process_data(sdr, fs, t_seg, n_per_seg):
    """
    Acquire data from the SDR, apply a moving average filter,
    split it into two halves to simulate low and high bands, and extract PSD-based features.
    """
    n_samples = int(t_seg / 1000 * fs)
    
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
    
    # Apply moving average filter for consistency with training
    low_band_filtered = moving_average_filter(low_band, window_size=5)
    high_band_filtered = moving_average_filter(high_band, window_size=5)
    
    # Extract combined PSD features from both bands
    features = extract_dronerf_features(low_band_filtered, high_band_filtered, fs, n_per_seg, feat_name=feat_name, to_norm=True)
    
    return features.reshape(1, -1)  # Reshape for single sample prediction

def predict_drone_presence(model, features):
    """Predict drone presence using the trained model."""
    try:
        prediction = model.predict(features)
        return prediction[0]  # Return as scalar instead of array
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def main():
    model_path = 'drone_detector-2.pkl'
    window_size = 10  # Number of predictions to track
    prediction_history = []
    
    # Load the trained model
    model = load_model(model_path)
    if model is None:
        return
    
    # Configure the SDR device
    sdr = configure_sdr()
    if sdr is None:
        return
    
    print("\nStarting continuous drone detection...")
    print("Press Ctrl+C to stop the monitoring\n")
    
    try:
        while True:
            # Acquire and process data
            features = acquire_and_process_data(sdr, fs, t_seg, n_per_seg)
            
            # Make prediction
            prediction = predict_drone_presence(model, features)
            
            # Add to history and maintain window size
            prediction_history.append(prediction)
            if len(prediction_history) > window_size:
                prediction_history.pop(0)
            
            # Calculate current status based on majority vote
            if prediction_history:
                drone_count = sum(prediction_history)
                total_count = len(prediction_history)
                drone_percentage = (drone_count / total_count) * 100
                
                # Clear the current line and print status
                print("\r", end="")
                if drone_count > total_count / 2:
                    print(f"DRONE DETECTED! ({drone_percentage:.1f}% confidence) - {drone_count}/{total_count} positive detections", end="")
                else:
                    print(f"No drone detected ({100-drone_percentage:.1f}% confidence) - {total_count-drone_count}/{total_count} negative detections", end="")
            
            time.sleep(0.5)  # Short delay between readings
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
    finally:
        print("\nShutting down SDR...")
        del sdr  # Clean up SDR resources

if __name__ == "__main__":
    main()