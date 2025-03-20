import numpy as np
import pickle
import time
import argparse
from scipy import signal
import os

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

def generate_drone_signal(duration_ms, fs, drone_type="phantom"):
    """
    Generate synthetic drone RF signals based on typical drone RF characteristics.
    
    Parameters:
    - duration_ms: Duration of the signal in milliseconds
    - fs: Sampling rate in Hz
    - drone_type: Type of drone to simulate ('phantom', 'ar', 'bebop')
    
    Returns:
    - Tuple of (low_band, high_band) signals
    """
    n_samples = int(duration_ms / 1000 * fs)
    
    # Base signal is white Gaussian noise
    base_low = np.random.normal(0, 1, n_samples)
    base_high = np.random.normal(0, 1, n_samples)
    
    # Add time-domain characteristics based on drone type
    t = np.linspace(0, duration_ms/1000, n_samples)
    
    if drone_type.lower() == "phantom":
        # DJI Phantom typically has strong periodic components around 2.4GHz band
        # Add multiple harmonic components
        low_band = base_low + 3 * np.sin(2 * np.pi * 50 * t) + 2 * np.sin(2 * np.pi * 120 * t)
        high_band = base_high + 4 * np.sin(2 * np.pi * 80 * t) + 2.5 * np.sin(2 * np.pi * 200 * t)
        
    elif drone_type.lower() == "ar":
        # Parrot AR Drone has different signature
        # Add characteristic bursts and dips
        low_band = base_low + 2.5 * np.sin(2 * np.pi * 35 * t) * np.sin(2 * np.pi * 5 * t)
        high_band = base_high + 3.5 * np.sin(2 * np.pi * 70 * t) * np.cos(2 * np.pi * 7 * t)
        
    elif drone_type.lower() == "bebop":
        # Bebop has yet another signature
        # Add more complex modulation pattern
        low_band = base_low + 3 * np.sin(2 * np.pi * 40 * t) + np.sin(2 * np.pi * 60 * t) * np.cos(2 * np.pi * 3 * t)
        high_band = base_high + 2 * np.sin(2 * np.pi * 90 * t) + 2 * np.sin(2 * np.pi * 130 * t)
    
    else:  # Default/generic drone
        # Generic drone signature with typical control signal characteristics
        low_band = base_low + 3 * np.sin(2 * np.pi * 45 * t) + 1.5 * np.sin(2 * np.pi * 95 * t)
        high_band = base_high + 2.5 * np.sin(2 * np.pi * 75 * t) + 2 * np.sin(2 * np.pi * 125 * t)
    
    # Take absolute value to simulate power readings
    return np.abs(low_band), np.abs(high_band)

def generate_background_signal(duration_ms, fs, noise_type="wifi"):
    """
    Generate synthetic background RF signals.
    
    Parameters:
    - duration_ms: Duration of the signal in milliseconds
    - fs: Sampling rate in Hz
    - noise_type: Type of background noise ('wifi', 'bluetooth', 'empty', 'mixed')
    
    Returns:
    - Tuple of (low_band, high_band) signals
    """
    n_samples = int(duration_ms / 1000 * fs)
    t = np.linspace(0, duration_ms/1000, n_samples)
    
    # Base noise level
    base_low = np.random.normal(0, 0.5, n_samples)
    base_high = np.random.normal(0, 0.5, n_samples)
    
    if noise_type.lower() == "wifi":
        # WiFi has periodic burst patterns
        wifi_bursts = np.zeros(n_samples)
        burst_positions = np.random.choice(n_samples - 100, 10)
        for pos in burst_positions:
            wifi_bursts[pos:pos+100] = 3 * np.sin(np.linspace(0, np.pi, 100))
        
        low_band = base_low + 0.5 * wifi_bursts
        high_band = base_high + 0.7 * np.roll(wifi_bursts, 50)
        
    elif noise_type.lower() == "bluetooth":
        # Bluetooth has frequency hopping characteristics
        hop_rate = int(n_samples / 20)  # 20 hops over the duration
        hop_pattern_low = np.zeros(n_samples)
        hop_pattern_high = np.zeros(n_samples)
        
        for i in range(0, n_samples, hop_rate):
            end = min(i + hop_rate, n_samples)
            freq_low = np.random.uniform(30, 70)
            freq_high = np.random.uniform(100, 150)
            hop_pattern_low[i:end] = np.sin(2 * np.pi * freq_low * t[i:end])
            hop_pattern_high[i:end] = np.sin(2 * np.pi * freq_high * t[i:end])
        
        low_band = base_low + 0.8 * hop_pattern_low
        high_band = base_high + 0.9 * hop_pattern_high
        
    elif noise_type.lower() == "empty":
        # Just thermal noise with slight random spikes
        spike_positions_low = np.random.choice(n_samples, 5)
        spike_positions_high = np.random.choice(n_samples, 5)
        
        low_band = base_low.copy()
        high_band = base_high.copy()
        
        for pos in spike_positions_low:
            pos = min(pos, n_samples - 10)
            low_band[pos:pos+10] += np.random.uniform(0.5, 1.5)
            
        for pos in spike_positions_high:
            pos = min(pos, n_samples - 10)
            high_band[pos:pos+10] += np.random.uniform(0.5, 1.5)
            
    else:  # mixed or default
        # Combination of different noise types
        # Some WiFi-like bursts
        wifi_bursts = np.zeros(n_samples)
        burst_positions = np.random.choice(n_samples - 50, 5)
        for pos in burst_positions:
            wifi_bursts[pos:pos+50] = 2 * np.sin(np.linspace(0, np.pi, 50))
        
        # Some frequency hopping
        hop_pattern = np.zeros(n_samples)
        for i in range(0, n_samples, int(n_samples/10)):
            end = min(i + int(n_samples/10), n_samples)
            freq = np.random.uniform(40, 120)
            hop_pattern[i:end] = 0.8 * np.sin(2 * np.pi * freq * t[i:end])
        
        low_band = base_low + 0.4 * wifi_bursts + 0.3 * hop_pattern
        high_band = base_high + 0.3 * np.roll(wifi_bursts, 30) + 0.5 * np.roll(hop_pattern, 50)
    
    # Take absolute value to simulate power readings
    return np.abs(low_band), np.abs(high_band)

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

def predict_drone_presence(model, features):
    """Predict drone presence using the trained model."""
    try:
        prediction = model.predict(features)
        proba = model.predict_proba(features)
        return prediction[0], proba[0]
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Test drone detection with synthetic RF signals')
    parser.add_argument('--model', type=str, default='drone_detector_stratified_model.pkl', help='Path to the trained model')
    parser.add_argument('--mode', type=str, default='drone', choices=['drone', 'background', 'alternating'], 
                        help='Signal generation mode')
    parser.add_argument('--drone_type', type=str, default='phantom', choices=['phantom', 'ar', 'bebop', 'generic'],
                        help='Type of drone to simulate')
    parser.add_argument('--noise_type', type=str, default='mixed', choices=['wifi', 'bluetooth', 'empty', 'mixed'],
                        help='Type of background noise')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--interval', type=float, default=0.5, help='Sampling interval in seconds')
    args = parser.parse_args()
    
    # Load the trained model
    model = load_model(args.model)
    if model is None:
        return
    
    print(f"\nStarting synthetic signal test in {args.mode} mode")
    print(f"Duration: {args.duration} seconds, Interval: {args.interval} seconds")
    
    if args.mode == 'drone':
        print(f"Simulating {args.drone_type} drone signals")
    elif args.mode == 'background':
        print(f"Simulating {args.noise_type} background noise")
    else:  # alternating
        print(f"Alternating between {args.drone_type} drone and {args.noise_type} background")
    
    print("\nPress Ctrl+C to stop the test\n")
    
    try:
        start_time = time.time()
        end_time = start_time + args.duration
        
        drone_phase = True  # For alternating mode
        switch_time = start_time + 5  # Switch every 5 seconds in alternating mode
        
        while time.time() < end_time:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Determine what signal to generate
            generate_drone = args.mode == 'drone'
            
            if args.mode == 'alternating':
                if current_time >= switch_time:
                    drone_phase = not drone_phase
                    switch_time = current_time + 5
                    print(f"\n[{elapsed:.1f}s] Switching to {'DRONE' if drone_phase else 'BACKGROUND'} signal")
                    
                generate_drone = drone_phase
            
            # Generate appropriate signal
            if generate_drone:
                low_band, high_band = generate_drone_signal(t_seg, fs, args.drone_type)
                true_label = 1
            else:
                low_band, high_band = generate_background_signal(t_seg, fs, args.noise_type)
                true_label = 0
            
            # Apply filtering just like in the real data processing
            low_band_filtered = moving_average_filter(low_band, window_size=5)
            high_band_filtered = moving_average_filter(high_band, window_size=5)
            
            # Extract features
            features = extract_dronerf_features(low_band_filtered, high_band_filtered, fs, n_per_seg, feat_name=feat_name)
            features = features.reshape(1, -1)  # Reshape for model input
            
            # Make prediction
            prediction, probabilities = predict_drone_presence(model, features)
            
            # Format and print the result
            accuracy = "CORRECT" if prediction == true_label else "INCORRECT"
            if prediction == 1:
                confidence = probabilities[1] * 100
                status = f"DRONE DETECTED! ({confidence:.1f}% confidence) - {accuracy}"
            else:
                confidence = probabilities[0] * 100
                status = f"No drone detected ({confidence:.1f}% confidence) - {accuracy}"
            
            print(f"\r[{elapsed:.1f}s] {status}", end="")
            
            # Sleep until next sample
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nTest stopped by user")
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")

if __name__ == "__main__":
    main()