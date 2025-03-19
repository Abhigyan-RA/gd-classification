import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import adi
import time
from scipy import signal
from tqdm import tqdm
import torch


feat_name = 'PSD'
t_seg = 20  # ms
n_per_seg = 1024
output_name = 'drones'
feat_format = 'ARR'
high_low = 'L'  


def compute_psd(data, fs, n_per_seg, win_type='hamming'):
    fpsd, Pxx_den = signal.welch(data, fs, window=win_type, nperseg=n_per_seg)
    return fpsd, Pxx_den


def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
           
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def configure_sdr(center_freq=2.4e9, sample_rate=40e6, rx_buffer_size=10000, rx_gain=70):
    try:
        sdr = adi.Pluto(uri="ip:192.168.2.1")
        sdr.sample_rate = int(sample_rate)  # 40 MHz to match DroneRF dataset
        sdr.rx_rf_bandwidth = int(sample_rate)
        sdr.rx_lo = int(center_freq)  # Center frequency
        sdr.rx_buffer_size = rx_buffer_size
        sdr.gain_control_mode_chan0 = 'manual'
        sdr.rx_hardwaregain_chan0 = rx_gain
        print("SDR configured successfully")
        return sdr
    except Exception as e:
        print(f"Error configuring SDR: {e}")
        return None


def acquire_and_process_data(sdr, fs, t_seg, n_per_seg, batch_size=5):
  
    n_samples = int(t_seg/1000 * fs) 
    
    
    processed_features = []
    
    for i in tqdm(range(batch_size), desc="Acquiring data batches"):
  
        raw_data = sdr.rx()
        
        
        if high_low == 'L':
           
            data_magnitude = np.abs(raw_data)
        else:
          
            data_magnitude = np.abs(raw_data)
        
      
        if len(data_magnitude) < n_samples:
            print(f"Warning: Received fewer samples than expected ({len(data_magnitude)} < {n_samples})")
        
            data_magnitude = np.pad(data_magnitude, (0, n_samples - len(data_magnitude)))
        
    
        _, psd = compute_psd(data_magnitude, fs, n_per_seg)
        
       
        processed_features.append(psd)
        
    
        time.sleep(0.1)
    
    return np.array(processed_features)

def predict_drone_type(model, features):
    try:
        predictions = model.predict(features)
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None


def display_results(predictions, confidence=None):
    if predictions is None:
        print("No predictions available")
        return
    
    unique_preds, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction Results:")
    print("-------------------")
    for pred, count in zip(unique_preds, counts):
        percentage = (count / len(predictions)) * 100
        print(f"Drone Type: {pred} - {count} samples ({percentage:.2f}%)")
    
   
    most_common_idx = np.argmax(counts)
    print(f"\nMost likely drone: {unique_preds[most_common_idx]}")

def main():
    
    model_path = 'dronerf_SVM_PSD_1024_20_1.pkl'
    
    # Sample rate of SDR (should match what was used in training)
    fs = 40e6  # 40 MHz
    
 
    model = load_model(model_path)
    if model is None:
        return
    
    # Configure the SDR
    sdr = configure_sdr(sample_rate=fs)
    if sdr is None:
        return
    
    print("\nAcquiring data from SDR and processing...")
    # Acquire and process data
    features = acquire_and_process_data(sdr, fs, t_seg, n_per_seg)
    
    print(f"Processed feature shape: {features.shape}")
    
    # Make predictions
    predictions = predict_drone_type(model, features)
    
    # Display results
    display_results(predictions)
    
    # Plot PSD of the last sample for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(features[-1])
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Power')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()