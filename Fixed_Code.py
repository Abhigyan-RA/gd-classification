import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import adi
import time
from scipy import signal
from tqdm import tqdm


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


def configure_sdr(center_freq=2.4e9, sample_rate=30.72e6, rx_buffer_size=800000, rx_gain=70):
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
        
        try:
            print("\nCurrent SDR Configuration:")
            print(f"Sample Rate: {sdr.sample_rate}")
            print(f"RF Bandwidth: {sdr.rx_rf_bandwidth}")
            print(f"Center Frequency: {sdr.rx_lo}")
            print(f"Buffer Size: {sdr.rx_buffer_size}")
            print(f"Gain Control Mode: {sdr.gain_control_mode_chan0}")
            print(f"Hardware Gain: {sdr.rx_hardwaregain_chan0}")
        except Exception as e:
            print(f"Error printing configuration: {e}")
        
        print("SDR configured successfully")
        return sdr
    except Exception as e:
        print(f"Error connecting to SDR: {e}")
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
    
    model_path = r'C:\Users\Saransh Duharia\OneDrive\Desktop\DEMON\gd-try-main\gd-try-main\5_6258240131682014782.pkl'
    
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