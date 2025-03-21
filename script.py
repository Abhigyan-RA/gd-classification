import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import adi
import time
from scipy import signal
import threading
from tqdm import tqdm
from datetime import datetime
import xgboost as xgb

# Configuration parameters
MODEL_PATH = 'xgboost_drone_detection_model.json'
CENTER_FREQ = 2.4e9  # Center frequency in Hz
SAMPLE_RATE = 40e6   # Sample rate in Hz (40 MHz as used in training)
RX_GAIN = 70         # Hardware gain
BUFFER_SIZE = 800000  # Buffer size
T_SEG = 20           # Time segment in ms
N_PER_SEG = 1024     # Number of samples per segment for PSD
WINDOW_SIZE = 10     # Number of predictions to keep in the sliding window
DELAY_BETWEEN_SAMPLES = 0.1  # Delay between samples in seconds

# Global variables for visualization
last_psd = None
prediction_history = []
plot_lock = threading.Lock()

def extract_dft_features(signal_data, n_fft=2048):
    """
    Extract DFT magnitude spectrum features
    
    Parameters:
        signal_data: Input signal data
        n_fft: Number of points for FFT (default: 2048)
    
    Returns:
        Magnitude spectrum features
    """
    dft = np.fft.fft(signal_data, n=n_fft)
    magnitude_spectrum = np.abs(dft[:n_fft//2])
    
    if np.max(magnitude_spectrum) > 0:
        magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
    
    return magnitude_spectrum

def load_model(model_path):
    """Load the trained model from a JSON file"""
    try:
       
        model = xgb.XGBClassifier()
        
      
        model.load_model(model_path)
        
        print(f"Model loaded from JSON: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from JSON: {e}")
        return None

def configure_sdr(center_freq=CENTER_FREQ, sample_rate=SAMPLE_RATE, 
                  rx_buffer_size=BUFFER_SIZE, rx_gain=RX_GAIN):
    """Configure the PlutoSDR with the specified parameters"""
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

def acquire_and_process_data(sdr, fs=SAMPLE_RATE, t_seg=T_SEG):
    """
    Acquire data from SDR and process it into feature vectors
    
    Returns:
        Feature vector for classification
    """
    global last_psd
    

    n_samples = int(t_seg/1000 * fs)
    
    try:
    
        raw_data = sdr.rx()
        
      
        data_magnitude = np.abs(raw_data)
        
      
        if len(data_magnitude) < n_samples:
            print(f"Warning: Received fewer samples than expected ({len(data_magnitude)} < {n_samples})")
            data_magnitude = np.pad(data_magnitude, (0, n_samples - len(data_magnitude)))
            
        
        features = extract_dft_features(data_magnitude)
        
      
        with plot_lock:
            last_psd = features
            
        return features.reshape(1, -1)  
        
    except Exception as e:
        print(f"Error acquiring or processing data: {e}")
        return None

def predict_drone_presence(model, features):
    """
    Predict drone presence using the trained model
    
    Returns:
        True if drone detected, False otherwise
    """
    try:
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1] 
        return prediction[0], probability
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

def update_visualization():
    """Update the visualization in a separate thread"""
    plt.ion()  
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    
    psd_line, = ax1.plot([], [], 'b-')
    ax1.set_title('Power Spectral Density / DFT Magnitude')
    ax1.set_xlabel('Frequency Bin')
    ax1.set_ylabel('Normalized Magnitude')
    ax1.grid(True)
    
    prediction_line, = ax2.plot([], [], 'r-')
    threshold_line, = ax2.plot([], [], 'g--')
    ax2.set_title('Drone Detection Probability Over Time')
    ax2.set_xlabel('Sample Number')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    
    plt.tight_layout()
    
    x_psd = np.arange(1024)  
    
    try:
        while True:
            with plot_lock:
                current_psd = last_psd
                history = prediction_history.copy()
            
            if current_psd is not None:
               
                psd_line.set_data(x_psd, current_psd)
                ax1.relim()
                ax1.autoscale_view()
                
          
                x_pred = np.arange(len(history))
                if len(history) > 0:
                    prediction_line.set_data(x_pred, history)
                    threshold_line.set_data([0, len(history)-1], [0.5, 0.5])
                    ax2.set_xlim(0, max(10, len(history)))
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            time.sleep(0.5)
    except Exception as e:
        print(f"Visualization error: {e}")

def main():

    model = load_model(MODEL_PATH)
    if model is None:
        return
    

    sdr = configure_sdr()
    if sdr is None:
        return
    

    drone_count = 0
    no_drone_count = 0
    total_predictions = 0
    
    
    vis_thread = threading.Thread(target=update_visualization)
    vis_thread.daemon = True
    vis_thread.start()
    
    print("\nStarting continuous drone detection. Press Ctrl+C to stop.")
    
    try:
        while True:
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
          
            features = acquire_and_process_data(sdr)
            
            if features is not None:
               
                prediction, probability = predict_drone_presence(model, features)
                
           
                with plot_lock:
                    prediction_history.append(probability)
                    # Keep only the last WINDOW_SIZE predictions
                    if len(prediction_history) > WINDOW_SIZE:
                        prediction_history.pop(0)
                
               
                if len(prediction_history) > 0:
                    avg_probability = sum(prediction_history) / len(prediction_history)
                    majority_vote = avg_probability > 0.5
                    
                   
                    total_predictions += 1
                    if prediction == 1:
                        drone_count += 1
                    else:
                        no_drone_count += 1
                    
                    
                    print(f"[{timestamp}] Current: {'DRONE' if prediction == 1 else 'NO DRONE'} "
                          f"(Prob: {probability:.4f}), "
                          f"Sliding Window: {'DRONE' if majority_vote else 'NO DRONE'} "
                          f"(Avg Prob: {avg_probability:.4f})")
                    
                    
                    if total_predictions % 10 == 0:
                        drone_percentage = (drone_count / total_predictions) * 100
                        print(f"\nOverall Statistics after {total_predictions} predictions:")
                        print(f"Drone detections: {drone_count} ({drone_percentage:.2f}%)")
                        print(f"No drone detections: {no_drone_count} ({100-drone_percentage:.2f}%)\n")
            
            
            time.sleep(DELAY_BETWEEN_SAMPLES)
            
    except KeyboardInterrupt:
        print("\nDrone detection stopped by user.")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
       
        if sdr is not None:
            print("Closing SDR connection...")
        

if __name__ == "__main__":
    main()