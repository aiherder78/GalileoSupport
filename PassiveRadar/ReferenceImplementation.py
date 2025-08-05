#https://grok.com/share/bGVnYWN5_bc31de05-bb45-4431-bb0b-e87df86d8e48 (LOL about the plasma balls - we can take that stuff out if desired)
#https://grok.com/chat/fecf3784-2e79-4d1e-ae33-018098944629 Code rundown.
#https://sci-hub.ru/10.1049/ip-rsn:19990322 (the paper Mitch Randall referred to in one of the interviews - I'm not in any way a radar expert,
#   and I have not physically implemented yet - I will have to buy another SDR since mine is single channel before I can get one running).
#https://youtu.be/fLjDDS6m9Cs (interview of Mitch Randall about his proposed system)
#https://hackaday.com/2015/06/05/building-your-own-sdr-based-passive-radar-on-a-shoestring/
#https://www.nicap.org/CATEGORIES/09-RADAR_Cases/MUFONPresentation.pdf Paper by Peter Davenport talking about passive radar - Mitch Randall
#   mentioned him as his inspiration.  The paper is great for explaining basic terms. Peter Davenport 
#   runs NUFORC (National UFO Reporting Center) now.

#Remember to install the dependencies:
#sudo apt-get install gnuradio
#pip install numpy scipy matplotlib
#Ensure the SDR drivers are installed:  (e.g., rtl-sdr package)
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import osmosdr  # GNU Radio SDR interface
import time
import matplotlib.pyplot as plt

# Constants
SAMPLE_RATE = 2.4e6  # 2.4 MHz for RTL-SDR
CENTER_FREQ = 107e6  # FM tower at 107 MHz
CAPTURE_DURATION = 5  # 5 seconds
C = 3e8  # Speed of light (m/s)
N_SAMPLES = int(SAMPLE_RATE * CAPTURE_DURATION)  # Total samples

def capture_signals():
    """
    Capture reference (dipole, FM tower) and surveillance (Yagi, air traffic) signals
    using two RTL-SDRs at 107 MHz.
    Returns: ref_signal, surv_signal (complex IQ arrays)
    """
    # Configure SDRs (assumes two RTL-SDRs, USB indices 0 and 1)
    sdr_ref = osmosdr.source(device_name="rtl=0")
    sdr_surv = osmosdr.source(device_name="rtl=1")
    
    for sdr in [sdr_ref, sdr_surv]:
        sdr.set_sample_rate(SAMPLE_RATE)
        sdr.set_center_freq(CENTER_FREQ)
        sdr.set_gain(40)  # Typical gain for RTL-SDR
        sdr.set_bandwidth(2.4e6)  # Match sample rate
    
    # Capture 5 seconds of IQ data
    ref_signal = np.zeros(N_SAMPLES, dtype=np.complex64)
    surv_signal = np.zeros(N_SAMPLES, dtype=np.complex64)
    
    sdr_ref.start()
    sdr_surv.start()
    time.sleep(0.1)  # Allow SDRs to stabilize
    
    for i in range(0, N_SAMPLES, 1024):
        ref_signal[i:i+1024] = sdr_ref.read_samples(1024)
        surv_signal[i:i+1024] = sdr_surv.read_samples(1024)
    
    sdr_ref.stop()
    sdr_surv.stop()
    return ref_signal, surv_signal

def preprocess_signals(ref_signal, surv_signal):
    """
    Filter, normalize, and synchronize signals.
    Args: ref_signal, surv_signal (complex IQ arrays)
    Returns: ref_signal_filt, surv_signal_filt (filtered, normalized arrays)
    """
    # Bandpass filter at 107 MHz ± 1 MHz
    nyquist = SAMPLE_RATE / 2
    low = (CENTER_FREQ - 1e6) / nyquist
    high = (CENTER_FREQ + 1e6) / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    ref_signal_filt = signal.filtfilt(b, a, ref_signal)
    surv_signal_filt = signal.filtfilt(b, a, surv_signal)
    
    # Normalize amplitudes (reference signal typically stronger)
    ref_signal_filt /= np.max(np.abs(ref_signal_filt))
    surv_signal_filt /= np.max(np.abs(surv_signal_filt))
    
    # Time synchronization (assume small cable delay, align via cross-correlation)
    corr = signal.correlate(ref_signal_filt, surv_signal_filt, mode='full')
    lag = np.argmax(np.abs(corr)) - len(ref_signal_filt) + 1
    if lag > 0:
        surv_signal_filt = np.roll(surv_signal_filt, -lag)
    elif lag < 0:
        ref_signal_filt = np.roll(ref_signal_filt, lag)
    
    return ref_signal_filt, surv_signal_filt

def cross_correlate_signals(ref_signal, surv_signal):
    """
    Cross-correlate to detect reflection delays.
    Args: ref_signal, surv_signal (filtered, normalized arrays)
    Returns: delays, correlation (time delays and correlation values)
    """
    corr = signal.correlate(surv_signal, ref_signal, mode='full')
    delays = np.arange(-len(ref_signal) + 1, len(ref_signal)) / SAMPLE_RATE
    correlation = np.abs(corr)
    
    # Normalize correlation
    correlation /= np.max(correlation)
    return delays, correlation

def subtract_direct_signal(ref_signal, surv_signal):
    """
    Subtract direct signal leakage from surveillance channel using LMS filter.
    Args: ref_signal, surv_signal (filtered, normalized arrays)
    Returns: surv_signal_clean (reflection-only signal)
    """
    # Simple LMS filter implementation
    mu = 0.01  # Step size
    n_taps = 32  # Filter length
    weights = np.zeros(n_taps, dtype=np.complex64)
    surv_signal_clean = np.zeros_like(surv_signal, dtype=np.complex64)
    
    for i in range(n_taps, len(surv_signal)):
        x = ref_signal[i-n_taps:i][::-1]  # Input vector
        y = surv_signal[i]  # Desired output
        y_hat = np.dot(weights, x)  # Filter output
        error = y - y_hat
        weights += mu * error * x.conj()  # Update weights
        surv_signal_clean[i] = error  # Output is error (reflections)
    
    return surv_signal_clean

def extract_target_properties(delays, correlation, surv_signal_clean):
    """
    Extract range, Doppler shift, and electron density (for plasmas).
    Args: delays, correlation (from cross-correlation), surv_signal_clean
    Returns: range (m), velocity (m/s), electron_density (cm^-3)
    """
    # Find reflection peak
    peak_idx = np.argmax(correlation)
    delay = delays[peak_idx]
    
    # Calculate bistatic range (R_t + R_r - L)
    L = 10000  # Baseline distance (m, transmitter to receiver, assumed)
    bistatic_range = delay * C + L  # R_t + R_r
    
    # Doppler shift (velocity)
    N = len(surv_signal_clean)
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    spectrum = np.abs(fft(surv_signal_clean))
    doppler_idx = np.argmax(spectrum[:N//2])  # Positive frequencies
    doppler_freq = freqs[doppler_idx]
    velocity = doppler_freq * (C / CENTER_FREQ) / 2  # Bistatic Doppler approximation
    
    # Electron density (for plasmas, assuming 107 MHz is below f_p)
    if correlation[peak_idx] > 0.1:  # Threshold for detection
        f_p = CENTER_FREQ  # Assume reflection at 107 MHz implies f_p >= 107 MHz
        electron_density = (f_p / 9) ** 2  # n_e in cm^-3
    else:
        electron_density = None  # No plasma detected
    
    return bistatic_range, velocity, electron_density

def main():
    """
    Main function to run bistatic radar processing.
    """
    # Step 1: Capture signals
    print("Capturing signals...")
    ref_signal, surv_signal = capture_signals()
    
    # Step 2: Pre-process signals
    print("Preprocessing signals...")
    ref_signal_filt, surv_signal_filt = preprocess_signals(ref_signal, surv_signal)
    
    # Step 3: Cross-correlate signals
    print("Correlating signals...")
    delays, correlation = cross_correlate_signals(ref_signal_filt, surv_signal_filt)
    
    # Step 4: Subtract direct signal
    print("Subtracting direct signal...")
    surv_signal_clean = subtract_direct_signal(ref_signal_filt, surv_signal_filt)
    
    # Step 5: Extract target properties
    print("Extracting target properties...")
    bistatic_range, velocity, electron_density = extract_target_properties(
        delays, correlation, surv_signal_clean
    )
    
    # Print results
    print(f"Bistatic Range (R_t + R_r): {bistatic_range:.2f} m")
    print(f"Velocity: {velocity:.2f} m/s")
    if electron_density:
        print(f"Electron Density (if plasma): {electron_density:.2e} cm^-3")
    else:
        print("No plasma detected (reflection may be from plane).")
    
    # Plot correlation
    plt.plot(delays * 1e6, correlation)
    plt.xlabel("Time Delay (µs)")
    plt.ylabel("Correlation Magnitude")
    plt.title("Cross-Correlation of Reference and Surveillance Signals")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
