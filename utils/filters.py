import numpy as np
from scipy.signal import butter, filtfilt

def moving_average_filter(signal, window_size=5):
    """
    이동 평균 필터 (Moving Average)
    """
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='same')
    return smoothed_signal

def butter_lowpass_filter(signal, cutoff_freq, fs, order=4):
    """
    위상 지연이 없는(Zero-phase) 버터워스 로우패스 필터
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal