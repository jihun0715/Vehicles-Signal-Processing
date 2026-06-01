import numpy as np

def resample_signal(t_target, t_source, y_source):
    """
    선형 보간법을 이용한 시계열 리샘플링 (np.interp 활용 최적화)
    
    Parameters:
        t_target: 맞추고자 하는 타겟 타임스탬프 배열 (예: Camera 주기)
        t_source: 원본 신호의 타임스탬프 배열 (예: IMU 주기)
        y_source: 원본 신호 데이터
    """
    return np.interp(t_target, t_source, y_source)

def perform_ncc(f, g, dt):
    """
    Normalized Cross-Correlation (NCC) 기반 타임 오프셋 추정
    
    Parameters:
        f: 기준 신호 (예: Camera)
        g: 비교 대상 신호 (예: IMU)
        dt: 두 신호의 공통 샘플링 주기
        
    Returns:
        tau_hat: 추정된 시간 오차 (초)
        ncc_curve: 상호 상관계수 배열
        max_idx: Peak 지점 인덱스
    """
    # Zero-mean 정규화
    f_zero = f - np.mean(f)
    g_zero = g - np.mean(g)
    
    # 상호 상관성 연산
    ncc_curve = np.correlate(f_zero, g_zero, mode='full')
    
    # 에너지 정규화 (분모)
    denom = np.sqrt(np.sum(f_zero**2)) * np.sqrt(np.sum(g_zero**2))
    if denom != 0:
        ncc_curve = ncc_curve / denom
    else:
        ncc_curve = np.zeros_like(ncc_curve)
        
    # Peak 탐색
    max_idx = np.argmax(ncc_curve)
    zero_lag_idx = len(g) - 1
    shift_samples = max_idx - zero_lag_idx
    tau_hat = shift_samples * dt
    
    return tau_hat, ncc_curve, max_idx