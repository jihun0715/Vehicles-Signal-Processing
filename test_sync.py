"""
독립 파이프라인 테스트 스크립트 (위상 정정 및 도커 저장 최적화 버전)
극한 환경 모사 데이터 생성 -> 필터링 -> NCC 정합 검증
"""
import numpy as np
import matplotlib.pyplot as plt

# 구축한 utils 모듈 임포트
from utils.filters import butter_lowpass_filter
from utils.sync_engine import resample_signal, perform_ncc

def inject_noise_and_spikes(signal, noise_std, num_spikes, spike_amp):
    """가상 노이즈 및 스파이크 주입 헬퍼 함수"""
    noisy = signal + np.random.normal(0, noise_std, len(signal))
    if num_spikes > 0:
        spike_idx = np.random.choice(len(signal), size=num_spikes, replace=False)
        noisy[spike_idx] += np.random.choice([-spike_amp, spike_amp], size=num_spikes)
    return noisy

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. 가상 데이터 생성 (동적 모션: 급가속 처프 신호)
    # ---------------------------------------------------------
    dt_imu = 0.01  # 100Hz
    t_imu = np.arange(0, 15, dt_imu)
    
    # 시간에 따라 진폭과 주파수가 변하는 역동적 파형
    amplitude = 1.0 + 0.5 * np.sin(0.5 * t_imu)
    phase = 1.5 * t_imu + 0.2 * t_imu**2
    clean_motion = amplitude * np.sin(phase)
    
    # 1.1 카메라 신호 (10Hz, 기준 신호 f(t), 약한 노이즈)
    dt_cam = 0.1
    t_cam = np.arange(0, 15, dt_cam)
    f_clean = np.interp(t_cam, t_imu, clean_motion)
    f_cam = inject_noise_and_spikes(f_clean, noise_std=0.05, num_spikes=2, spike_amp=1.5)
    
    # 1.2 IMU 신호 (100Hz, 지연 주입 g(t), 강한 노이즈 및 스파이크)
    true_offset = 1.7  # 1.7초 지연 주입 (물리적인 시간 평행 이동 정정)
    t_imu_delayed = t_imu - true_offset
    g_clean_delayed = np.interp(t_imu_delayed, t_imu, clean_motion, left=0, right=0)
    g_imu = inject_noise_and_spikes(g_clean_delayed, noise_std=0.2, num_spikes=6, spike_amp=3.0)
    
    # ---------------------------------------------------------
    # 2. 전처리 (Filtering & Resampling)
    # ---------------------------------------------------------
    # IMU 신호 필터링 (스파이크/노이즈 제거, 컷오프 3Hz)
    g_imu_filtered = butter_lowpass_filter(g_imu, cutoff_freq=3.0, fs=100.0, order=4)
    
    # IMU 신호를 카메라 타임스탬프(10Hz)에 맞춰 리샘플링
    g_resampled = resample_signal(t_cam, t_imu, g_imu_filtered)
    
    # ---------------------------------------------------------
    # 3. NCC 동기화 알고리즘 구동
    # ---------------------------------------------------------
    tau_hat, ncc_curve, peak_idx = perform_ncc(f_cam, g_resampled, dt_cam)
    
    # 수식 쉬프트 방향에 따른 부호 반전 정렬
    tau_hat = -tau_hat
    
    print("========================================")
    print(f"✅ 주입된 실제 오차 (Ground Truth): {true_offset:.2f} s")
    print(f"🎯 추정된 타임 오프셋 (Estimation): {tau_hat:.2f} s")
    print(f"📉 오차(Error): {abs(true_offset - tau_hat):.4f} s")
    print("========================================")
    
    # ---------------------------------------------------------
    # 4. 결과 시각화 및 파일 저장 (도커 환경 최적화)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.4)
    
    # [1] 원본 비동기 신호 (노이즈 포함)
    axes[0].plot(t_cam, f_cam, label='Camera (10Hz, Ref)', linewidth=2)
    axes[0].plot(t_imu, g_imu, label=f'IMU (100Hz, True Offset={true_offset}s)', alpha=0.5)
    axes[0].set_title("Step 1: Raw Misaligned Signals with Noise & Spikes", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # [2] 필터링 및 리샘플링 완료된 신호 (정렬 전 상태 유지)
    axes[1].plot(t_cam, f_cam, label='Camera (Ref)', linewidth=2)
    axes[1].plot(t_cam, g_resampled, label='IMU (Filtered & Resampled to 10Hz)', color='orange')
    axes[1].set_title("Step 2: After LPF and Resampling (Before Alignment)", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # [3] 타임 오프셋 보정 후 정렬된 신호 (X축 보정 적용)
    # IMU 데이터 자체의 시간축을 tau_hat만큼 쉬프트시켜 정확한 겹침 시각화
    axes[2].plot(t_cam, f_cam, label='Camera (Ref)', linewidth=2)
    axes[2].plot(t_cam - tau_hat, g_resampled, label=f'IMU Aligned (Est={tau_hat:.2f}s)', color='green', linestyle='--')
    axes[2].set_title("Step 3: Corrected & Perfectly Aligned Signals", fontweight='bold')
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 도커 환경 에러 방지를 위해 plt.show() 대신 이미지 파일 저장 사용
    plt.savefig('sync_test_result.png', dpi=150)
    print("-> 📊 [성공] 위상 정정이 완료된 결과 차트가 'sync_test_result.png'로 저장되었습니다.")