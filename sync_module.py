import numpy as np
import matplotlib
# Docker 컨테이너 환경을 위해 Agg 백엔드 설정 (GUI가 없는 경우 필수)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 1. [Item 1] 선형 보간 리샘플링 함수
def resample_imu(t_cam, t_imu, y_imu):
    """
    수식: y = y0 + (t_cam - t0) * (y1 - y0) / (t1 - t0)
    """
    idx_t1 = np.searchsorted(t_imu, t_cam, side='right')
    idx_t1 = np.clip(idx_t1, 1, len(t_imu) - 1)
    idx_t0 = idx_t1 - 1
    
    t0, t1, y0, y1 = t_imu[idx_t0], t_imu[idx_t1], y_imu[idx_t0], y_imu[idx_t1]
    return y0 + (t_cam - t0) * ((y1 - y0) / (t1 - t0))

# 2. [Item 2] NCC 계산 및 피크 탐색 함수
def perform_ncc(f, g, dt):
    """
    Zero-mean 정규화 후 상호 상관성 계산
    """
    f_zero = f - np.mean(f)
    g_zero = g - np.mean(g)
    
    # 모든 지연(Lag)에 대해 상관계수 산출
    ncc_curve = np.correlate(f_zero, g_zero, mode='full')
    denom = np.sqrt(np.sum(f_zero**2)) * np.sqrt(np.sum(g_zero**2))
    ncc_curve = ncc_curve / denom if denom != 0 else np.zeros_like(ncc_curve)
    
    # Peak(최대값) 지점 탐색
    max_idx = np.argmax(ncc_curve)
    zero_lag_idx = len(g) - 1
    shift_samples = max_idx - zero_lag_idx
    tau_hat = shift_samples * dt
    
    return tau_hat, ncc_curve, max_idx

# 3. [Main] 데이터 생성 및 통합 시각화
if __name__ == "__main__":
    # 데이터 생성 파라미터
    dt_cam = 0.1  # 10Hz
    dt_imu = 0.01 # 100Hz
    true_offset = 1.3 # 주입할 실제 오차 (1.3초)

    # 신호 생성 (차량 모션을 모사한 사인파 조합)
    t_imu = np.arange(0, 15, dt_imu)
    y_imu = np.sin(0.7 * t_imu) + 0.3 * np.sin(2.1 * t_imu) + np.random.normal(0, 0.1, len(t_imu))
    
    t_cam = np.arange(0, 10, dt_cam)
    f_cam = np.sin(0.7 * (t_cam + true_offset)) + 0.3 * np.sin(2.1 * (t_cam + true_offset))

    # --- 파이프라인 가동 ---
    # Step 1: IMU를 카메라 주기에 맞춰 리샘플링
    y_imu_resampled = resample_imu(t_cam, t_imu, y_imu)
    
    # Step 2: NCC를 통한 오차 추정
    tau_hat, ncc_curve, peak_idx = perform_ncc(f_cam, y_imu_resampled, dt_cam)

    # --- [Item 3] 고도화 시각화 ---
    fig, axes = plt.subplots(3, 1, figsize=(11, 14))
    plt.subplots_adjust(hspace=0.4)
    
    # (1) 시간 오차가 주입된 원본 신호
    axes[0].plot(t_cam, f_cam, label='Camera (Reference)', linewidth=2)
    axes[0].plot(t_cam, y_imu_resampled, label=f'IMU (Delayed, Offset={true_offset}s)', alpha=0.7)
    axes[0].set_title("Step 1: Original Misaligned Signals", fontweight='bold')
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (2) NCC 상관계수 결과 곡선 (Peak 지점 표시) - 핵심 요청 사항!
    lags = (np.arange(len(ncc_curve)) - (len(y_imu_resampled) - 1)) * dt_cam
    axes[1].plot(lags, ncc_curve, color='purple', alpha=0.8, label='NCC Coefficient')
    # Peak 지점에 붉은 점과 텍스트 표시
    axes[1].scatter(tau_hat, ncc_curve[peak_idx], color='red', s=100, edgecolors='black', zorder=5)
    axes[1].annotate(f'Peak Point\n($\\hat{{\\tau}}$ = {tau_hat:.2f}s)', 
                     xy=(tau_hat, ncc_curve[peak_idx]), xytext=(tau_hat + 1, ncc_curve[peak_idx]),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                     fontweight='bold', color='red')
    axes[1].set_title("Step 2: NCC Curve & Peak Point Detection", fontweight='bold')
    axes[1].set_xlabel("Time Lag (seconds)")
    axes[1].set_ylabel("Correlation")
    axes[1].grid(True, alpha=0.3)

    # (3) 보정 후 정렬된 신호
    axes[2].plot(t_cam, f_cam, label='Camera (Reference)', linewidth=2)
    axes[2].plot(t_cam - tau_hat, y_imu_resampled, label=f'IMU Aligned (Est={tau_hat:.2f}s)', 
                 color='green', linestyle='--', alpha=0.8)
    axes[2].set_title("Step 3: Corrected and Aligned Signals", fontweight='bold')
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 결과 저장
    plt.savefig('baseline_final_report.png', dpi=300)
    print(f"✅ 통합 시각화 완료! 추정치: {tau_hat:.2f}s (실제: {true_offset}s)")
    print("결과 파일: 'baseline_final_report.png'")