import cv2
import numpy as np
import os
import time

# === 체스보드 설정 ===
chessboard_size = (8, 6)  # 내부 코너 수
square_size = 25  # mm 단위

# === 3D 객체 포인트 ===
obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
obj_points *= square_size

# === 포인트 리스트 ===
obj_points_list = []
img_points_list = []

# === 디버깅 폴더 생성 ===
os.makedirs("debug_frames", exist_ok=True)

# === 비디오 로드 ===
cap = cv2.VideoCapture("data/chessboard.mp4")

frame_id = 0
last_gray = None

print("▶️ 체스보드 코너 감지 시작...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("📹 영상이 끝났습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, chessboard_size, flags)

    if found:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        cv2.drawChessboardCorners(frame, chessboard_size, corners2, found)
        obj_points_list.append(obj_points)
        img_points_list.append(corners2)
        last_gray = gray.copy()
    else:
        print(f"[WARN] Frame {frame_id}: 코너 감지 실패")
        cv2.imwrite(f"debug_frames/fail_frame_{frame_id}.jpg", frame)

    cv2.imshow("Chessboard Detection", frame)
    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 사용자에 의해 중단되었습니다.")
        break

cap.release()
cv2.destroyAllWindows()

# === 보정 수행 ===

print("\n==========================")
print(f"디버깅: last_gray is None? {last_gray is None}")
print(f"디버깅: 감지된 체스보드 수: {len(obj_points_list)}")

if last_gray is None:
    print("❌ 보정 실패: 코너를 단 한 번도 감지하지 못했습니다.")
elif len(obj_points_list) < 5:
    print(f"❌ 보정 실패: 감지된 체스보드 수가 부족합니다. ({len(obj_points_list)}개, 최소 5개 필요)")
else:
    try:
        print("📌 calibrateCamera() (속도 최적화 버전)")
        print("⏳ 빠르게 보정 중...")

        # 샘플 수 제한
        max_samples = 50
        obj_sample = obj_points_list[:max_samples]
        img_sample = img_points_list[:max_samples]

        # 반복 조건 최적화
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-5)

        start_time = time.time()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_sample, img_sample, last_gray.shape[::-1], None, None, criteria=criteria
        )
        elapsed_time = time.time() - start_time
        print(f"✅ 보정 완료! ⏱️ 걸린 시간: {elapsed_time:.2f}초")

        # RMSE 계산
        total_error = 0
        for i in range(len(obj_sample)):
            img_points2, _ = cv2.projectPoints(obj_sample[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(img_sample[i], img_points2, cv2.NORM_L2) / len(img_points2)
            total_error += error
        mean_error = total_error / len(obj_sample)

        # 결과 출력
        print("\n🎯 카메라 캘리브레이션 결과:")
        print(f"fx: {mtx[0,0]:.4f}, fy: {mtx[1,1]:.4f}")
        print(f"cx: {mtx[0,2]:.4f}, cy: {mtx[1,2]:.4f}")
        print(f"왜곡 계수: {dist.ravel()}")
        print(f"📈 평균 RMSE: {mean_error:.6f}")

    except Exception as e:
        print("❌ 예외 발생:", str(e))

print("==========================")
print("🔚 프로그램 종료")
