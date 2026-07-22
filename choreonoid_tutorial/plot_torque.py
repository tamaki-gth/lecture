import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv("inference_log_cmd_4_by_34.csv")

# NaNを削除
df = df.dropna()

# 異常値も削除
df = df[df["base_vx"].abs() < 20]
# goalまで
goal=10.0
df = df[df["time"] <= goal]

# --- 1枚目のグラフ（0〜10秒全体の速度） ---
plt.figure(figsize=(10, 5))


plt.plot(df["time"], df["torque_hip_pitch_right_joint"], label="Hip Pitch", linewidth=1)
plt.plot(df["time"], df["torque_knee_pitch_right_joint"], label="Knee Pitch", linewidth=1)
plt.plot(df["time"], df["torque_ankle_pitch_right_joint"], label="Ankle Pitch", linewidth=1)

# 軸ラベル
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
# タイトル
plt.title("Right Leg Joint Torque")
# 凡例
plt.legend()
# グリッド
plt.grid(True)
# レイアウト調整
plt.tight_layout()
# 保存
plt.savefig("right_leg_joint_torque.png", dpi=300)
print("Saved as right_leg_joint_torque.png")


# ==========================================
# ここから追加：任意の時間（1周期分など）を切り出したグラフ
# ==========================================

# 抽出したい時間の範囲を指定（0.01秒刻みで指定可能）
start_time = 4.83
end_time = 5.49
time2=5.00
time3=5.16
time4=5.33

# 指定した時間範囲でデータをフィルタリング（条件に合う行だけを抽出）
df_cycle = df[(df["time"] >= start_time) & (df["time"] <= end_time)]

# 2枚目のグラフを作成
plt.figure(figsize=(10, 5))

# ※現在は速度をプロットしていますが、以下の列名を変更すれば関節角度やトルクのグラフになります
# 例：plt.plot(df_cycle["time"], df_cycle["torque_hip_pitch_right_joint"], label="Right Hip Torque", color="red")
plt.plot(df_cycle["time"], df_cycle["torque_hip_pitch_right_joint"], label="Hip Pitch", linewidth=1)
plt.plot(df_cycle["time"], df_cycle["torque_knee_pitch_right_joint"], label="Knee Pitch", linewidth=1)
plt.plot(df_cycle["time"], df_cycle["torque_ankle_pitch_right_joint"], label="Ankle Pitch", linewidth=1)

plt.axvline(x=start_time, color='black', linestyle='--', linewidth=1)
plt.axvline(x=time2, color='black', linestyle='--', linewidth=1)
plt.axvline(x=time3, color='black', linestyle='--', linewidth=1)
plt.axvline(x=time4, color='black', linestyle='--', linewidth=1)
plt.axvline(x=end_time, color='black', linestyle='--', linewidth=1)

# 軸ラベル
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")

# タイトル（何秒から何秒を切り出したか自動で表示）
plt.title(f"Right Leg Joint Torque during 1 Stride ({start_time}s - {end_time}s)")

# 凡例
plt.legend()
# グリッド
plt.grid(True)
# レイアウト調整
plt.tight_layout()
# 2枚目の画像を保存
plt.savefig("right_leg_joint_torque_cycle.png", dpi=300)
print("Saved as right_leg_joint_torque_cycle.png")

# 表示（必要に応じてコメントアウトを外す）
# plt.show()