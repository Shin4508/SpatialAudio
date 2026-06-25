# generate_hrtf.py
import os
import scipy.io
import numpy as np


mat_file_path = os.path.join("..", "data", "small_pinna_final.mat")
# 1. 本物の.matファイルをロード (ファイル名はお好みで変更してください)
mat_data = scipy.io.loadmat(mat_file_path)

# 2. 配列の抽出 (Matlabは縦×横なので、200行×72列になっている)
# C言語で扱いやすくするために、転置(.T)して [72方向][200タップ] の形に変換します
hrir_left = mat_data["left"].T.astype(np.float32)  # 形式: [72][200]
hrir_right = mat_data["right"].T.astype(np.float32)  # 形式: [72][200]

# 方向数とタップ数を自動取得 (72 と 200 になる)
NUM_DIRECTIONS = hrir_left.shape[0]
NUM_TAPS = hrir_left.shape[1]

# 3. C言語のヘッダファイルとして出力
header_path = "src/hrtf_data.h"

with open(header_path, "w") as f:
    f.write("/* Automatically generated from KEMAR HRIR Dataset */\n")
    f.write("#ifndef HRTF_DATA_H\n")
    f.write("#define HRTF_DATA_H\n\n")

    f.write(f"#define NUM_DIRECTIONS {NUM_DIRECTIONS}\n")
    f.write(f"#define NUM_TAPS {NUM_TAPS}\n\n")

    # 左耳のデータを書き出し
    f.write("const float hrtf_left[NUM_DIRECTIONS][NUM_TAPS] = {\n")
    for row in hrir_left:
        f.write("    {" + ", ".join([f"{v:.6f}f" for v in row]) + "},\n")
    f.write("};\n\n")

    # 右耳のデータを書き出し
    f.write("const float hrtf_right[NUM_DIRECTIONS][NUM_TAPS] = {\n")
    for row in hrir_right:
        f.write("    {" + ", ".join([f"{v:.6f}f" for v in row]) + "},\n")
    f.write("};\n\n")

    f.write("#endif /* HRTF_DATA_H */\n")

print(
    f"✨ 本物のHRTFデータ（{NUM_DIRECTIONS}方向 × {NUM_TAPS}タップ）を {header_path} にエクスポートしました！"
)
