import numpy as np
from scipy.io import loadmat
import soundfile as sf

# matファイルの読み込み
data_hz = loadmat("data/small_pinna_final.mat")

# Rustで読み込みやすいように、各角度のインパルス応答をWAVとして保存
# ※フォーマットは32bit Floatにして精度の劣化を防ぎます
sf.write("hrtf_left_9.wav", data_hz["left"][:, 0], 44100, subtype="FLOAT")
sf.write("hrtf_right_9.wav", data_hz["right"][:, 0], 44100, subtype="FLOAT")
sf.write("hrtf_left_63.wav", data_hz["left"][:, 63], 44100, subtype="FLOAT")
sf.write("hrtf_right_9.wav", data_hz["right"][:, 9], 44100, subtype="FLOAT")
sf.write("hrtf_left_40.wav", data_hz["left"][:, 40], 44100, subtype="FLOAT")
sf.write("hrtf_right_32.wav", data_hz["right"][:, 32], 44100, subtype="FLOAT")

print("✅ HRTFデータのWAV書き出しが完了しました！")
