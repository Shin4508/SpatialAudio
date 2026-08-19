import numpy as np
from scipy.io import loadmat
import soundfile as sf

# matファイルの読み込み
data_hz = loadmat("small_pinna_final.mat")

# Rustで読み込みやすいように、各角度のインパルス応答をWAVとして保存
# ※フォーマットは32bit Floatにして精度の劣化を防ぎます
sf.write("hrtf_0.wav", data_hz["left"][:, 0], 44100, subtype="FLOAT")


print("✅ HRTFデータのWAV書き出しが完了しました！")
