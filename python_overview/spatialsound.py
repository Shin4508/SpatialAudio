import numpy as np
from scipy import signal
from scipy.signal import fftconvolve, convolve, butter, filtfilt
from scipy.io import loadmat
import soundfile as sf
from IPython.display import Audio
from matplotlib import pyplot as plt

sample, sr = sf.read("sample.mp3")
print(sr)
if sample.ndim > 1:
    sample_1 = sample[:, 0]

print(sample.shape)

def midSide(sample):
  left = sample[:, 0]
  right = sample[:, 1]
  mid = (left + right) / 2
  side = (left - right) / 2
  return mid, side

def crossover_split(sound, sr, cutoff_freq=200):
    """
    指定した周波数(cutoff_freq)で、音を「低音成分」と「高音成分」に分割する関数
    """
    nyq = 0.5 * sr
    normal_cutoff = cutoff_freq / nyq

    # 4次のバターワースフィルター（自然で綺麗な切れ味のフィルター）を作成
    b_low, a_low = butter(4, normal_cutoff, btype='low', analog=False)
    b_high, a_high = butter(4, normal_cutoff, btype='high', analog=False)

    # filtfiltを使って位相をズラさずに分割（これがプロのテクニック！）
    low_band = filtfilt(b_low, a_low, sound)
    high_band = filtfilt(b_high, a_high, sound)

    return low_band * 4.0, high_band

def audio_position(arr1, arr2, data1, data2, sound):

  dict1 = {}
  dict2 = {}

  for i in range(len(arr1)+len(arr2)):

    if i < len(arr1):
      dict1[i] = {"left"+str(i): fftconvolve(sound, data1["left"][:, arr1[i]], mode="full")}
      dict2[i] = {"right"+str(i): fftconvolve(sound, data1["right"][:, arr1[i]], mode="full")}

    else:
      dict1[i] = {"left"+str(i): fftconvolve(sound, data2["left"][:, arr2[i-len(arr1)]], mode="full")}
      dict2[i] = {"right"+str(i): fftconvolve(sound, data2["right"][:, arr2[i-len(arr1)]], mode="full")}

  return dict1, dict2

def get_low_shelf_coeffs(sr, f0, gain_db, q=0.707):
    """
    Audio EQ Cookbookの式に基づいたLow-Shelfフィルター係数の計算
    f0: カットオフ周波数, gain_db: 増幅量, q: 鋭さ
    """
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/q - 1) + 2)
    cos_w0 = np.cos(w0)
    sqrt_A_alpha_2 = 2 * np.sqrt(A) * alpha

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + sqrt_A_alpha_2)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - sqrt_A_alpha_2)
    a0 = (A + 1) + (A - 1) * cos_w0 + sqrt_A_alpha_2
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - sqrt_A_alpha_2

    return [b0, b1, b2], [a0, a1, a2]
def apply_active_eq(sound, sr, target_freq=120):
  # 1. 音量解析 (RMS)
  rms = np.sqrt(np.mean(sound**2))

  # 2. ブースト量の決定
  # 音が小さい(0.01以下)時に最大+6dB、大きい時に0dBへ
  threshold = 0.05
  gain_db = 6.0 * max(0, 1.0 - (rms / threshold)) if rms < threshold else 0.0

  # 3. Low-Shelf 係数の取得
  b, a = get_low_shelf_coeffs(sr, target_freq, gain_db)

  # 4. フィルタリング実行
  return signal.lfilter(b, a, sound)

mid, side = midSide(sample)

side = side * 1.5

left = mid + side
right = mid - side

mid_side = np.vstack([left, right]).T
# 保存
sf.write("mid_side.mp3", mid_side, sr)

data_horizontal = loadmat("small_pinna_final.mat")
data_frontal = loadmat("small_pinna_frontal.mat")

mid = apply_active_eq(mid, sr)
side = apply_active_eq(side, sr)

mid_low, mid_high = crossover_split(mid, sr, cutoff_freq=120)

mid_left, mid_right = audio_position([9, 63], [48], data_horizontal, data_frontal, mid_high)
side_left, side_right = audio_position([36], [48], data_horizontal, data_frontal, side)

# 1. ゼロ埋め（長さ揃え）するための関数をここで定義しておく
def pad(data, length):
    return np.pad(data, (0, length - len(data)), 'constant')

all_lengths = [len(mid_low)]
for i in range(len(mid_left)):
    all_lengths.append(len(mid_left[i]["left"+str(i)]))
for j in range(len(side_left)):
    all_lengths.append(len(side_left[j]["left"+str(j)]))

max_len_hrtf = max(all_lengths)

# 3. 空っぽ（ゼロ）の配列を用意する
left_sum = np.zeros(max_len_hrtf)
right_sum = np.zeros(max_len_hrtf)

# 4. すべて max_len_hrtf に長さを揃えながら足していく！
left_sum += pad(mid_low, max_len_hrtf)
right_sum += pad(mid_low, max_len_hrtf)

for i in range(len(mid_left)):
  left_sum += pad(mid_left[i]["left"+str(i)], max_len_hrtf)
  right_sum += pad(mid_right[i]["right"+str(i)], max_len_hrtf)

for j in range(len(side_left)):
  left_sum += pad(side_left[j]["left"+str(j)], max_len_hrtf)
  # 【重要】右耳のSideは「引く」！
  right_sum -= pad(side_right[j]["right"+str(j)], max_len_hrtf)

max_val = np.max(np.abs([left_sum, right_sum]))
if max_val > 0:
  left_sum = left_sum / max_val
  right_sum = right_sum / max_val

stereo = np.vstack([left_sum, right_sum]).T
# 保存
sf.write("hrtf.mp3", stereo, sr)

def add_reflection(source, data_h, sr, reflect_idx, delay_ms=20, gain=0.4, cutoff_hz=3000):
  delay_samples = int(sr * (delay_ms / 1000))
    # 遅延させた音源
  reflected_source = np.zeros(len(source) + delay_samples)
  reflected_source[delay_samples:] = source * gain

  nyq = 0.5 * sr
  normal_cutoff = cutoff_hz / nyq
  b, a = signal.butter(1, normal_cutoff, btype='low', analog=False)
  reflected_source = signal.lfilter(b, a, reflected_source)

    # 反射音のHRTF適用
  reflect_l = fftconvolve(reflected_source, data_h["left"][:, reflect_idx], mode="full")
  reflect_r = fftconvolve(reflected_source, data_h["right"][:, reflect_idx], mode="full")
  return reflect_l, reflect_r

# 反射音を左右別々に作る
ref_r_l, ref_r_r = add_reflection(side, data_horizontal, sr, reflect_idx=27, delay_ms=18)
ref_l_l, ref_l_r = add_reflection(side, data_horizontal, sr, reflect_idx=45, delay_ms=23)
# 全ての長さを揃えて足す
max_len = max(len(left_sum), len(ref_r_l), len(ref_l_l))

def pad(data, length):
    return np.pad(data, (0, length - len(data)), 'constant')

final_l = pad(left_sum, max_len) + pad(ref_r_l, max_len) + pad(ref_l_l, max_len)
final_r = pad(right_sum, max_len) + pad(ref_r_r, max_len) + pad(ref_l_r, max_len)


audio_final = np.vstack([final_l, final_r]).T

#処理後の音源に倍率をかけて全体をガッツリ持ち上げる
audio_final = audio_final * 1.5

audio_final = np.tanh(audio_final)

# 保存
sf.write("final.mp3", audio_final, sr)

plt.plot(sample)
plt.show()

plt.plot(audio_final)
plt.show()

Audio('sample.mp3')

Audio('mid_side.mp3')

Audio('hrtf.mp3')

Audio('final.mp3')
