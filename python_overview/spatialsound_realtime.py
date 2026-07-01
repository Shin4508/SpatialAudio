import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from scipy.io import loadmat
import soundfile as sf

# Basically EQ for low Hz and high Hz


def get_low_shelf_coeffs(sr, f0, gain_db, q=0.707):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / q - 1) + 2)
    cos_w0 = np.cos(w0)
    sqrt_A_alpha_2 = 2 * np.sqrt(A) * alpha

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + sqrt_A_alpha_2)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - sqrt_A_alpha_2)
    a0 = (A + 1) + (A - 1) * cos_w0 + sqrt_A_alpha_2
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - sqrt_A_alpha_2

    return [b0, b1, b2], [a0, a1, a2]


def get_high_shelf_coeffs(sr, f0, gain_db, q=0.707):
    """I heared human brains are sensitive to higher sound so that the calculated sound cannot fool human brain, so I needed to cutoff"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / q - 1) + 2)
    cos_w0 = np.cos(w0)
    sqrt_A_alpha_2 = 2 * np.sqrt(A) * alpha

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + sqrt_A_alpha_2)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - sqrt_A_alpha_2)
    a0 = (A + 1) - (A - 1) * cos_w0 + sqrt_A_alpha_2
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - sqrt_A_alpha_2

    return [b0, b1, b2], [a0, a1, a2]


# For realtime computing


class RealTimeBiquadFilter:
    """IR class for Realtime"""

    def __init__(self, b, a):
        self.b = b
        self.a = a
        self.zi = signal.lfilter_zi(b, a)  # combine b and a

    def process(self, block):
        output, self.zi = signal.lfilter(self.b, self.a, block, zi=self.zi)
        return output


class RealTimeCrossover:
    """when I use bose earphone I felt lower sound is the same as normal 2channel AudioSegment
    so I splitted to lower sound and higher sound not to move the lower sound."""

    def __init__(self, sr, cutoff_freq=120):
        nyq = 0.5 * sr
        normal_cutoff = cutoff_freq / nyq
        self.b_low, self.a_low = signal.butter(4, normal_cutoff, btype="low")
        self.b_high, self.a_high = signal.butter(4, normal_cutoff, btype="high")
        self.zi_low = signal.lfilter_zi(self.b_low, self.a_low)
        self.zi_high = signal.lfilter_zi(self.b_high, self.a_high)

    def process(self, block):
        low_band, self.zi_low = signal.lfilter(
            self.b_low, self.a_low, block, zi=self.zi_low
        )
        high_band, self.zi_high = signal.lfilter(
            self.b_high, self.a_high, block, zi=self.zi_high
        )
        return low_band * 1.5, high_band


class RealTimeActiveEQ:
    """the function that is on the top is for this class"""

    def __init__(self, sr, target_freq=120):
        self.sr = sr
        self.target_freq = target_freq
        self.zi = np.zeros(2)

    def process(self, block):
        rms = np.sqrt(np.mean(block**2))
        threshold = 0.05
        gain_db = 6.0 * max(0, 1.0 - (rms / threshold)) if rms < threshold else 0.0

        b, a = get_low_shelf_coeffs(self.sr, self.target_freq, gain_db)
        output_block, self.zi = signal.lfilter(b, a, block, zi=self.zi)
        return output_block


class RealTimeOverlapSaveOverlap:
    def __init__(self, hrtf_impulse, block_size=128):
        self.L = block_size
        self.N = len(hrtf_impulse)
        self.M = self.L + self.N - 1
        self.fft_size = int(2 ** np.ceil(np.log2(self.M)))
        self.H = fft(hrtf_impulse, n=self.fft_size)
        self.input_buffer = np.zeros(self.fft_size)

    def process(self, block):
        self.input_buffer[: -(self.L)] = self.input_buffer[self.L :]
        self.input_buffer[-(self.L) :] = block

        X = fft(self.input_buffer, n=self.fft_size)
        Y = X * self.H
        y = np.real(ifft(Y))
        return y[-(self.L) :]


class RealTimeDelayLine:
    def __init__(self, sr, delay_ms, gain, cutoff_hz=3000):
        self.gain = gain
        self.delay_samples = int(sr * (delay_ms / 1000))
        self.buffer = np.zeros(self.delay_samples + 256)
        self.b, self.a = signal.butter(1, cutoff_hz / (0.5 * sr), btype="low")
        self.zi = signal.lfilter_zi(self.b, self.a)

    def process(self, block):
        self.buffer[: -len(block)] = self.buffer[len(block) :]
        self.buffer[-len(block) :] = block

        delayed = self.buffer[: len(block)]
        output, self.zi = signal.lfilter(self.b, self.a, delayed, zi=self.zi)
        return output * self.gain


# Start using them


def run_audio_pipeline(input_signal, sr, data_hz, data_fr, use_hrtf=True):
    BLOCK_SIZE = 128
    total_samples = len(input_signal)

    out_left = np.zeros(total_samples)
    out_right = np.zeros(total_samples)

    # make instance
    crossover = RealTimeCrossover(sr, cutoff_freq=120)
    eq_mid = RealTimeActiveEQ(sr, target_freq=120)
    eq_side = RealTimeActiveEQ(sr, target_freq=120)

    # HRTF instance
    mid_l_9 = RealTimeOverlapSaveOverlap(data_hz["left"][:, 0], BLOCK_SIZE)
    mid_r_9 = RealTimeOverlapSaveOverlap(data_hz["right"][:, 0], BLOCK_SIZE)
    # mid_l_63 = RealTimeOverlapSaveOverlap(data_hz["left"][:, 0], BLOCK_SIZE)
    # mid_r_63 = RealTimeOverlapSaveOverlap(data_hz["right"][:, 0], BLOCK_SIZE)
    # mid_l_48 = RealTimeOverlapSaveOverlap(data_fr["left"][:, 48], BLOCK_SIZE)
    # mid_r_48 = RealTimeOverlapSaveOverlap(data_fr["right"][:, 48], BLOCK_SIZE)

    side_l_36 = RealTimeOverlapSaveOverlap(data_hz["left"][:, 63], BLOCK_SIZE)
    side_r_36 = RealTimeOverlapSaveOverlap(data_hz["right"][:, 9], BLOCK_SIZE)
    # side_l_48 = RealTimeOverlapSaveOverlap(data_fr["left"][:, 48], BLOCK_SIZE)
    # side_r_48 = RealTimeOverlapSaveOverlap(data_fr["right"][:, 48], BLOCK_SIZE)

    # Instance of Reflections
    delay_r = RealTimeDelayLine(sr, delay_ms=18, gain=0.4)
    reflect_r_l = RealTimeOverlapSaveOverlap(data_hz["left"][:, 40], BLOCK_SIZE)
    reflect_r_r = RealTimeOverlapSaveOverlap(data_hz["right"][:, 32], BLOCK_SIZE)

    delay_l = RealTimeDelayLine(sr, delay_ms=23, gain=0.4)
    reflect_l_l = RealTimeOverlapSaveOverlap(data_hz["left"][:, 40], BLOCK_SIZE)
    reflect_l_r = RealTimeOverlapSaveOverlap(data_hz["right"][:, 32], BLOCK_SIZE)

    # Instance for cross delay
    # Even you hear from one side there is delay sound to the other side ear
    cross_delay_l = RealTimeDelayLine(sr, delay_ms=0.5, gain=1.0, cutoff_hz=1000)
    cross_delay_r = RealTimeDelayLine(sr, delay_ms=0.5, gain=1.0, cutoff_hz=1000)
    cross_feed_level = 0.04

    # cut off high tone (higher thatn 9000 Hz)
    b_lp, a_lp = signal.butter(2, 6000 / (0.5 * sr), btype="low")
    ir_lp_l = RealTimeBiquadFilter(b_lp, a_lp)
    ir_lp_r = RealTimeBiquadFilter(b_lp, a_lp)

    # Decrease the high tone ("I heard decrease the high tone make it more comfortable")
    b_sh, a_sh = get_high_shelf_coeffs(sr, f0=3000, gain_db=-5.0)
    side_shelf_l = RealTimeBiquadFilter(b_sh, a_sh)
    side_shelf_r = RealTimeBiquadFilter(b_sh, a_sh)

    # ★ 空間オーディオのブレンド率 (1.0で空間全振り、0.0でノーマル)
    # 友達の好みに合わせて 0.5 ~ 0.8 あたりでチューニングするのがおすすめ！
    alpha = 0.7

    # 128サンプルずつのブロック処理ループ
    for i in range(0, total_samples, BLOCK_SIZE):
        block = input_signal[i : i + BLOCK_SIZE]
        if len(block) < BLOCK_SIZE:
            break

        # バイパススイッチがONの場合
        if not use_hrtf:
            out_left[i : i + BLOCK_SIZE] = block[:, 0]
            out_right[i : i + BLOCK_SIZE] = block[:, 1]
            continue

        # --- 空間オーディオ処理パイプライン ---
        # 1. Mid / Side 分解
        mid = (block[:, 0] + block[:, 1]) / 2
        side = (block[:, 0] - block[:, 1]) / 2

        # 2. Active EQ
        mid = eq_mid.process(mid)
        side = eq_side.process(side)

        # 3. Crossover
        mid_low, mid_high = crossover.process(mid)

        # 4. HRTF 畳み込み (Mid/Sideの高域)
        mh_l = (
            mid_l_9.process(mid_high)  # + mid_l_63.process(mid_high)
            # + mid_l_48.process(mid_high)
        )
        mh_r = (
            mid_r_9.process(mid_high)  # + mid_r_63.process(mid_high)
            # + mid_r_48.process(mid_high)
        )

        mh_l *= 0.85
        mh_r *= 0.85

        s_l = side_l_36.process(side) * 0.6  # + side_l_48.process(side)
        s_r = side_r_36.process(side) * 0.6

        s_l = side_shelf_l.process(s_l) * 1.2
        s_r = side_shelf_r.process(s_r) * 1.2  # + side_r_48.process(side)

        # 5. 初期反射音の生成
        ref_r_source = delay_r.process(mid)
        ref_l_source = delay_l.process(mid)

        # ★ クロスフィード構造
        # 右の壁からの反射（ref_r）も左の耳（reflect_r_l）と右の耳（reflect_r_r）両方に届くように統合
        er_l = reflect_r_l.process(ref_r_source) + reflect_l_l.process(ref_l_source)
        er_r = reflect_r_r.process(ref_r_source) + reflect_l_r.process(ref_l_source)

        # 6. 空間オーディオ成分（Wet）の統合
        wet_l = (mid_low + mh_l + s_l + er_l) * 0.6
        wet_r = (mid_low + mh_r - s_r + er_r) * 0.6

        wet_l = ir_lp_l.process(wet_l)
        wet_r = ir_lp_r.process(wet_r)

        # 7. 元のステレオ成分（Dry）の取得
        dry_l = block[:, 0]
        dry_r = block[:, 1]

        # ★ 8. Bose流：Dry と Wet の線形ブレンド
        final_l = (alpha * wet_l) + ((1.0 - alpha) * dry_l)
        final_r = (alpha * wet_r) + ((1.0 - alpha) * dry_r)

        g_delayed_l = cross_delay_l.process(final_l)
        g_delayed_r = cross_delay_r.process(final_r)

        out_l_mixed = final_l + (g_delayed_r * cross_feed_level)
        out_r_mixed = final_r + (g_delayed_l * cross_feed_level)

        # 9. ゲイン調整と非線形クリッピング（修正したミックスを出力に流す）
        out_left[i : i + BLOCK_SIZE] = np.tanh(out_l_mixed * 0.85)
        out_right[i : i + BLOCK_SIZE] = np.tanh(out_r_mixed * 0.85)
        # 9. ゲイン調整と非線形クリッピング (tanh)

    return np.vstack([out_left, out_right]).T


# ==========================================
# 3. テスト実行用スクリプト
# ==========================================
data_horizontal = loadmat("small_pinna_final.mat")
data_frontal = loadmat("small_pinna_frontal.mat")
sample, sr = sf.read("sample.mp3")

# # 音源の実行
stereo_output = run_audio_pipeline(
    sample, sr, data_horizontal, data_frontal, use_hrtf=True
)
# sf.write("final_rt_sim.mp3", stereo_output, sr)
import io
from pydub import AudioSegment

# メモリ上にWAVとして一度書き出す
wav_io = io.BytesIO()
sf.write(wav_io, stereo_output, sr, format="WAV", subtype="PCM_16")
wav_io.seek(0)

# pydubを使って、最高音質（320kbps）のmp3としてカチッと保存する
sound = AudioSegment.from_wav(wav_io)
sound.export("final_rt_sim.mp3", format="mp3", bitrate="320k")
