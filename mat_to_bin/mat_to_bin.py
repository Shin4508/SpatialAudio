import numpy as np
import soundfile as sf
import glob
import os


def wav_to_bin(wav_filename):
    bin_filename = wav_filename.replace(".wav", ".bin")

    try:
        # soundfileは自動的にデータを -1.0〜1.0 のfloat形式で読み込みます
        data, sample_rate = sf.read(wav_filename)

        # もしステレオやマルチチャンネルだった場合、Lチャンネル（最初のチャンネル）のみを使用
        if len(data.shape) > 1:
            print(
                f"⚠️ {wav_filename} はマルチチャンネルです。最初のチャンネルのみ抽出します。"
            )
            data = data[:, 0]

        # Rustの `f32` (リトルエンディアンの 32-bit 浮動小数点数) に変換
        data_f32 = np.array(data, dtype="<f4").flatten()

        # バイナリファイルとして書き出し
        data_f32.tofile(bin_filename)
        print(
            f"✅ 変換成功: {wav_filename} -> {bin_filename} ({sample_rate}Hz, 長さ: {len(data_f32)} サンプル)"
        )

    except Exception as e:
        print(f"❌ エラー: {wav_filename} の処理中に問題が発生しました。詳細: {e}")


if __name__ == "__main__":
    wav_files = glob.glob("*.wav")

    if not wav_files:
        print("⚠️ カレントディレクトリに .wav ファイルが見つかりません。")

    for wav_file in wav_files:
        wav_to_bin(wav_file)
