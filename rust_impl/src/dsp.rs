use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

// ==========================================
// 1. フィルタ係数計算
// ==========================================

fn get_low_shelf_coeffs(sr: f32, f0: f32, gain_db: f32, q: f32) -> ([f32; 3], [f32; 3]) {
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * f0 / sr;
    let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0).sqrt();
    let cos_w0 = w0.cos();
    let sqrt_a_alpha_2 = 2.0 * a.sqrt() * alpha;

    let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + sqrt_a_alpha_2);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - sqrt_a_alpha_2);
    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + sqrt_a_alpha_2;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - sqrt_a_alpha_2;

    // RustのBiquad用に a0 で正規化
    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

fn get_high_shelf_coeffs(sr: f32, f0: f32, gain_db: f32, q: f32) -> ([f32; 3], [f32; 3]) {
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * f0 / sr;
    let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0).sqrt();
    let cos_w0 = w0.cos();
    let sqrt_a_alpha_2 = 2.0 * a.sqrt() * alpha;

    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + sqrt_a_alpha_2);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - sqrt_a_alpha_2);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + sqrt_a_alpha_2;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - sqrt_a_alpha_2;

    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

// 簡易的な2次バターワースLPF係数
fn butter_lowpass_2nd(sr: f32, cutoff: f32) -> ([f32; 3], [f32; 3]) {
    let w0 = 2.0 * PI * cutoff / sr;
    let alpha = w0.sin() / (2.0 * 0.7071); // Q = 1/sqrt(2)
    let cos_w0 = w0.cos();

    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = (1.0 - cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

// ==========================================
// 2. リアルタイム処理クラス群
// ==========================================

#[derive(Clone)]
pub struct RealTimeBiquadFilter {
    b: [f32; 3],
    a: [f32; 3],
    z1: f32,
    z2: f32,
}

impl RealTimeBiquadFilter {
    pub fn new(b: [f32; 3], a: [f32; 3]) -> Self {
        Self {
            b,
            a,
            z1: 0.0,
            z2: 0.0,
        }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(input.len());
        for &x in input {
            // Direct Form II Transposed
            let y = self.b[0] * x + self.z1;
            self.z1 = self.b[1] * x - self.a[1] * y + self.z2;
            self.z2 = self.b[2] * x - self.a[2] * y;
            output.push(y);
        }
        output
    }
}

pub struct RealTimeCrossover {
    lp: RealTimeBiquadFilter,
    hp: RealTimeBiquadFilter,
}

impl RealTimeCrossover {
    pub fn new(sr: f32, cutoff_freq: f32) -> Self {
        let (b_lp, a_lp) = butter_lowpass_2nd(sr, cutoff_freq);
        // HPFはLPFの係数を元に簡易生成（実用では正確なHPF係数計算を推奨）
        let mut b_hp = b_lp;
        b_hp[0] = (1.0 + b_lp[0]) / 2.0;
        b_hp[1] = -(1.0 + b_lp[0]);
        b_hp[2] = (1.0 + b_lp[0]) / 2.0;

        Self {
            lp: RealTimeBiquadFilter::new(b_lp, a_lp),
            hp: RealTimeBiquadFilter::new(b_hp, a_lp),
        }
    }

    pub fn process(&mut self, block: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let low = self.lp.process(block).iter().map(|&x| x * 1.5).collect();
        let high = self.hp.process(block);
        (low, high)
    }
}

pub struct RealTimeActiveEQ {
    sr: f32,
    target_freq: f32,
    filter: RealTimeBiquadFilter,
}

impl RealTimeActiveEQ {
    pub fn new(sr: f32, target_freq: f32) -> Self {
        let (b, a) = get_low_shelf_coeffs(sr, target_freq, 0.0, 0.707);
        Self {
            sr,
            target_freq,
            filter: RealTimeBiquadFilter::new(b, a),
        }
    }

    pub fn process(&mut self, block: &[f32]) -> Vec<f32> {
        let rms = (block.iter().map(|&x| x * x).sum::<f32>() / block.len() as f32).sqrt();
        let threshold = 0.05;
        let gain_db = if rms < threshold {
            6.0 * f32::max(0.0, 1.0 - (rms / threshold))
        } else {
            0.0
        };

        let (b, a) = get_low_shelf_coeffs(self.sr, self.target_freq, gain_db, 0.707);
        self.filter.b = b;
        self.filter.a = a;

        self.filter.process(block)
    }
}

pub struct RealTimeDelayLine {
    gain: f32,
    buffer: Vec<f32>,
    write_idx: usize,
    delay_samples: usize,
    lpf: RealTimeBiquadFilter,
}

impl RealTimeDelayLine {
    pub fn new(sr: f32, delay_ms: f32, gain: f32, cutoff_hz: f32) -> Self {
        let delay_samples = (sr * (delay_ms / 1000.0)) as usize;
        let (b, a) = butter_lowpass_2nd(sr, cutoff_hz);
        Self {
            gain,
            buffer: vec![0.0; delay_samples + 1024], // リングバッファ
            write_idx: 0,
            delay_samples,
            lpf: RealTimeBiquadFilter::new(b, a),
        }
    }

    pub fn process(&mut self, block: &[f32]) -> Vec<f32> {
        let mut delayed = Vec::with_capacity(block.len());
        let cap = self.buffer.len();

        for &sample in block {
            let read_idx = (self.write_idx + cap - self.delay_samples) % cap;
            delayed.push(self.buffer[read_idx]);

            self.buffer[self.write_idx] = sample;
            self.write_idx = (self.write_idx + 1) % cap;
        }

        self.lpf
            .process(&delayed)
            .iter()
            .map(|&x| x * self.gain)
            .collect()
    }
}

pub struct RealTimeOverlapSave {
    l: usize,
    fft_size: usize,
    h_fft: Vec<Complex<f32>>,
    input_buffer: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
}

impl RealTimeOverlapSave {
    pub fn new(hrtf_impulse: &[f32], block_size: usize) -> Self {
        let n = hrtf_impulse.len();
        let m = block_size + n - 1;
        let fft_size = m.next_power_of_two(); // N = 2^ceil(log2(M))

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        // インパルス応答をパディングしてFFT
        let mut h_padded: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];
        for (i, &val) in hrtf_impulse.iter().enumerate() {
            h_padded[i] = Complex::new(val, 0.0);
        }
        fft.process(&mut h_padded);

        Self {
            l: block_size,
            fft_size,
            h_fft: h_padded,
            input_buffer: vec![0.0; fft_size],
            fft,
            ifft,
        }
    }

    pub fn process(&mut self, block: &[f32]) -> Vec<f32> {
        // バッファを左にシフトし、新しいブロックを末尾に追加
        self.input_buffer.copy_within(self.l.., 0);
        let end_idx = self.fft_size;
        let start_idx = end_idx - self.l;
        self.input_buffer[start_idx..].copy_from_slice(block);

        let mut work_buf: Vec<Complex<f32>> = self
            .input_buffer
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        self.fft.process(&mut work_buf);

        // 周波数領域での乗算
        for (x, h) in work_buf.iter_mut().zip(self.h_fft.iter()) {
            *x *= *h;
        }

        self.ifft.process(&mut work_buf);

        // スケール調整と実部の抽出（最後の L サンプルが有効データ）
        let scale = 1.0 / self.fft_size as f32;
        work_buf[start_idx..].iter().map(|c| c.re * scale).collect()
    }
}

// ==========================================
// 3. メインパイプライン
// ==========================================

pub fn run_audio_pipeline(
    input_l: &[f32],
    input_r: &[f32],
    sr: f32,
    hrtf_data: &HrtfData, // 仮想のHRTF構造体
    use_hrtf: bool,
) -> (Vec<f32>, Vec<f32>) {
    const BLOCK_SIZE: usize = 128;
    let total_samples = input_l.len();

    let mut out_left = vec![0.0; total_samples];
    let mut out_right = vec![0.0; total_samples];

    let mut crossover = RealTimeCrossover::new(sr, 120.0);
    let mut eq_mid = RealTimeActiveEQ::new(sr, 120.0);
    let mut eq_side = RealTimeActiveEQ::new(sr, 120.0);

    // HRTFや反射のインスタンス化 (※ HrtfData から適宜スライスを渡す想定)
    let mut mid_l_9 = RealTimeOverlapSave::new(&hrtf_data.left_0, BLOCK_SIZE);
    let mut mid_r_9 = RealTimeOverlapSave::new(&hrtf_data.right_0, BLOCK_SIZE);
    let mut side_l_36 = RealTimeOverlapSave::new(&hrtf_data.left_63, BLOCK_SIZE); // Python側の実装に合わせる
    let mut side_r_36 = RealTimeOverlapSave::new(&hrtf_data.right_9, BLOCK_SIZE);

    let mut reflect_r_l = RealTimeOverlapSave::new(&hrtf_data.left_40, BLOCK_SIZE);
    let mut reflect_r_r = RealTimeOverlapSave::new(&hrtf_data.right_32, BLOCK_SIZE);
    let mut reflect_l_l = RealTimeOverlapSave::new(&hrtf_data.left_40, BLOCK_SIZE);
    let mut reflect_l_r = RealTimeOverlapSave::new(&hrtf_data.right_32, BLOCK_SIZE);

    let mut delay_r = RealTimeDelayLine::new(sr, 18.0, 0.4, 3000.0);
    let mut delay_l = RealTimeDelayLine::new(sr, 23.0, 0.4, 3000.0);
    let mut cross_delay_l = RealTimeDelayLine::new(sr, 0.5, 1.0, 1000.0);
    let mut cross_delay_r = RealTimeDelayLine::new(sr, 0.5, 1.0, 1000.0);

    let (b_lp, a_lp) = butter_lowpass_2nd(sr, 6000.0);
    let mut ir_lp_l = RealTimeBiquadFilter::new(b_lp, a_lp);
    let mut ir_lp_r = RealTimeBiquadFilter::new(b_lp, a_lp);

    let (b_sh, a_sh) = get_high_shelf_coeffs(sr, 3000.0, -5.0, 0.707);
    let mut side_shelf_l = RealTimeBiquadFilter::new(b_sh, a_sh);
    let mut side_shelf_r = RealTimeBiquadFilter::new(b_sh, a_sh);

    let alpha = 0.7;
    let cross_feed_level = 0.06;

    for i in (0..total_samples).step_by(BLOCK_SIZE) {
        let end = usize::min(i + BLOCK_SIZE, total_samples);
        if end - i < BLOCK_SIZE {
            break;
        } // 簡単のため端数は無視

        let block_l = &input_l[i..end];
        let block_r = &input_r[i..end];

        if !use_hrtf {
            out_left[i..end].copy_from_slice(block_l);
            out_right[i..end].copy_from_slice(block_r);
            continue;
        }

        // 1. Mid / Side 分解
        let mid: Vec<f32> = block_l
            .iter()
            .zip(block_r)
            .map(|(l, r)| (l + r) / 2.0)
            .collect();
        let side: Vec<f32> = block_l
            .iter()
            .zip(block_r)
            .map(|(l, r)| (l - r) * 0.8 / 2.0)
            .collect();

        // 2. Active EQ
        let mid_eq = eq_mid.process(&mid);
        let side_eq = eq_side.process(&side);

        // 3. Crossover
        let (mid_low, mid_high) = crossover.process(&mid_eq);

        // 4. HRTF 畳み込み
        let mh_l = mid_l_9
            .process(&mid_high)
            .iter()
            .map(|&x| x * 0.85)
            .collect::<Vec<_>>();
        let mh_r = mid_r_9
            .process(&mid_high)
            .iter()
            .map(|&x| x * 0.85)
            .collect::<Vec<_>>();

        let mut s_l = side_l_36
            .process(&side_eq)
            .iter()
            .map(|&x| x * 0.6)
            .collect::<Vec<_>>();
        let mut s_r = side_r_36
            .process(&side_eq)
            .iter()
            .map(|&x| x * 0.6)
            .collect::<Vec<_>>();
        s_l = side_shelf_l
            .process(&s_l)
            .iter()
            .map(|&x| x * 1.2)
            .collect();
        s_r = side_shelf_r
            .process(&s_r)
            .iter()
            .map(|&x| x * 1.2)
            .collect();

        // 5. 初期反射音の生成
        let ref_r_source = delay_r.process(&mid_eq);
        let ref_l_source = delay_l.process(&mid_eq);

        let er_l: Vec<f32> = reflect_r_l
            .process(&ref_r_source)
            .iter()
            .zip(reflect_l_l.process(&ref_l_source))
            .map(|(r, l)| r + l)
            .collect();
        let er_r: Vec<f32> = reflect_r_r
            .process(&ref_r_source)
            .iter()
            .zip(reflect_l_r.process(&ref_l_source))
            .map(|(r, l)| r + l)
            .collect();

        // 6. Wet統合
        let wet_l: Vec<f32> = mid_low
            .iter()
            .enumerate()
            .map(|(idx, &m_low)| (m_low + mh_l[idx] + s_l[idx] + er_l[idx]) * 0.6)
            .collect();
        let wet_r: Vec<f32> = mid_low
            .iter()
            .enumerate()
            .map(|(idx, &m_low)| (m_low + mh_r[idx] - s_r[idx] + er_r[idx]) * 0.6)
            .collect();

        let wet_l_filtered = ir_lp_l.process(&wet_l);
        let wet_r_filtered = ir_lp_r.process(&wet_r);

        // 7 & 8. 線形ブレンドとクロスフィード
        let final_l: Vec<f32> = wet_l_filtered
            .iter()
            .zip(block_l)
            .map(|(&w, &d)| (alpha * w) + ((1.0 - alpha) * d))
            .collect();
        let final_r: Vec<f32> = wet_r_filtered
            .iter()
            .zip(block_r)
            .map(|(&w, &d)| (alpha * w) + ((1.0 - alpha) * d))
            .collect();

        let g_delayed_l = cross_delay_l.process(&final_l);
        let g_delayed_r = cross_delay_r.process(&final_r);

        // 9. 非線形クリッピング
        for j in 0..BLOCK_SIZE {
            let mixed_l = final_l[j] + (g_delayed_r[j] * cross_feed_level);
            let mixed_r = final_r[j] + (g_delayed_l[j] * cross_feed_level);
            out_left[i + j] = (mixed_l * 0.85).tanh();
            out_right[i + j] = (mixed_r * 0.85).tanh();
        }
    }

    (out_left, out_right)
}

// HRTFのインパルス応答を保持するダミー構造体
pub struct HrtfData {
    pub left_0: Vec<f32>,
    pub right_0: Vec<f32>,
    pub left_63: Vec<f32>,
    pub right_9: Vec<f32>,
    pub left_40: Vec<f32>,
    pub right_32: Vec<f32>,
}
// --- 前回のコード (run_audio_pipeline や 構造体の定義) はここにある想定 ---

use hound;

fn main() {
    println!("🚀 テスト実行を開始します...");

    // 1. サンプル音源の読み込み (WAV形式に変換済みのものを使用)
    let mut reader = hound::WavReader::open("spatial_sound/data/ouput.wav")
        .expect("サンプルのWAVファイルが見つかりません。");
    let spec = reader.spec();
    let sr = spec.sample_rate as f32;

    // ステレオ(Interleaved: L, R, L, R...)のi16データを、LとRのf32配列に分離する(-1.0 ~ 1.0に正規化)
    let mut input_l = Vec::new();
    let mut input_r = Vec::new();

    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    for chunk in samples.chunks(2) {
        if chunk.len() == 2 {
            input_l.push(chunk[0] as f32 / i16::MAX as f32);
            input_r.push(chunk[1] as f32 / i16::MAX as f32);
        }
    }
    println!("✅ 音源ロード完了: {} Hz, {} サンプル", sr, input_l.len());

    // 2. HRTFデータの準備
    // ※ 実際はここで "hrtf_left_9.wav" などのWAVファイルからインパルス応答を読み込みます。
    // 今回はテスト実行を回すため、ダミーの配列（すべて0.0）で構造体を初期化します。
    let hrtf_data = HrtfData {
        left_0: load_impulse_response("spatial_sound/data/hrtf_left_9.wav")
            .unwrap_or(vec![0.0; 512]),
        right_0: load_impulse_response("spatial_sound/data/hrtf_right_9.wav")
            .unwrap_or(vec![0.0; 512]),
        left_63: load_impulse_response("spatial_sound/data/hrtf_left_63.wav")
            .unwrap_or(vec![0.0; 512]),
        right_9: load_impulse_response("spatial_sound/data/hrtf_right_9.wav")
            .unwrap_or(vec![0.0; 512]),
        left_40: load_impulse_response("spatial_sound/data/hrtf_left_40.wav")
            .unwrap_or(vec![0.0; 512]),
        right_32: load_impulse_response("spatial_sound/data/hrtf_right_32.wav")
            .unwrap_or(vec![0.0; 512]),
    };
    println!("✅ HRTFデータ準備完了");

    // 3. 空間オーディオパイプラインの実行
    println!("🎧 空間オーディオ処理を実行中...");
    let (out_l, out_r) = run_audio_pipeline(&input_l, &input_r, sr, &hrtf_data, true);

    // 4. 結果をWAVファイルとして書き出し
    let mut writer =
        hound::WavWriter::create("final_rt_sim.wav", spec).expect("ファイルの作成に失敗しました。");

    for i in 0..out_l.len() {
        // f32 (-1.0 ~ 1.0) を i16 に戻す。クリッピング防止でclampする
        let l_sample = (out_l[i].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let r_sample = (out_r[i].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;

        writer.write_sample(l_sample).unwrap();
        writer.write_sample(r_sample).unwrap();
    }
    writer
        .finalize()
        .expect("ファイルの書き込みに失敗しました。");

    println!("🎉 処理完了！ 'final_rt_sim.wav' として保存しました。");
}

// （おまけ）WAVファイルからインパルス応答(モノラル)を読み込むヘルパー関数
fn load_impulse_response(filename: &str) -> Option<Vec<f32>> {
    let mut reader = hound::WavReader::open(filename).ok()?;
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();
    Some(samples)
}
