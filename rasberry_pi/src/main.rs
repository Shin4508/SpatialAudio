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
// 3. リアルタイム DSP エンジン
// ==========================================
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Producer, Split};
use std::process::Command;

// エフェクトの「状態」を維持するための構造体
pub struct SpatialAudioEngine {
    crossover: RealTimeCrossover,
    eq_mid: RealTimeActiveEQ,
    eq_side: RealTimeActiveEQ,
    mid_l_9: RealTimeOverlapSave,
    mid_r_9: RealTimeOverlapSave,
    side_l_36: RealTimeOverlapSave,
    side_r_36: RealTimeOverlapSave,
    reflect_r_l: RealTimeOverlapSave,
    reflect_r_r: RealTimeOverlapSave,
    reflect_l_l: RealTimeOverlapSave,
    reflect_l_r: RealTimeOverlapSave,
    delay_r: RealTimeDelayLine,
    delay_l: RealTimeDelayLine,
    cross_delay_l: RealTimeDelayLine,
    cross_delay_r: RealTimeDelayLine,
    ir_lp_l: RealTimeBiquadFilter,
    ir_lp_r: RealTimeBiquadFilter,
    side_shelf_l: RealTimeBiquadFilter,
    side_shelf_r: RealTimeBiquadFilter,
}

impl SpatialAudioEngine {
    pub fn new(sr: f32, hrtf_data: &HrtfData, block_size: usize) -> Self {
        let (b_lp, a_lp) = butter_lowpass_2nd(sr, 6000.0);
        let (b_sh, a_sh) = get_high_shelf_coeffs(sr, 3000.0, -5.0, 0.707);

        Self {
            crossover: RealTimeCrossover::new(sr, 120.0),
            eq_mid: RealTimeActiveEQ::new(sr, 85.0),
            eq_side: RealTimeActiveEQ::new(sr, 85.0),
            mid_l_9: RealTimeOverlapSave::new(&hrtf_data.left_0, block_size),
            mid_r_9: RealTimeOverlapSave::new(&hrtf_data.right_0, block_size),
            side_l_36: RealTimeOverlapSave::new(&hrtf_data.left_63, block_size),
            side_r_36: RealTimeOverlapSave::new(&hrtf_data.right_9, block_size),
            reflect_r_l: RealTimeOverlapSave::new(&hrtf_data.left_40, block_size),
            reflect_r_r: RealTimeOverlapSave::new(&hrtf_data.right_32, block_size),
            reflect_l_l: RealTimeOverlapSave::new(&hrtf_data.left_40, block_size),
            reflect_l_r: RealTimeOverlapSave::new(&hrtf_data.right_32, block_size),
            delay_r: RealTimeDelayLine::new(sr, 18.0, 0.4, 3000.0),
            delay_l: RealTimeDelayLine::new(sr, 23.0, 0.4, 3000.0),
            cross_delay_l: RealTimeDelayLine::new(sr, 0.5, 1.0, 1000.0),
            cross_delay_r: RealTimeDelayLine::new(sr, 0.5, 1.0, 1000.0),
            ir_lp_l: RealTimeBiquadFilter::new(b_lp, a_lp),
            ir_lp_r: RealTimeBiquadFilter::new(b_lp, a_lp),
            side_shelf_l: RealTimeBiquadFilter::new(b_sh, a_sh),
            side_shelf_r: RealTimeBiquadFilter::new(b_sh, a_sh),
        }
    }

    // 1ブロック分のリアルタイム処理
    pub fn process_block(&mut self, input_l: &[f32], input_r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let alpha = 0.7;
        let cross_feed_level = 0.06;

        let mid: Vec<f32> = input_l
            .iter()
            .zip(input_r)
            .map(|(l, r)| (l + r) / 2.0)
            .collect();
        let side: Vec<f32> = input_l
            .iter()
            .zip(input_r)
            .map(|(l, r)| (l - r) * 0.8 / 2.0)
            .collect();

        let mid_eq = self.eq_mid.process(&mid);
        let side_eq = self.eq_side.process(&side);
        let (mid_low, mid_high) = self.crossover.process(&mid_eq);

        let mh_l = self
            .mid_l_9
            .process(&mid_high)
            .iter()
            .map(|&x| x * 0.85)
            .collect::<Vec<_>>();
        let mh_r = self
            .mid_r_9
            .process(&mid_high)
            .iter()
            .map(|&x| x * 0.85)
            .collect::<Vec<_>>();

        let mut s_l = self
            .side_l_36
            .process(&side_eq)
            .iter()
            .map(|&x| x * 0.6)
            .collect::<Vec<_>>();
        let mut s_r = self
            .side_r_36
            .process(&side_eq)
            .iter()
            .map(|&x| x * 0.6)
            .collect::<Vec<_>>();
        s_l = self
            .side_shelf_l
            .process(&s_l)
            .iter()
            .map(|&x| x * 1.2)
            .collect();
        s_r = self
            .side_shelf_r
            .process(&s_r)
            .iter()
            .map(|&x| x * 1.2)
            .collect();

        let ref_r_source = self.delay_r.process(&mid_eq);
        let ref_l_source = self.delay_l.process(&mid_eq);

        let er_l: Vec<f32> = self
            .reflect_r_l
            .process(&ref_r_source)
            .iter()
            .zip(self.reflect_l_l.process(&ref_l_source))
            .map(|(r, l)| r + l)
            .collect();
        let er_r: Vec<f32> = self
            .reflect_r_r
            .process(&ref_r_source)
            .iter()
            .zip(self.reflect_l_r.process(&ref_l_source))
            .map(|(r, l)| r + l)
            .collect();

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

        let wet_l_filtered = self.ir_lp_l.process(&wet_l);
        let wet_r_filtered = self.ir_lp_r.process(&wet_r);

        let final_l: Vec<f32> = wet_l_filtered
            .iter()
            .zip(input_l)
            .map(|(&w, &d)| (alpha * w) + ((1.0 - alpha) * d))
            .collect();
        let final_r: Vec<f32> = wet_r_filtered
            .iter()
            .zip(input_r)
            .map(|(&w, &d)| (alpha * w) + ((1.0 - alpha) * d))
            .collect();

        let g_delayed_l = self.cross_delay_l.process(&final_l);
        let g_delayed_r = self.cross_delay_r.process(&final_r);

        let mut out_left = vec![0.0; input_l.len()];
        let mut out_right = vec![0.0; input_r.len()];

        for j in 0..input_l.len() {
            let mixed_l = final_l[j] + (g_delayed_r[j] * cross_feed_level);
            let mixed_r = final_r[j] + (g_delayed_l[j] * cross_feed_level);
            out_left[j] = (mixed_l * 1.2).tanh();
            out_right[j] = (mixed_r * 1.2).tanh();
        }

        (out_left, out_right)
    }
}

pub struct HrtfData {
    pub left_0: Vec<f32>,
    pub right_0: Vec<f32>,
    pub left_63: Vec<f32>,
    pub right_9: Vec<f32>,
    pub left_40: Vec<f32>,
    pub right_32: Vec<f32>,
}

fn load_impulse_response(filename: &str) -> Vec<f32> {
    let mut reader = match hound::WavReader::open(filename) {
        Ok(r) => r,
        Err(_) => {
            eprintln!("❌ エラー: HRTFファイル '{}' が見つかりません！", filename);
            // 見つからない場合は無音のダミーを返してパニックを防ぐ
            return vec![0.0; 256];
        }
    };

    let samples: Vec<f32> = reader
        .samples::<f32>()
        .map(|s| s.expect("サンプルの読み込みに失敗しました"))
        .collect();

    println!(
        "✅ {} を読み込みました！(長さ: {} サンプル)",
        filename,
        samples.len()
    );
    samples
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 空間オーディオ DSPエンジン起動...");

    let host = cpal::host_from_id(cpal::HostId::Jack).expect("JACKが動いていません");
    let input_device = host.default_input_device().expect("入力なし");
    let output_device = host.default_output_device().expect("出力なし");

    let config: cpal::StreamConfig = output_device.default_output_config()?.into();
    let sr = config.sample_rate.0 as f32;

    // バッファサイズ（通常PipeWire/JACKは 256 や 512 で回ります）
    let block_size = 256;

    // リングバッファ（入力を出力スレッドへ渡す土管）
    let ring = HeapRb::<f32>::new((sr * 0.1) as usize * 2);
    let (mut producer, mut consumer) = ring.split();

    // DSPエンジンの初期化
    let hrtf_data = HrtfData {
        left_0: load_impulse_response("hrtf_left_9.wav"),
        right_0: load_impulse_response("hrtf_right_9.wav"),
        left_63: load_impulse_response("hrtf_left_63.wav"),
        right_9: load_impulse_response("hrtf_right_9.wav"),
        left_40: load_impulse_response("hrtf_left_40.wav"),
        right_32: load_impulse_response("hrtf_right_32.wav"),
    };
    let mut engine = SpatialAudioEngine::new(sr, &hrtf_data, block_size);

    // 入力ストリーム
    let input_stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            for &s in data {
                let _ = producer.try_push(s);
            }
        },
        |err| eprintln!("入力エラー: {}", err),
        None,
    )?;

    // 一時バッファ
    let mut buf_l = Vec::with_capacity(block_size);
    let mut buf_r = Vec::with_capacity(block_size);

    // 出力ストリーム (ここでDSPを回す)
    let output_stream = output_device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &_| {
            buf_l.clear();
            buf_r.clear();

            // 必要な分だけリングバッファから取り出す（ステレオなので2個ずつ）
            for _ in 0..(data.len() / 2) {
                buf_l.push(consumer.try_pop().unwrap_or(0.0));
                buf_r.push(consumer.try_pop().unwrap_or(0.0));
            }

            // DSPエンジンに通す
            let (out_l, out_r) = engine.process_block(&buf_l, &buf_r);

            // ステレオに再合成して出力
            for (i, frame) in data.chunks_mut(2).enumerate() {
                frame[0] = out_l[i];
                frame[1] = out_r[i];
            }
        },
        |err| eprintln!("出力エラー: {}", err),
        None,
    )?;

    input_stream.play()?;
    output_stream.play()?;

    // 自動結線スクリプト
    std::thread::sleep(std::time::Duration::from_millis(500));
    let auto_link_script = r#"
        L_PORT=$(pw-link -io | grep -o 'bluez_input[^:]*:output_FL' | head -n 1)
        R_PORT=$(pw-link -io | grep -o 'bluez_input[^:]*:output_FR' | head -n 1)

        # 2. ✂️ OSが勝手に繋いだ「直結バイパス」を強制切断（エラーは無視）
        pw-link -d "$L_PORT" "alsa_output.usb-Creative_Technology_Ltd_Sound_Blaster_Play__3_YDSB1730613003087M-00.analog-stereo:playback_FL" 2>/dev/null || true
        pw-link -d "$R_PORT" "alsa_output.usb-Creative_Technology_Ltd_Sound_Blaster_Play__3_YDSB1730613003087M-00.analog-stereo:playback_FR" 2>/dev/null || true
        if [ -n "$L_PORT" ] && [ -n "$R_PORT" ]; then
            pw-link "$L_PORT" "cpal_client_in:in_0"
            pw-link "$R_PORT" "cpal_client_in:in_1"
        fi
    "#;
    Command::new("bash")
        .arg("-c")
        .arg(auto_link_script)
        .status()
        .unwrap();

    println!("🎧 DSP稼働中... (Ctrl+Cで終了)");
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
