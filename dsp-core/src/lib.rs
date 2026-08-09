#![no_std]
#![no_main]

use core::f32::consts::PI;
use defmt::*;
use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_rp::pio::{Pio, PioPin};
use embassy_rp::pio_program;
use libm::{cosf, sinf};
use {defmt_rtt as _, panic_probe as _};

const SAMPLE_RATE: f32 = 48000.0;
const BLOCK_SIZE: usize = 128;

// --- 1. 固定長Biquadフィルタ ---
#[derive(Clone, Copy)]
pub struct RealTimeBiquadFilter {
    b: [f32; 3],
    a: [f32; 3],
    z1: f32,
    z2: f32,
}

impl RealTimeBiquadFilter {
    pub const fn new(b: [f32; 3], a: [f32; 3]) -> Self {
        Self {
            b,
            a,
            z1: 0.0,
            z2: 0.0,
        }
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        for (x, y) in input.iter().zip(output.iter_mut()) {
            let out = self.b[0] * (*x) + self.z1;
            self.z1 = self.b[1] * (*x) - self.a[1] * out + self.z2;
            self.z2 = self.b[2] * (*x) - self.a[2] * out;
            *y = out;
        }
    }
}

// Low-pass Filter 係数計算関数
fn butter_lowpass_2nd(sr: f32, cutoff: f32) -> ([f32; 3], [f32; 3]) {
    let w0 = 2.0 * PI * cutoff / sr;
    let alpha = sinf(w0) / (2.0 * 0.7071);
    let cos_w0 = cosf(w0);

    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = (1.0 - cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

// --- 2. 空間オーディオ DSPエンジン（固定長） ---
pub struct SpatialAudioEngine {
    lpf_l: RealTimeBiquadFilter,
    lpf_r: RealTimeBiquadFilter,
}

impl SpatialAudioEngine {
    pub fn new(sr: f32) -> Self {
        let (b, a) = butter_lowpass_2nd(sr, 6000.0);
        Self {
            lpf_l: RealTimeBiquadFilter::new(b, a),
            lpf_r: RealTimeBiquadFilter::new(b, a),
        }
    }

    pub fn process_block(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        // DSP処理（メモリ割り当てなし）
        self.lpf_l.process(in_l, out_l);
        self.lpf_r.process(in_r, out_r);
    }
}

// --- 3. 静的バッファ（BSS領域に配置しスタックオーバーフローを予防） ---
static mut DSP_ENGINE: Option<SpatialAudioEngine> = None;
static mut IN_L: [f32; BLOCK_SIZE] = [0.0; BLOCK_SIZE];
static mut IN_R: [f32; BLOCK_SIZE] = [0.0; BLOCK_SIZE];
static mut OUT_L: [f32; BLOCK_SIZE] = [0.0; BLOCK_SIZE];
static mut OUT_R: [f32; BLOCK_SIZE] = [0.0; BLOCK_SIZE];

// --- 4. メイン処理ルーチン ---
#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    info!("RP2350 DSP Board Init...");

    // DSPエンジンの構築
    unsafe {
        DSP_ENGINE = Some(SpatialAudioEngine::new(SAMPLE_RATE));
    }

    // 画像から取得した実際のGPIO割り当て
    // 入力ピン (Bluetooth BTM334)
    let _bt_lrck = p.PIN_0; // GPIO0
    let _bt_data = p.PIN_1; // GPIO1
    let _bt_bck = p.PIN_2; // GPIO2

    // 出力ピン (PCM5102A DAC)
    let _dac_lrck = p.PIN_3; // GPIO3
    let _dac_data = p.PIN_4; // GPIO4
    let _dac_bck = p.PIN_5; // GPIO5

    info!("GPIO Initialized: BT_I2S(0,1,2), DAC_I2S(3,4,5)");

    // リアルタイム制御ループ
    loop {
        unsafe {
            // STEP A: BT受信バッファ（i32）から f32 正規化データ（-1.0〜1.0）へコピー
            for i in 0..BLOCK_SIZE {
                // 仮の読み込み用ダミー（実際のI2S DMA割り込みバッファと結合）
                IN_L[i] = 0.0;
                IN_R[i] = 0.0;
            }

            // STEP B: DSP処理実行
            if let Some(engine) = DSP_ENGINE.as_mut() {
                engine.process_block(&IN_L, &IN_R, &mut OUT_L, &mut OUT_R);
            }

            // STEP C: DAC出力バッファ（i32）へ戻して転送設定
            for i in 0..BLOCK_SIZE {
                let _sample_l = (OUT_L[i] * 8388607.0) as i32;
                let _sample_r = (OUT_R[i] * 8388607.0) as i32;
            }
        }

        // DMAバッファの周回同期ウェイト（embassyのtimer/dma非同期待機）
        embassy_time::Timer::after_micros(2666).await; // 128サンプル @ 48kHz ≒ 2.66ms
    }
}
