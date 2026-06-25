// src/main.c
#include "hrtf_data.h"
#include <stdint.h>

// オーディオバッファのサイズ（低遅延を狙うなら32〜128サンプル）
#define BUFFER_SIZE 64

// 過去の音声入力を記憶しておくディレイライン（畳み込み用バッファ）
static float input_history[NUM_TAPS] = {0.0f};
static uint16_t history_index = 0;

// 立体音響の定位させたい角度のインデックス（例: 0〜35）
// 本番はこれをリアルタイム（センサーやMIDI）に変えられるようにします
static uint16_t current_azimuth_left = 63;
static uint16_t current_azimuth_right = 9;

/* * リアルタイム空間オーディオ処理関数
 * モノラル入力音声（input）を受け取り、HRTFを畳み込んでL/Rのステレオで出力する
 */
void process_spatial_audio(const float *input_buffer, float *output_left,
                           float *output_right, uint16_t size) {
  for (uint16_t i = 0; i < size; i++) {
    // 1. 最新の入力サンプルをディレイラインに保存
    input_history[history_index] = input_buffer[i];

    float acc_left = 0.0f;
    float acc_right = 0.0f;
    uint16_t h_idx = history_index;

    // 2. HRTFの畳み込み（FIRフィルター計算）
    // H723ZGのCortex-M7は、このループ内の「積和演算（MAC）」を1クロックで処理できます！
    for (uint16_t t = 0; t < NUM_TAPS; t++) {
      acc_left += input_history[h_idx] * hrtf_left[current_azimuth_left][t];
      acc_right += input_history[h_idx] * hrtf_right[current_azimuth_right][t];

      // リングバッファのインデックスを過去方向に巻き戻す
      if (h_idx == 0) {
        h_idx = NUM_TAPS - 1;
      } else {
        h_idx--;
      }
    }

    // 3. 出力バッファへ書き込み
    output_left[i] = acc_left;
    output_right[i] = acc_right;

    // 4. リングバッファのインデックスを次に進める
    history_index++;
    if (history_index >= NUM_TAPS) {
      history_index = 0;
    }
  }
}

int main(void) {
  // 信号処理用の入出力バッファを確保
  float mic_in[BUFFER_SIZE] = {0.0f};
  float audio_out_l[BUFFER_SIZE] = {0.0f};
  float audio_out_r[BUFFER_SIZE] = {0.0f};

  // テスト用のダミー入力信号（1kHzの正弦波など）を突っ込むならココ
  mic_in[0] = 1.0f; // インパルス応答テスト用

  /* マイコン駆動のメイン無限ループ */
  while (1) {
    // 本番はここに、DMA（Direct Memory
    // Access）から「オーディオ入力が半分溜まったぞ！」
    // という割り込み通知が来るのを待つ処理が入ります。

    // 空間オーディオ計算（爆速で処理が終わります）
    process_spatial_audio(mic_in, audio_out_l, audio_out_r, BUFFER_SIZE);

    // 擬似的なウェイト（テスト用）
    for (volatile int i = 0; i < 10000; i++)
      ;
  }

  return 0;
}
