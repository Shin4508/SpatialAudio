/* STM32H723ZGTx Memory Layout */
ENTRY(Reset_Handler)

/* 最高の計算速度を出すために、オーディオバッファを配置する超高速RAM（AXISRAM）などを定義 */
MEMORY
{
  FLASH (rx)      : ORIGIN = 0x08000000, LENGTH = 1024K
  DTCMRAM (rw)    : ORIGIN = 0x20000000, LENGTH = 128K
  SRAM1 (rw)      : ORIGIN = 0x24000000, LENGTH = 256K
  SRAM2 (rw)      : ORIGIN = 0x24040000, LENGTH = 56K
}

SECTIONS
{
  /* ベクタテーブル（割り込み処理のジャンプ先リスト）をFLASHの先頭に配置 */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } >FLASH

  /* 実行コード本体 */
  .text :
  {
    . = ALIGN(4);
    *(.text)
    *(.text*)
    . = ALIGN(4);
  } >FLASH

  /* 定数データ（ここにPythonから書き出したHRTFの巨大フィルター係数を載せます） */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >FLASH

  /* 初期値を持つグローバル変数など */
  .data :
  {
    . = ALIGN(4);
    *(.data)
    *(.data*)
    . = ALIGN(4);
  } >SRAM1 AT> FLASH

  /* 初期値が0のグローバル変数（リアルタイム・オーディオバッファはここに確保） */
  .bss :
  {
    . = ALIGN(4);
    *(.bss)
    *(.bss*)
    . = ALIGN(4);
  } >SRAM1
}
