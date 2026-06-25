cmake_minimum_required(VERSION 3.22)

# クロスコンパイルの設定（ARM用コンパイラを明示）
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
# ----------------------------

project(hrtf_dsp C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(MCU_FLAGS
    -mcpu=cortex-m7
    -mthumb
    -mfpu=fpv5-d16
    -mfloat-abi=hard
)

# H723ZG (Cortex-M7) のハードウェア性能をフルに引き出すコンパイルオプション
# ここでハードウェアFPU（浮動小数点演算器）を有効化しています
add_compile_options(
    -mcpu=cortex-m7
    -mthumb
    -mfpu=fpv5-d16
    -mfloat-abi=hard
    -O3   # 空間オーディオの計算を最速にするための最大最適化
    -Wall
    -fdata-sections
    -ffunction-sections
)

# 実行ファイルの定義
add_executable(${PROJECT_NAME}
    src/main.c
)


# リンカスクリプトの指定（後ほどファイルを作成します）
set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/stm32h723zg.ld)

target_link_options(${PROJECT_NAME} PRIVATE
    ${MCU_FLAGS}
    -T${LINKER_SCRIPT}
    -Wl,--gc-sections
    --specs=nosys.specs
)

add_link_options(
    -mcpu=cortex-m7
    -mthumb
    -mfpu=fpv5-d16
    -mfloat-abi=hard
    -T${LINKER_SCRIPT}
    -Wl,--gc-sections
    --specs=nano.specs
)
