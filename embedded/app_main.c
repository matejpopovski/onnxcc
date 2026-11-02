/* Runs a generated model on test vectors under QEMU and prints outputs and timer ticks.
 * vectors.h is written by experiments/embedded.py and defines:
 *   VECTOR_COUNT, MODEL_INPUT_ELEMENTS, MODEL_OUTPUT_ELEMENTS, model_input_t, model_output_t,
 *   vector_inputs[VECTOR_COUNT][MODEL_INPUT_ELEMENTS], and MODEL_RUN. */
#include <stdint.h>

#include "vectors.h"

extern void board_print(const char *text);

/* CMSDK APB timer 0 on MPS2. It counts down at the 25 MHz peripheral clock. */
#define TIMER_CTRL (*(volatile uint32_t *)0x40000000u)
#define TIMER_VALUE (*(volatile uint32_t *)0x40000004u)
#define TIMER_RELOAD (*(volatile uint32_t *)0x40000008u)

static model_output_t output[MODEL_OUTPUT_ELEMENTS];

static void print_hex(uint32_t value)
{
    static const char digits[] = "0123456789abcdef";
    char text[10];
    for (int i = 0; i < 8; ++i) {
        text[i] = digits[(value >> (28 - 4 * i)) & 0xFu];
    }
    text[8] = ' ';
    text[9] = '\0';
    board_print(text);
}

int main(void)
{
    TIMER_RELOAD = 0xFFFFFFFFu;
    TIMER_VALUE = 0xFFFFFFFFu;
    TIMER_CTRL = 1u;
    for (int v = 0; v < VECTOR_COUNT; ++v) {
        const uint32_t start = TIMER_VALUE;
        MODEL_RUN(vector_inputs[v], output);
        const uint32_t ticks = start - TIMER_VALUE;
        print_hex(ticks);
        for (int i = 0; i < MODEL_OUTPUT_ELEMENTS; ++i) {
            union {
                model_output_t value;
                uint32_t bits;
            } view;
            view.bits = 0u;
            view.value = output[i];
            print_hex(view.bits);
        }
        board_print("\n");
    }
    return 0;
}
