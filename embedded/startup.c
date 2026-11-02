/* Bare-metal runtime for the QEMU MPS2 AN386 Cortex-M4. It exists to run a generated model under
 * emulation. The generated file itself needs none of this. */
#include <stdint.h>

extern uint32_t __data_start, __data_end, __data_load, __bss_start, __bss_end, __stack_top;
extern int main(void);

static void semihost(uint32_t operation, const void *argument)
{
    register uint32_t r0 __asm__("r0") = operation;
    register const void *r1 __asm__("r1") = argument;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
}

void board_print(const char *text)
{
    semihost(0x04u, text);
}

void board_exit(void)
{
    /* On AArch32 the reason code travels in r1 itself. ADP_Stopped_ApplicationExit means success. */
    semihost(0x18u, (const void *)0x20026u);
    for (;;) {
    }
}

void reset_handler(void)
{
    volatile uint32_t *const cpacr = (volatile uint32_t *)0xE000ED88u;
    const uint32_t *source = &__data_load;
    uint32_t *target = &__data_start;
    while (target < &__data_end) {
        *target++ = *source++;
    }
    target = &__bss_start;
    while (target < &__bss_end) {
        *target++ = 0u;
    }
    *cpacr |= 0xFu << 20; /* grant access to CP10 and CP11, which is the FPU */
    __asm__ volatile("dsb\n\tisb");
    (void)main();
    board_exit();
}

static void fault_handler(void)
{
    board_print("FAULT\n");
    board_exit();
}

__attribute__((section(".vectors"), used)) static void (*const vectors[16])(void) = {
    (void (*)(void))(&__stack_top),
    reset_handler,
    fault_handler,
    fault_handler,
    fault_handler,
    fault_handler,
    fault_handler,
};
