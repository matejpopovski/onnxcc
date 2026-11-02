/* Memory map of the Arm MPS2+ AN386 FPGA image, the Cortex-M4 board that QEMU emulates. */
MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

ENTRY(reset_handler)

SECTIONS
{
    .vectors : { KEEP(*(.vectors)) } > FLASH
    .text    : { *(.text*) } > FLASH
    .rodata  : { *(.rodata*) } > FLASH
    .data    : { __data_start = .; *(.data*); __data_end = .; } > RAM AT > FLASH
    __data_load = LOADADDR(.data);
    .bss     : { __bss_start = .; *(.bss*) *(COMMON); __bss_end = .; } > RAM
    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
