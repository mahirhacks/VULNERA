/*
 * test_26.c
 *
 * Inspired by: CVE-2023-43115 (Ghostscript PostScript stack/device overflow, Oct 2023)
 * CWE: CWE-121 (Stack-based Buffer Overflow)
 *
 * Bug: device output path copies decoded hex string into a fixed stack buffer
 * without checking decoded length (2x hex input size).
 */

#include <string.h>

#define DEVICE_LINE_MAX 128

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int gs_device_write_hex_line(const char *hex_line, char *device_out)
{
    char stack_buf[DEVICE_LINE_MAX];
    size_t i = 0;
    size_t out = 0;
    size_t hex_len;

    if (hex_line == NULL || device_out == NULL)
        return -1;

    hex_len = strlen(hex_line);

    /*
     * BUG: each pair of hex chars -> one byte; loop runs hex_len/2 times
     * with no cap against DEVICE_LINE_MAX. Long PostScript hex string
     * smashes the stack buffer.
     */
    while (i + 1 < hex_len) {
        int hi = hex_nibble(hex_line[i]);
        int lo = hex_nibble(hex_line[i + 1]);
        if (hi < 0 || lo < 0)
            return -1;
        stack_buf[out++] = (char)((hi << 4) | lo);
        i += 2;
    }
    stack_buf[out] = '\0';

    strcpy(device_out, stack_buf);
    return 0;
}
