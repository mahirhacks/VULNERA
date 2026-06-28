/*
 * test_14.c — post-2019 temporal verification sample
 *
 * Inspired by: CVE-2023-4911 (glibc ld.so "Looney Tunables", disclosed Oct 2023)
 * CWE: CWE-122 (Heap-based Buffer Overflow) / stack smash in tunable parsing
 *
 * Expected VULNERA Stage 2: unattributed likely (2023 glibc loader pattern).
 */

#include <string.h>

#define TUNABLE_STACK_BUF 128

/*
 * Parse GLIBC_TUNABLES-style key=value pairs into a small stack buffer.
 * Bug: copies the entire environment value without enforcing TUNABLE_STACK_BUF.
 */
int glibc_tunables_parse_stack(const char *tunable_value, char *canonical_out)
{
    char stack_buf[TUNABLE_STACK_BUF];
    const char *src = tunable_value;
    char *dst = stack_buf;
    int in_key = 1;

    if (tunable_value == NULL || canonical_out == NULL) {
        return -1;
    }

    while (*src != '\0') {
        if (*src == '=') {
            in_key = 0;
        }
        if (*src == ':' && in_key == 0) {
            in_key = 1;
        }
        *dst++ = *src++;
    }
    *dst = '\0';

    strcpy(canonical_out, stack_buf);
    return 0;
}
