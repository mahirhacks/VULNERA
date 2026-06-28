/*
 * test_50.c
 *
 * Label: SAFE — pre-2019 era sample (snprintf correct limit)
 * Inspired by: correct snprintf bound (pre-2019)
 * CWE: none
 */
#include <stdio.h>

#define HDR_BUF 32

int pack_header(char *out, const char *payload)
{
    if (out == NULL || payload == NULL)
        return -1;
    if (snprintf(out, HDR_BUF, "HDR:%s", payload) >= HDR_BUF)
        return -1;
    return 0;
}
