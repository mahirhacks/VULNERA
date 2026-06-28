/*
 * test_49.c
 *
 * Label: VULNERABLE — pre-2019 era sample (snprintf wrong limit)
 * Inspired by: CVE-2014-0160 cousin (2014)
 * CWE: CWE-787
 */
#include <stdio.h>
#include <string.h>

#define HDR_BUF 32

int pack_header(char *out, const char *payload)
{
    if (out == NULL || payload == NULL)
        return -1;
    /* BUG: snprintf size is strlen(payload), not sizeof(out) */
    snprintf(out, strlen(payload) + 1, "HDR:%s", payload);
    return 0;
}
