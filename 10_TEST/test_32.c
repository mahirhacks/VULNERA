/*
 * test_32.c
 *
 * Label: SAFE — pre-2019 era sample (strncpy bounded copy)
 * Inspired by: CVE-2012-0056 fix pattern (2012)
 * CWE: CWE-120 (mitigated)
 */
#include <string.h>

#define USERNAME_MAX 32

int store_username(char *dst, const char *src)
{
    if (dst == NULL || src == NULL)
        return -1;
    strncpy(dst, src, USERNAME_MAX - 1);
    dst[USERNAME_MAX - 1] = '\0';
    return 0;
}
