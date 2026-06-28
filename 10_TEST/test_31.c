/*
 * test_31.c
 *
 * Label: VULNERABLE — pre-2019 era sample (strcpy stack overflow)
 * Inspired by: CVE-2012-0056 style (2012)
 * CWE: CWE-120
 */
#include <string.h>

#define USERNAME_MAX 32

int store_username(char *dst, const char *src)
{
    if (dst == NULL || src == NULL)
        return -1;
    /* BUG: unbounded strcpy into fixed stack buffer */
    strcpy(dst, src);
    return 0;
}
