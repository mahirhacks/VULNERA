/*
 * test_33.c
 *
 * Label: VULNERABLE — pre-2019 era sample (gets line reader)
 * Inspired by: CVE-2008-0595 class (2008)
 * CWE: CWE-676 / CWE-120
 */
#include <stdio.h>

#define LINE_BUF 128

int read_user_line(char *buf)
{
    if (buf == NULL)
        return -1;
    /* BUG: deprecated gets() with no length limit */
    if (gets(buf) == NULL)
        return -1;
    return 0;
}
