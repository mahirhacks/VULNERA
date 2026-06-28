/*
 * test_57.c
 *
 * Label: VULNERABLE — pre-2019 era sample (memcpy user length)
 * Inspired by: CVE-2017-5715 cousin memcpy (2017)
 * CWE: CWE-125 / CWE-787
 */
#include <string.h>

#define PKT_BUF 48

int copy_payload(unsigned char *dst, const unsigned char *src, unsigned short len)
{
    if (dst == NULL || src == NULL)
        return -1;
    /* BUG: trusts wire length without cap */
    memcpy(dst, src, len);
    return 0;
}
