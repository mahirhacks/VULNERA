/*
 * test_58.c
 *
 * Label: SAFE — pre-2019 era sample (memcpy capped length)
 * Inspired by: capped memcpy (pre-2019)
 * CWE: none
 */
#include <string.h>

#define PKT_BUF 48

int copy_payload(unsigned char *dst, const unsigned char *src, unsigned short len)
{
    size_t n;
    if (dst == NULL || src == NULL)
        return -1;
    n = (len > PKT_BUF) ? PKT_BUF : len;
    memcpy(dst, src, n);
    return (len > PKT_BUF) ? -1 : 0;
}
