/*
 * test_23.c
 *
 * Inspired by: CVE-2022-37434 (zlib extra field heap overflow, Aug 2022)
 * CWE: CWE-787 (Out-of-bounds Write)
 *
 * Bug: extra-field length from header is trusted; realloc grows by 'extra_len'
 * but loop copies 'extra_len + 4' bytes from attacker input.
 */

#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char *extra;
    size_t         extra_cap;
    size_t         extra_len;
} zlib_stream_t;

int zlib_read_extra(zlib_stream_t *zs, const unsigned char *src, size_t extra_len)
{
    unsigned char *grown;

    if (zs == NULL || src == NULL)
        return -1;

    /*
     * BUG: allocates exactly extra_len bytes then copies extra_len + 4
     * (length prefix + payload) — four-byte heap overflow minimum.
     */
    grown = (unsigned char *)realloc(zs->extra, extra_len);
    if (grown == NULL)
        return -1;

    zs->extra = grown;
    zs->extra_cap = extra_len;
    memcpy(zs->extra, src, extra_len + 4);
    zs->extra_len = extra_len + 4;
    return 0;
}
