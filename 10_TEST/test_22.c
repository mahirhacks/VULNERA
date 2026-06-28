/*
 * test_22.c
 *
 * Inspired by: CVE-2023-38545 (curl SOCKS5 heap buffer overflow, Oct 2023)
 * CWE: CWE-122 (Heap-based Buffer Overflow)
 *
 * Bug: SOCKS5 target hostname length is stored in uint8_t but copied from a
 * size_t without capping; attacker length 0xFF+ causes malloc(256) then
 * memcpy of the full attacker string.
 */

#include <stdlib.h>
#include <string.h>

#define SOCKS5_NAME_MAX 255

typedef struct {
    char *hostname;
    size_t host_len;
} socks5_target_t;

int socks5_set_target(socks5_target_t *tgt, const char *host, size_t host_len)
{
    unsigned char name_len;
    char *heap_name;

    if (tgt == NULL || host == NULL || host_len == 0)
        return -1;

    /*
     * BUG: truncates length to uint8_t for the wire format but uses the
     * original host_len for allocation sizing vs memcpy size mismatch when
     * host_len > 255 — memcpy writes past 256-byte allocation.
     */
    name_len = (unsigned char)host_len;
    heap_name = (char *)malloc((size_t)name_len + 1);
    if (heap_name == NULL)
        return -1;

    memcpy(heap_name, host, host_len);   /* host_len may be >> 255 */
    heap_name[name_len] = '\0';

    tgt->hostname = heap_name;
    tgt->host_len = host_len;
    return 0;
}
