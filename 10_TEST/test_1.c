/*
 * test_1.c
 *
 * Inspired by: CVE-2014-0160 (OpenSSL "Heartbleed", disclosed Apr 2014)
 * CWE: CWE-125 (Out-of-bounds Read)
 *
 * Bug: TLS heartbeat handler trusts attacker-controlled payload_length and
 * memcpy's that many bytes from the request into the response without checking
 * against the actual allocated heartbeat buffer.
 */

#include <stdlib.h>
#include <string.h>

#define HEARTBEAT_BUF_SIZE 16

typedef struct {
    unsigned char buf[HEARTBEAT_BUF_SIZE];
    unsigned short payload_len;
} tls_heartbeat_t;

int tls_process_heartbeat(tls_heartbeat_t *hb, unsigned char *out, unsigned short *out_len)
{
    if (hb == NULL || out == NULL || out_len == NULL)
        return -1;

    /*
     * BUG: hb->payload_len comes from the wire (attacker-controlled).
     * memcpy reads past hb->buf when payload_len > HEARTBEAT_BUF_SIZE.
     */
    memcpy(out, hb->buf, hb->payload_len);
    *out_len = hb->payload_len;
    return 0;
}
