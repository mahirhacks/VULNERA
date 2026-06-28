/*
 * test_10.c
 *
 * Inspired by: CVE-2014-0160 (OpenSSL Heartbleed — write side, Apr 2014)
 * CWE: CWE-787 (Out-of-bounds Write) / CWE-200 (Info leak)
 *
 * Bug: heartbeat response echoes request payload; when payload_length exceeds
 * the request buffer, memcpy reads heap memory into the outbound packet.
 * Companion to pre2019_01 (read side into response buffer).
 */

#include <string.h>

#define HB_REQUEST_MAX 16

typedef struct {
    unsigned char request[HB_REQUEST_MAX];
    unsigned char response[64];
    unsigned short payload_len;
} hb_state_t;

int tls_build_heartbeat_response(hb_state_t *st, unsigned short *sent_len)
{
    if (st == NULL || sent_len == NULL)
        return -1;

    /*
     * BUG: payload_len not capped to HB_REQUEST_MAX before memcpy into
     * response — leaks heap/stack bytes past request[] (Heartbleed).
     */
    memcpy(st->response, st->request, st->payload_len);
    *sent_len = st->payload_len;
    return 0;
}
