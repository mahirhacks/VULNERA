/*
 * test_5.c
 *
 * Inspired by: CVE-2015-7547 (glibc getaddrinfo stack overflow, disclosed Feb 2016)
 * CWE: CWE-121 (Stack-based Buffer Overflow)
 *
 * Bug: DNS AAAA answer parser copies RDATA into a fixed stack buffer using
 * the length byte from the packet without capping against buffer size.
 */

#include <string.h>

#define ANSWER_STACK_MAX 48

int parse_dns_aaaa_rdata(const unsigned char *rdata, unsigned char rdlength,
                         char *canonical_out)
{
    char stack_buf[ANSWER_STACK_MAX];

    if (rdata == NULL || canonical_out == NULL)
        return -1;

    /*
     * BUG: rdlength is attacker-controlled (from DNS packet). memcpy copies
     * rdlength bytes even when rdlength > ANSWER_STACK_MAX.
     */
    memcpy(stack_buf, rdata, rdlength);
    stack_buf[rdlength] = '\0';

    strcpy(canonical_out, stack_buf);
    return 0;
}
