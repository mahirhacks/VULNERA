/*
 * test_6.c
 *
 * Inspired by: CVE-2013-2028 (nginx chunked HTTP parser, disclosed May 2013)
 * CWE: CWE-190 (Integer Overflow) -> CWE-122 (Heap Overflow)
 *
 * Bug: chunked body size accumulated in size_t but passed through signed int
 * for guard check; large chunk sizes wrap negative and bypass limit.
 */

#include <stdlib.h>
#include <string.h>

#define CHUNK_LIMIT 65536

typedef struct {
    char  *body;
    size_t body_cap;
    size_t body_len;
} http_request_t;

int nginx_chunked_append(http_request_t *req, const char *chunk, size_t chunk_len)
{
    int remaining;
    char *grown;

    if (req == NULL || chunk == NULL || chunk_len == 0)
        return -1;

    remaining = (int)(req->body_cap - req->body_len);

    /*
     * BUG: comparing size_t chunk_len to signed remaining — when remaining
     * is negative, (size_t)remaining is huge and check passes.
     */
    if (chunk_len > (size_t)remaining)
        return -1;

    grown = (char *)realloc(req->body, req->body_len + chunk_len + 1);
    if (grown == NULL)
        return -1;

    req->body = grown;
    memcpy(req->body + req->body_len, chunk, chunk_len);
    req->body_len += chunk_len;
    req->body[req->body_len] = '\0';
    return 0;
}
