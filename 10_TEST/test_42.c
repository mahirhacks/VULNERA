/*
 * test_42.c
 *
 * Label: SAFE — pre-2019 era sample (free then null session)
 * Inspired by: UAF-safe teardown (pre-2019)
 * CWE: none
 */
#include <stdlib.h>
#include <string.h>

typedef struct {
    char token[16];
} session_t;

int validate_session(session_t *s)
{
    int ok;
    if (s == NULL)
        return -1;
    ok = (s->token[0] != '\0') ? 0 : -1;
    free(s);
    return ok;
}
