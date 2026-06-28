/*
 * test_41.c
 *
 * Label: VULNERABLE — pre-2019 era sample (use after free session)
 * Inspired by: CVE-2013-2094 class (2013)
 * CWE: CWE-416
 */
#include <stdlib.h>
#include <string.h>

typedef struct {
    char token[16];
} session_t;

int validate_session(session_t *s)
{
    if (s == NULL)
        return -1;
    free(s);
    /* BUG: reads freed object */
    return (s->token[0] != '\0') ? 0 : -1;
}
