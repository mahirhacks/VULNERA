/*
 * test_43.c
 *
 * Label: VULNERABLE — pre-2019 era sample (double free cache)
 * Inspired by: CVE-2006-0450 class (2006)
 * CWE: CWE-415
 */
#include <stdlib.h>

typedef struct {
    int id;
} cache_entry_t;

void cache_release(cache_entry_t *e)
{
    if (e == NULL)
        return;
    free(e);
    /* BUG: second free on same pointer */
    free(e);
}
