/*
 * test_44.c
 *
 * Label: SAFE — pre-2019 era sample (single free cache)
 * Inspired by: idempotent release (pre-2019)
 * CWE: none
 */
#include <stdlib.h>

typedef struct {
    int id;
} cache_entry_t;

void cache_release(cache_entry_t **e)
{
    if (e == NULL || *e == NULL)
        return;
    free(*e);
    *e = NULL;
}
