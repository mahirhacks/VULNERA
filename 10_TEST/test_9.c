/*
 * test_9.c
 *
 * Inspired by: CVE-2015-1538 (Android Stagefright, disclosed Jul 2015)
 * CWE: CWE-122 (Heap-based Buffer Overflow)
 *
 * Bug: MP4 'stsc' sample-to-chunk table allocates based on entry_count from
 * the file but reads 8 * entry_count bytes without an upper bound check.
 */

#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned int first_chunk;
    unsigned int samples_per_chunk;
} stsc_entry_t;

int stagefright_parse_stsc(const unsigned char *table, unsigned int entry_count,
                           stsc_entry_t **out, unsigned int *out_count)
{
    stsc_entry_t *entries;
    size_t need_bytes;

    if (table == NULL || out == NULL || out_count == NULL)
        return -1;

    /*
     * BUG: entry_count attacker-controlled; malloc small then memcpy huge.
     * Classic integer/size mismatch — allocate count * sizeof but copy 8*count.
     */
    entries = (stsc_entry_t *)malloc(entry_count * sizeof(stsc_entry_t));
    if (entries == NULL)
        return -1;

    need_bytes = (size_t)entry_count * 8U;
    memcpy(entries, table, need_bytes);

    *out = entries;
    *out_count = entry_count;
    return 0;
}
