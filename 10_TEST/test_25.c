/*
 * test_25.c
 *
 * Inspired by: CVE-2022-23222 (Linux BPF map element out-of-bounds, Jan 2022)
 * CWE: CWE-787 (Out-of-bounds Write)
 *
 * Bug: index is bounds-checked against 'max_entries' but multiplied by
 * 'value_size' in 32-bit arithmetic before adding to base — product wraps
 * and store lands inside the map object instead of past the end (or OOB).
 */

#include <string.h>

#define BPF_MAP_MAX_ENTRIES 1024
#define BPF_MAP_VALUE_SIZE  256

typedef struct {
    unsigned int max_entries;
    unsigned int value_size;
    char         storage[BPF_MAP_MAX_ENTRIES * BPF_MAP_VALUE_SIZE];
} bpf_array_map_t;

int bpf_array_update_elem(bpf_array_map_t *map, unsigned int index,
                          const void *value, unsigned int value_size)
{
    unsigned int offset;

    if (map == NULL || value == NULL)
        return -1;

    if (index >= map->max_entries)
        return -1;

    if (value_size > map->value_size)
        return -1;

    /*
     * BUG: 32-bit multiply can wrap: index * value_size overflows,
     * offset points inside map metadata region.
     */
    offset = (unsigned int)(index * map->value_size);
    memcpy(map->storage + offset, value, value_size);
    return 0;
}
