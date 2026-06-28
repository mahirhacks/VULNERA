/*
 * test_15.c — post-2019 temporal verification sample
 *
 * Inspired by: CVE-2024-1086 (Linux nf_tables double-free, disclosed Jan 2024)
 * CWE: CWE-415 (Double Free)
 *
 * Expected VULNERA Stage 2: unattributed / potential zero-day vs pre-2019 KB.
 */

#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned int key;
    void *payload;
    size_t payload_len;
} nft_set_elem_t;

/*
 * Remove two set elements that may alias the same underlying object.
 * Bug: when elem_a and elem_b refer to the same allocation, both are freed.
 */
int nft_set_flush_pair(nft_set_elem_t *elem_a, nft_set_elem_t *elem_b)
{
    if (elem_a == NULL || elem_b == NULL) {
        return -1;
    }

    if (elem_a->payload != NULL) {
        memset(elem_a->payload, 0, elem_a->payload_len);
        free(elem_a->payload);
        elem_a->payload = NULL;
    }

    if (elem_b->payload != NULL) {
        memset(elem_b->payload, 0, elem_b->payload_len);
        free(elem_b->payload);
        elem_b->payload = NULL;
    }

    return 0;
}

int nft_set_elem_attach(nft_set_elem_t *primary, nft_set_elem_t *alias, const void *data, size_t len)
{
    void *block;

    if (primary == NULL || alias == NULL || data == NULL || len == 0) {
        return -1;
    }

    block = malloc(len);
    if (block == NULL) {
        return -1;
    }
    memcpy(block, data, len);

    primary->payload = block;
    primary->payload_len = len;
    alias->payload = block;
    alias->payload_len = len;
    return 0;
}
