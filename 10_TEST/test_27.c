/*
 * test_27.c
 *
 * Inspired by: CVE-2023-42115 (Exim SMTP header buffer truncation, Sep 2023)
 * CWE: CWE-120 (Buffer Copy without Checking Size of Input)
 *
 * Bug: header line accumulated in a fixed buffer; 'remain' computed as signed
 * int but compared to size_t append length — negative remain allows strcat.
 */

#include <string.h>

#define HEADER_BUF_SIZE 256

typedef struct {
    char  line[HEADER_BUF_SIZE];
    int   used;
} smtp_header_t;

int smtp_header_append(smtp_header_t *hdr, const char *chunk, size_t chunk_len)
{
    int remain;

    if (hdr == NULL || chunk == NULL || chunk_len == 0)
        return -1;

    remain = HEADER_BUF_SIZE - hdr->used - 1;

    /*
     * BUG: if hdr->used already exceeds HEADER_BUF_SIZE-1 (corrupt state or
     * prior bug), remain goes negative. Casting remain to size_t in comparison
     * makes (size_t)remain huge, so chunk_len < remain passes.
     */
    if (chunk_len < (size_t)remain) {
        memcpy(hdr->line + hdr->used, chunk, chunk_len);
        hdr->used += (int)chunk_len;
        hdr->line[hdr->used] = '\0';
        return 0;
    }
    return -1;
}
