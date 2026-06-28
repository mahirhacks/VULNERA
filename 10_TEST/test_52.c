/*
 * test_52.c
 *
 * Label: SAFE — pre-2019 era sample (realloc assign pointer)
 * Inspired by: safe realloc pattern (pre-2019)
 * CWE: none
 */
#include <stdlib.h>
#include <string.h>

int grow_buffer(char **buf, size_t *len, size_t need)
{
    void *tmp;
    if (buf == NULL || len == NULL || *buf == NULL)
        return -1;
    if (need <= *len)
        return 0;
    tmp = realloc(*buf, need);
    if (tmp == NULL)
        return -1;
    *buf = tmp;
    memset((char *)*buf + *len, 0, need - *len);
    *len = need;
    return 0;
}
