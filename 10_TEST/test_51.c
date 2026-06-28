/*
 * test_51.c
 *
 * Label: VULNERABLE — pre-2019 era sample (realloc lost pointer)
 * Inspired by: CVE-2011-1083 class (2011)
 * CWE: CWE-401 / CWE-415
 */
#include <stdlib.h>
#include <string.h>

int grow_buffer(char **buf, size_t *len, size_t need)
{
    if (buf == NULL || len == NULL)
        return -1;
    if (need <= *len)
        return 0;
    /* BUG: old pointer lost if realloc moves block */
    realloc(*buf, need);
    memset(*buf + *len, 0, need - *len);
    *len = need;
    return 0;
}
