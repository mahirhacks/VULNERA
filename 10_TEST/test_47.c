/*
 * test_47.c
 *
 * Label: VULNERABLE — pre-2019 era sample (off by one loop)
 * Inspired by: CVE-2008-1517 class (2008)
 * CWE: CWE-787
 */
#include <string.h>

#define TAG_MAX 8

int copy_tags(char dst[][TAG_MAX], const char *src[], int count)
{
    int i;
    if (dst == NULL || src == NULL || count < 0)
        return -1;
    /* BUG: i <= count allows write past dst[count-1] */
    for (i = 0; i <= count; i++) {
        if (src[i] == NULL)
            return -1;
        strncpy(dst[i], src[i], TAG_MAX - 1);
        dst[i][TAG_MAX - 1] = '\0';
    }
    return 0;
}
