/*
 * test_48.c
 *
 * Label: SAFE — pre-2019 era sample (bounded tag copy)
 * Inspired by: correct loop bound (pre-2019)
 * CWE: none
 */
#include <string.h>

#define TAG_MAX 8

int copy_tags(char dst[][TAG_MAX], const char *src[], int count)
{
    int i;
    if (dst == NULL || src == NULL || count <= 0)
        return -1;
    for (i = 0; i < count; i++) {
        if (src[i] == NULL)
            return -1;
        strncpy(dst[i], src[i], TAG_MAX - 1);
        dst[i][TAG_MAX - 1] = '\0';
    }
    return 0;
}
