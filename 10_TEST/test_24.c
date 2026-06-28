/*
 * test_24.c
 *
 * Inspired by: CVE-2021-3156 (sudo Baron Samedit, Jan 2021)
 * CWE: CWE-787 (Out-of-bounds Write) — heap
 *
 * Bug: heap buffer sized to raw input length, but backslash escapes emit TWO
 * output bytes per source character. This version intentionally reproduces
 * the miscount (unlike post2019_01 which paired i++/j++ correctly).
 */

#include <stdlib.h>
#include <string.h>

int sudoers_unescape_heap_v2(const char *user_in, char **out, size_t *out_len)
{
    size_t in_len;
    size_t i = 0;
    size_t j = 0;
    char *heap_buf;

    if (user_in == NULL || out == NULL || out_len == NULL)
        return -1;

    in_len = strlen(user_in);
    heap_buf = (char *)malloc(in_len + 1);
    if (heap_buf == NULL)
        return -1;

    while (user_in[i] != '\0') {
        if (user_in[i] == '\\' && user_in[i + 1] != '\0') {
            /*
             * BUG: two bytes written, only one source char consumed for sizing.
             * Second i++ happens but capacity was in_len+1 total, not 2*escapes.
             */
            heap_buf[j++] = '\\';
            heap_buf[j++] = user_in[++i];
        } else {
            heap_buf[j++] = user_in[i];
        }
        i++;
    }
    heap_buf[j] = '\0';

    *out = heap_buf;
    *out_len = j;
    return 0;
}
