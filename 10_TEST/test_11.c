/*
 * test_11.c — post-2019 temporal verification sample
 *
 * Inspired by: CVE-2021-3156 (sudo "Baron Samedit", disclosed Jan 2021)
 * CWE: CWE-787 (Out-of-bounds Write) / heap overflow via escape expansion
 *
 * Expected VULNERA Stage 2: high exploit risk possible; CWE match may be
 * UNATTRIBUTED or weak (pattern not dominant in pre-2019 KB).
 */

#include <stdlib.h>
#include <string.h>

/*
 * Expand backslash escapes into a heap buffer sized for the *raw* input length.
 * Bug: each escape writes two bytes while capacity assumes one byte per source char.
 */
int sudoers_unescape_heap(const char *user_in, char **out, size_t *out_len)
{
    size_t in_len;
    size_t i = 0;
    size_t j = 0;
    char *heap_buf;

    if (user_in == NULL || out == NULL || out_len == NULL) {
        return -1;
    }

    in_len = strlen(user_in);
    heap_buf = (char *)malloc(in_len + 1);
    if (heap_buf == NULL) {
        return -1;
    }

    while (user_in[i] != '\0') {
        if (user_in[i] == '\\' && user_in[i + 1] != '\0') {
            heap_buf[j++] = user_in[i++];
            heap_buf[j++] = user_in[i++];
        } else {
            heap_buf[j++] = user_in[i++];
        }
    }
    heap_buf[j] = '\0';

    *out = heap_buf;
    *out_len = j;
    return 0;
}
