/*
 * test_56.c
 *
 * Label: SAFE — pre-2019 era sample (strncat accumulate)
 * Inspired by: bounded strcat (pre-2019)
 * CWE: none
 */
#include <string.h>

#define MSG_MAX 64

int append_fragment(char *msg, const char *frag)
{
    size_t used;
    if (msg == NULL || frag == NULL)
        return -1;
    used = strnlen(msg, MSG_MAX);
    if (used >= MSG_MAX - 1)
        return -1;
    strncat(msg, frag, MSG_MAX - 1 - used);
    msg[MSG_MAX - 1] = '\0';
    return 0;
}
