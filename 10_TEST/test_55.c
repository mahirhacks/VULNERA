/*
 * test_55.c
 *
 * Label: VULNERABLE — pre-2019 era sample (strcat accumulate)
 * Inspired by: CVE-2002-0575 class (2002)
 * CWE: CWE-120
 */
#include <string.h>

#define MSG_MAX 64

int append_fragment(char *msg, const char *frag)
{
    if (msg == NULL || frag == NULL)
        return -1;
    /* BUG: strcat without remaining space check */
    strcat(msg, frag);
    return 0;
}
