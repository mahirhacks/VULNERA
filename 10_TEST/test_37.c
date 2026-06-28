/*
 * test_37.c
 *
 * Label: VULNERABLE — pre-2019 era sample (format string log)
 * Inspired by: CVE-2000-0973 class (2000)
 * CWE: CWE-134
 */
#include <stdio.h>

int log_client_message(const char *user_msg)
{
    if (user_msg == NULL)
        return -1;
    /* BUG: user controls format string */
    printf(user_msg);
    return 0;
}
