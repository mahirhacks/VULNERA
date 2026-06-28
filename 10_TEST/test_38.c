/*
 * test_38.c
 *
 * Label: SAFE — pre-2019 era sample (format string literal)
 * Inspired by: safe logging (pre-2019)
 * CWE: none
 */
#include <stdio.h>

int log_client_message(const char *user_msg)
{
    if (user_msg == NULL)
        return -1;
    printf("%s\n", user_msg);
    return 0;
}
