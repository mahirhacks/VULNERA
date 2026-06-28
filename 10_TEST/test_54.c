/*
 * test_54.c
 *
 * Label: SAFE — pre-2019 era sample (sscanf width limited)
 * Inspired by: width-limited sscanf (pre-2019)
 * CWE: none
 */
#include <stdio.h>

#define TOKEN_BUF 16

int parse_token(const char *line, char *token)
{
    if (line == NULL || token == NULL)
        return -1;
    if (sscanf(line, "TOKEN=%15s", token) != 1)
        return -1;
    return 0;
}
