/*
 * test_53.c
 *
 * Label: VULNERABLE — pre-2019 era sample (sscanf unbounded)
 * Inspired by: CVE-2006-0450 sscanf (2006)
 * CWE: CWE-120
 */
#include <stdio.h>
#include <string.h>

#define TOKEN_BUF 16

int parse_token(const char *line, char *token)
{
    if (line == NULL || token == NULL)
        return -1;
    /* BUG: %s with no field width */
    if (sscanf(line, "TOKEN=%s", token) != 1)
        return -1;
    return 0;
}
