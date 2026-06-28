/*
 * test_34.c
 *
 * Label: SAFE — pre-2019 era sample (fgets line reader)
 * Inspired by: POSIX safe read (pre-2019)
 * CWE: none
 */
#include <stdio.h>

#define LINE_BUF 128

int read_user_line(char *buf)
{
    if (buf == NULL)
        return -1;
    if (fgets(buf, LINE_BUF, stdin) == NULL)
        return -1;
    return 0;
}
