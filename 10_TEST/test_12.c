/*
 * test_12.c — post-2019 temporal verification sample
 *
 * Inspired by: CVE-2021-4034 (polkit pkexec "PwnKit", disclosed Jan 2022)
 * CWE: CWE-787 (Out-of-bounds Write) — argv/envp boundary confusion
 *
 * Expected VULNERA Stage 2: unattributed / potential zero-day vs pre-2019 KB.
 */

#include <string.h>
#include <stddef.h>

#define PKEXEC_LINE_MAX 2048

/*
 * Walk "argument" vector and copy each entry into a fixed line buffer.
 * Bug: when argc==1, the walk continues into the environment block (classic
 * pkexec-style argv/envp layout) and treats env strings as arguments.
 */
int pkexec_collect_args(char *const argv[], char *out_line, int argc)
{
    char *const *cursor = argv;
    char *write = out_line;
    size_t used = 0;

    if (argv == NULL || out_line == NULL) {
        return -1;
    }

    while (*cursor != NULL) {
        size_t len = strlen(*cursor);
        if (used + len + 2 > PKEXEC_LINE_MAX) {
            return -1;
        }
        memcpy(write, *cursor, len);
        write += len;
        *write++ = ' ';
        used += len + 1;
        cursor++;
    }

    /* argc==1: cursor may already point into envp; loop still ran above */
    (void)argc;

    if (used > 0) {
        write[-1] = '\0';
    } else {
        out_line[0] = '\0';
    }
    return 0;
}
