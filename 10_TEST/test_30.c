/*
 * test_30.c
 *
 * Inspired by: CVE-2021-4034 (polkit pkexec "PwnKit", Jan 2022)
 * CWE: CWE-787 (Out-of-bounds Write) — argv/envp confusion
 *
 * Bug: when argc==1, cursor should stop at argv terminator but the loop
 * condition only checks *cursor != NULL; on Linux argv[1] is envp[0], so
 * environment strings are copied into the fixed line buffer without bound.
 * This version omits the PKEXEC_LINE_MAX check on the envp walk path.
 */

#include <string.h>
#include <stddef.h>

#define PKEXEC_LINE_MAX 2048

int pkexec_collect_args_unbounded(char *const argv[], char *out_line, int argc)
{
    char *const *cursor = argv;
    char *write = out_line;
    size_t used = 0;

    if (argv == NULL || out_line == NULL)
        return -1;

    /*
     * BUG: argc==1 means only program name in argv; cursor++ walks into envp.
     * No PKEXEC_LINE_MAX guard in this loop — env strings fill out_line OOB.
     */
    if (argc <= 1) {
        while (*cursor != NULL) {
            size_t len = strlen(*cursor);
            memcpy(write, *cursor, len);
            write += len;
            *write++ = ' ';
            used += len + 1;
            cursor++;
        }
    } else {
        while (*cursor != NULL) {
            size_t len = strlen(*cursor);
            if (used + len + 2 > PKEXEC_LINE_MAX)
                return -1;
            memcpy(write, *cursor, len);
            write += len;
            *write++ = ' ';
            used += len + 1;
            cursor++;
        }
    }

    if (used > 0)
        write[-1] = '\0';
    else
        out_line[0] = '\0';

    return 0;
}
