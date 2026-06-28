/*
 * test_19.c
 *
 * Inspired by: CVE-2024-28085 ("WallEscape", util-linux wall, Mar 2024)
 * CWE: CWE-74 (Injection) / CWE-116 (Improper Encoding of Output)
 *
 * Bug: user-supplied message is written to other users' terminals without
 * filtering ANSI escape sequences, allowing fake sudo prompts or arbitrary
 * terminal manipulation.
 */

#include <stdio.h>
#include <string.h>

#define WALL_MAX 2048

int wall_broadcast(const char *sender, const char *message,
                   FILE **terminals, int term_count)
{
    char line[WALL_MAX];
    int i;

    if (sender == NULL || message == NULL || terminals == NULL)
        return -1;

    /*
     * BUG: no sanitisation of 'message'. Escape sequences such as
     * \033[8m (hide text) or \033]2; (set window title) pass through
     * directly, enabling social-engineering attacks (fake sudo prompt).
     */
    snprintf(line, sizeof(line), "\r\n"
             "Broadcast message from %s:\r\n"
             "%s\r\n",
             sender, message);

    for (i = 0; i < term_count; i++) {
        if (terminals[i] != NULL) {
            fputs(line, terminals[i]);
            fflush(terminals[i]);
        }
    }
    return 0;
}
