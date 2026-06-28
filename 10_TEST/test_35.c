/*
 * test_35.c
 *
 * Label: VULNERABLE — pre-2019 era sample (sprintf path build)
 * Inspired by: CVE-2011-1920 style (2011)
 * CWE: CWE-120
 */
#include <stdio.h>

#define PATH_MAX_LOCAL 64

int build_config_path(char *out, const char *name)
{
    if (out == NULL || name == NULL)
        return -1;
    /* BUG: sprintf without bound check on out[] */
    sprintf(out, "/etc/app/%s.conf", name);
    return 0;
}
