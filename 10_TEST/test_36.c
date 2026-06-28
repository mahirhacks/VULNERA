/*
 * test_36.c
 *
 * Label: SAFE — pre-2019 era sample (snprintf path build)
 * Inspired by: bounded path build (pre-2019)
 * CWE: none
 */
#include <stdio.h>

#define PATH_MAX_LOCAL 64

int build_config_path(char *out, const char *name)
{
    if (out == NULL || name == NULL)
        return -1;
    if (snprintf(out, PATH_MAX_LOCAL, "/etc/app/%s.conf", name) >= PATH_MAX_LOCAL)
        return -1;
    return 0;
}
