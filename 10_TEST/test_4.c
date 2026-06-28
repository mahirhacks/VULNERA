/*
 * test_4.c
 *
 * Inspired by: CVE-2017-7494 (Samba is_known_pipename RCE, disclosed May 2017)
 * CWE: CWE-22 (Path Traversal)
 *
 * Bug: client-supplied pipe name is concatenated into a shared-library path
 * without rejecting ".." or absolute path components.
 */

#include <stdio.h>
#include <string.h>

#define SMB_PIPE_ROOT "/var/lib/samba/pipes"

int samba_load_pipename_module(const char *pipename)
{
    char path[256];

    if (pipename == NULL || pipename[0] == '\0')
        return -1;

    /*
     * BUG: pipename like "/tmp/evil.so" or "../../../etc/passwd" is copied
     * directly into path used for dlopen-style loading.
     */
    snprintf(path, sizeof(path), "%s/%s", SMB_PIPE_ROOT, pipename);
    printf("loading module: %s\n", path);
    return 0;
}
