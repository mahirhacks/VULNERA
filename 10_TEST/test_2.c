/*
 * test_2.c
 *
 * Inspired by: CVE-2014-6271 (GNU Bash "Shellshock", disclosed Sep 2014)
 * CWE: CWE-78 (OS Command Injection)
 *
 * Bug: environment values beginning with "() {" are treated as function
 * definitions; trailing attacker commands after the closing brace are executed
 * when the variable is expanded through system().
 */

#include <stdlib.h>
#include <string.h>

int bash_exported_function_check(const char *name, const char *value)
{
    const char *tail;

    if (name == NULL || value == NULL)
        return -1;

  /*
   * BUG: only validates prefix looks like a function definition; does not
   * strip or reject shell metacharacters after the closing '}'.
   */
    if (strncmp(value, "() {", 4) != 0)
        return 0;

    tail = strchr(value, '}');
    if (tail == NULL)
        return -1;

    /* Attacker payload after '}' runs when passed to shell */
    system(tail + 1);
    return 0;
}
