/*
 * test_45.c
 *
 * Label: VULNERABLE — pre-2019 era sample (null deref lookup)
 * Inspired by: CVE-2009-1897 class (2009)
 * CWE: CWE-476
 */
typedef struct {
    int value;
} node_t;

int node_value(node_t *n)
{
    /* BUG: no NULL check before dereference */
    return n->value;
}
