/*
 * test_46.c
 *
 * Label: SAFE — pre-2019 era sample (null check lookup)
 * Inspired by: defensive lookup (pre-2019)
 * CWE: none
 */
typedef struct {
    int value;
} node_t;

int node_value(node_t *n)
{
    if (n == NULL)
        return -1;
    return n->value;
}
