/*
 * test_60.c
 *
 * Label: SAFE — pre-2019 era sample (divide by zero guard)
 * Inspired by: zero-safe average (pre-2019)
 * CWE: none
 */
int average_count(int total, int n)
{
    if (n == 0)
        return -1;
    return total / n;
}
