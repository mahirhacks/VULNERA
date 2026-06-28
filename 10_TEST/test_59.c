/*
 * test_59.c
 *
 * Label: VULNERABLE — pre-2019 era sample (divide by zero avg)
 * Inspired by: CVE-2009-1897 div (2009)
 * CWE: CWE-369
 */
int average_count(int total, int n)
{
    /* BUG: no check for n == 0 */
    return total / n;
}
