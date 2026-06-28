/*
 * test_3.c
 *
 * Inspired by: CVE-2016-5195 (Linux "Dirty COW", disclosed Oct 2016)
 * CWE: CWE-362 (Race Condition) -> CWE-787 (Write)
 *
 * Bug: private mapping write path checks PTE state, then writes without holding
 * the page lock across the check — concurrent madvise can COW the page between
 * check and store (TOCTOU).
 */

#include <string.h>

typedef struct {
    char   *page;
    int     writable;
    int     locked;
} cow_page_t;

int cow_follow_write(cow_page_t *pg, size_t offset, char value)
{
    if (pg == NULL || pg->page == NULL)
        return -1;

    /*
     * BUG: 'writable' checked without lock; another thread can flip mapping
     * between check and write — classic Dirty COW race window.
     */
    if (!pg->writable)
        return -1;

    /* --- race: concurrent path clears writable or swaps page pointer --- */

    pg->page[offset] = value;
    return 0;
}

int cow_madvise_dontneed(cow_page_t *pg)
{
    if (pg == NULL)
        return -1;

    pg->writable = 0;
    return 0;
}
