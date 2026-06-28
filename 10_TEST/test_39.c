/*
 * test_39.c
 *
 * Label: VULNERABLE — pre-2019 era sample (integer overflow malloc)
 * Inspired by: CVE-2010-3856 class (2010)
 * CWE: CWE-190 / CWE-122
 */
#include <stdlib.h>
#include <stdint.h>

unsigned char *alloc_pixel_buffer(unsigned int width, unsigned int height)
{
    unsigned int nbytes = width * height * 4;
    /* BUG: multiplication wraps; small allocation for huge dimensions */
    return (unsigned char *)malloc(nbytes);
}
