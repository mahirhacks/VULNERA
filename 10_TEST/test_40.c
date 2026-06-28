/*
 * test_40.c
 *
 * Label: SAFE — pre-2019 era sample (checked size malloc)
 * Inspired by: overflow-checked alloc (pre-2019)
 * CWE: none
 */
#include <stdlib.h>
#include <stddef.h>

unsigned char *alloc_pixel_buffer(unsigned int width, unsigned int height)
{
    size_t nbytes;
    if (width == 0 || height == 0)
        return NULL;
    if (__builtin_mul_overflow((size_t)width, (size_t)height, &nbytes))
        return NULL;
    if (__builtin_mul_overflow(nbytes, 4, &nbytes))
        return NULL;
    return (unsigned char *)malloc(nbytes);
}
