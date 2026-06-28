/*
 * test_16.c
 *
 * Inspired by: CVE-2022-0185 (Linux fs_context heap overflow, disclosed Jan 2022)
 * CWE: CWE-190 (Integer Overflow) -> CWE-122 (Heap Buffer Overflow)
 *
 * Bug: len is size_t but subtracted from signed int 'remaining', which wraps
 * negative and passes the (remaining > 0) guard, allowing unbounded memcpy.
 */

#include <stdlib.h>
#include <string.h>

#define PAGE_SIZE 4096

typedef struct {
    char *buffer;
    int   capacity;
    int   used;
} fs_context_t;

int legacy_parse_param(fs_context_t *ctx, const char *key, const char *value)
{
    size_t key_len, val_len, total;
    int remaining;

    if (ctx == NULL || key == NULL || value == NULL)
        return -1;

    key_len  = strlen(key);
    val_len  = strlen(value);
    total    = key_len + 1 + val_len + 1;   /* "key=value\0" */

    remaining = ctx->capacity - ctx->used;

    /*
     * BUG: 'remaining' is signed int; 'total' is size_t (unsigned).
     * When ctx->used > ctx->capacity, remaining goes negative, but the
     * comparison (size_t)remaining promotes via unsigned conversion to a
     * huge positive number, so the check always passes.
     */
    if (total > (size_t)remaining)
        return -1;

    memcpy(ctx->buffer + ctx->used, key, key_len);
    ctx->buffer[ctx->used + key_len] = '=';
    memcpy(ctx->buffer + ctx->used + key_len + 1, value, val_len);
    ctx->used += (int)total;

    return 0;
}

fs_context_t *fs_context_alloc(void)
{
    fs_context_t *ctx = (fs_context_t *)malloc(sizeof(fs_context_t));
    if (ctx == NULL)
        return NULL;
    ctx->buffer   = (char *)malloc(PAGE_SIZE);
    ctx->capacity = PAGE_SIZE;
    ctx->used     = 0;
    if (ctx->buffer == NULL) {
        free(ctx);
        return NULL;
    }
    return ctx;
}
