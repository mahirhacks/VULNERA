/*
 * test_18.c
 *
 * Inspired by: CVE-2024-0582 (Linux io_uring registered-buffer UAF, Jan 2024)
 * CWE: CWE-416 (Use After Free)
 *
 * Bug: io_unregister_pbuf frees the buffer group but leaves the ring's
 * pointer intact. A subsequent io_provide_buffers call dereferences the
 * dangling pointer.
 */

#include <stdlib.h>
#include <string.h>

typedef struct {
    void  *addr;
    size_t len;
    int    buf_group;
} io_buf_t;

typedef struct {
    io_buf_t *pbuf;
    int       pbuf_registered;
} io_ring_ctx_t;

int io_register_pbuf(io_ring_ctx_t *ctx, void *user_addr, size_t len, int group)
{
    io_buf_t *buf;

    if (ctx == NULL || user_addr == NULL || len == 0)
        return -1;

    buf = (io_buf_t *)malloc(sizeof(io_buf_t));
    if (buf == NULL)
        return -1;

    buf->addr      = user_addr;
    buf->len       = len;
    buf->buf_group = group;

    ctx->pbuf            = buf;
    ctx->pbuf_registered = 1;
    return 0;
}

int io_unregister_pbuf(io_ring_ctx_t *ctx)
{
    if (ctx == NULL || !ctx->pbuf_registered)
        return -1;

    free(ctx->pbuf);
    /*
     * BUG: sets registered flag to 0 but does NOT null out ctx->pbuf.
     * The dangling pointer remains accessible.
     */
    ctx->pbuf_registered = 0;
    return 0;
}

int io_provide_buffers(io_ring_ctx_t *ctx, void *new_addr, size_t new_len)
{
    if (ctx == NULL || new_addr == NULL)
        return -1;

    /*
     * BUG: checks the flag (which is 0 after unregister) but falls through
     * to the else branch which dereferences ctx->pbuf — a dangling pointer
     * freed in io_unregister_pbuf.
     */
    if (ctx->pbuf_registered) {
        ctx->pbuf->addr = new_addr;
        ctx->pbuf->len  = new_len;
    } else {
        /* "re-provide": reuses old slot — UAF if unregistered */
        if (ctx->pbuf != NULL) {
            ctx->pbuf->addr = new_addr;   /* write-through-dangling-pointer */
            ctx->pbuf->len  = new_len;
        }
    }
    return 0;
}
