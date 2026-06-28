/*
 * test_21.c
 *
 * Inspired by: CVE-2022-0847 (Linux "Dirty Pipe", disclosed Mar 2022)
 * CWE: CWE-787 (Out-of-bounds Write)
 *
 * Bug: splice-style copy assumes pipe buffer is fully writable from offset 0,
 * but a prior partial fill leaves data_offset > 0; write still starts at base
 * and overwrites adjacent pipe metadata / prior page contents.
 */

#include <string.h>

#define PIPE_BUF_SIZE 4096

typedef struct {
    unsigned int len;
    unsigned int offset;
    char           data[PIPE_BUF_SIZE];
} pipe_buffer_t;

typedef struct {
    pipe_buffer_t *buf;
    int            can_merge;
} pipe_inode_t;

int pipe_merge_write(pipe_inode_t *pipe, const char *src, unsigned int count)
{
    pipe_buffer_t *buf;
    char *dst;

    if (pipe == NULL || pipe->buf == NULL || src == NULL || count == 0)
        return -1;

    buf = pipe->buf;

    /*
     * BUG: ignores buf->offset (partial packet already in buffer).
     * Writes at data[0] instead of data[offset], clobbering in-use bytes
     * and effectively growing the logical overwrite past buf->len.
     */
    if (buf->offset + count > PIPE_BUF_SIZE)
        return -1;

    dst = buf->data;   /* should be buf->data + buf->offset */
    memcpy(dst, src, count);
    buf->len = count;  /* should be offset + count */
    return (int)count;
}
