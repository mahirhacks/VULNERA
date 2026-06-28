/*
 * test_28.c
 *
 * Inspired by: CVE-2022-22705 (Linux vmwgfx driver double-free, Jan 2022)
 * CWE: CWE-415 (Double Free)
 *
 * Bug: ioctl error path frees the backing BO; success path also frees when
 * re-submitting the same handle without clearing the stored pointer.
 */

#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned int handle;
    void        *backing;
    size_t       size;
} vmw_bo_t;

typedef struct {
    vmw_bo_t *objects[16];
} vmw_file_t;

int vmw_ioctl_destroy_bo(vmw_file_t *file, unsigned int handle)
{
    vmw_bo_t *bo;
    int i;

    if (file == NULL)
        return -1;

    for (i = 0; i < 16; i++) {
        if (file->objects[i] != NULL &&
            file->objects[i]->handle == handle) {
            bo = file->objects[i];
            free(bo->backing);
            free(bo);
            /*
             * BUG: slot left dangling; second destroy on same handle
             * or alias path frees backing again.
             */
            return 0;
        }
    }
    return -1;
}

int vmw_ioctl_destroy_bo_alias(vmw_file_t *file, vmw_bo_t *bo_ref)
{
    if (file == NULL || bo_ref == NULL || bo_ref->backing == NULL)
        return -1;

    /*
     * BUG: frees backing through reference without nulling other aliases
     * that share the same backing pointer.
     */
    free(bo_ref->backing);
    bo_ref->backing = NULL;
    return 0;
}

int vmw_bo_create_shared(vmw_file_t *file, vmw_bo_t *a, vmw_bo_t *b,
                         size_t size)
{
    void *block;

    if (file == NULL || a == NULL || b == NULL || size == 0)
        return -1;

    block = malloc(size);
    if (block == NULL)
        return -1;

    a->backing = block;
    b->backing = block;
    a->size = b->size = size;
    return 0;
}
