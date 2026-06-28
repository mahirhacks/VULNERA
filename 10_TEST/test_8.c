/*
 * test_8.c
 *
 * Inspired by: CVE-2017-0144 (EternalBlue SMBv1, disclosed Mar 2017)
 * CWE: CWE-787 (Out-of-bounds Write)
 *
 * Bug: SMB transaction setup path copies Trans2 parameters into a kernel
 * buffer using client-supplied TotalDataCount without verifying against the
 * fixed staging buffer.
 */

#include <string.h>

#define SMB_TRANS2_BUF 1024

typedef struct {
    unsigned char staging[SMB_TRANS2_BUF];
    unsigned short total_data_count;
} smb_trans2_t;

int smb_trans2_copy_params(smb_trans2_t *trans, const unsigned char *params,
                           unsigned short param_count)
{
    if (trans == NULL || params == NULL)
        return -1;

    trans->total_data_count = param_count;

    /*
     * BUG: param_count from client can exceed SMB_TRANS2_BUF; memcpy smashes
     * staging and adjacent fields.
     */
    memcpy(trans->staging, params, param_count);
    return 0;
}
