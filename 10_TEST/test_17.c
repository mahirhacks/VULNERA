/*
 * test_17.c
 *
 * Inspired by: CVE-2023-0179 (Linux nft_payload stack OOB, disclosed Mar 2023)
 * CWE: CWE-787 (Out-of-bounds Write)
 *
 * Bug: user-controlled 'offset' and 'len' are validated individually but
 * their sum is not checked against buffer size, allowing a write past the
 * end of the on-stack register area.
 */

#include <string.h>

#define NFT_REG_SIZE  16
#define NFT_REG_COUNT  4

typedef struct {
    unsigned char regs[NFT_REG_COUNT][NFT_REG_SIZE];
} nft_regs_t;

typedef struct {
    const unsigned char *data;
    unsigned int         data_len;
} nft_pkt_t;

int nft_payload_copy(nft_regs_t *regs,
                     unsigned int dreg,
                     const nft_pkt_t *pkt,
                     unsigned int offset,
                     unsigned int len)
{
    if (regs == NULL || pkt == NULL || pkt->data == NULL)
        return -1;

    /* Individual bounds checks pass for crafted values */
    if (dreg >= NFT_REG_COUNT)
        return -1;
    if (offset >= pkt->data_len)
        return -1;
    if (len > NFT_REG_SIZE)
        return -1;

    /*
     * BUG: offset + len can exceed pkt->data_len (no combined check),
     * and the destination &regs->regs[dreg][0] is only NFT_REG_SIZE bytes
     * but we copy 'len' bytes starting from an arbitrary 'offset'.
     * With offset near end-of-packet, the read goes OOB; with len == 16
     * and dreg == 3 the write is at the top of the regs array (stack).
     */
    memcpy(&regs->regs[dreg][0], pkt->data + offset, len);
    return 0;
}
