/*
 * test_29.c
 *
 * Inspired by: CVE-2022-24094 (MariaDB COM_FIELD heap overflow, Feb 2022)
 * CWE: CWE-122 (Heap-based Buffer Overflow)
 *
 * Bug: column name length read as 16-bit from packet; allocation uses truncated
 * low byte only while memcpy uses full 16-bit length from wire.
 */

#include <stdlib.h>
#include <string.h>

typedef struct {
    char  *name;
    size_t name_len;
} com_field_t;

int com_field_read_name(com_field_t *field, const unsigned char *packet,
                        unsigned int packet_len)
{
    unsigned short wire_len;
    unsigned char alloc_len;
    char *heap_name;

    if (field == NULL || packet == NULL || packet_len < 2)
        return -1;

    wire_len = (unsigned short)(packet[0] | (packet[1] << 8));

    if (wire_len + 2 > packet_len)
        return -1;

    /*
     * BUG: alloc_len is low 8 bits only; wire_len can be 0x1234 while
     * malloc gets 0x34 bytes then memcpy copies wire_len bytes from packet.
     */
    alloc_len = (unsigned char)wire_len;
    heap_name = (char *)malloc((size_t)alloc_len + 1);
    if (heap_name == NULL)
        return -1;

    memcpy(heap_name, packet + 2, wire_len);
    heap_name[alloc_len] = '\0';

    field->name = heap_name;
    field->name_len = wire_len;
    return 0;
}
