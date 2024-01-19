#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION ".nv_fatbin"
asm(".section .nv_fatbin, \"a\"\n"
    ".align 8\n"
    "fatbinData:\n"
    ".quad "
    "0x00100001ba55ed50,0x00000000000003c0,0x0000004001010002,"
    "0x0000000000000380\n"
    ".quad "
    "0x0000000000000000,0x0000003400010007,0x0000000000000000,"
    "0x0000000000000011\n"
    ".quad "
    "0x0000000000000000,0x0000000000000000,0x33010102464c457f,"
    "0x0000000000000007\n"
    ".quad "
    "0x0000007a00be0002,0x0000000000000000,0x0000000000000310,"
    "0x0000000000000190\n"
    ".quad "
    "0x0038004000340534,0x0001000600400002,0x7472747368732e00,"
    "0x747274732e006261\n"
    ".quad "
    "0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,"
    "0x7466752e766e2e00\n"
    ".quad "
    "0x2e007972746e652e,0x006f666e692e766e,0x6c6c61632e766e2e,"
    "0x6e2e006870617267\n"
    ".quad "
    "0x746f746f72702e76,0x2e766e2e00657079,0x697463612e6c6572,"
    "0x7368732e00006e6f\n"
    ".quad "
    "0x732e006261747274,0x732e006261747274,0x732e006261746d79,"
    "0x68735f6261746d79\n"
    ".quad "
    "0x2e766e2e0078646e,0x72746e652e746675,0x6e692e766e2e0079,"
    "0x632e766e2e006f66\n"
    ".quad "
    "0x68706172676c6c61,0x6f72702e766e2e00,0x2e00657079746f74,"
    "0x612e6c65722e766e\n"
    ".quad "
    "0x0000006e6f697463,0x0000000000000000,0x0000000000000000,"
    "0x0000000000000000\n"
    ".quad "
    "0x0004000300000040,0x0000000000000000,0x0000000000000000,"
    "0x000500030000005c\n"
    ".quad "
    "0x0000000000000000,0x0000000000000000,0xffffffff00000000,"
    "0xfffffffe00000000\n"
    ".quad "
    "0xfffffffd00000000,0xfffffffc00000000,0x0000000000000073,"
    "0x3605002511000000\n"
    ".quad "
    "0x0000000000000000,0x0000000000000000,0x0000000000000000,"
    "0x0000000000000000\n"
    ".quad "
    "0x0000000000000000,0x0000000000000000,0x0000000000000000,"
    "0x0000000000000000\n"
    ".quad "
    "0x0000000300000001,0x0000000000000000,0x0000000000000000,"
    "0x0000000000000040\n"
    ".quad "
    "0x000000000000006b,0x0000000000000000,0x0000000000000001,"
    "0x0000000000000000\n"
    ".quad "
    "0x000000030000000b,0x0000000000000000,0x0000000000000000,"
    "0x00000000000000ab\n"
    ".quad "
    "0x000000000000006b,0x0000000000000000,0x0000000000000001,"
    "0x0000000000000000\n"
    ".quad "
    "0x0000000200000013,0x0000000000000000,0x0000000000000000,"
    "0x0000000000000118\n"
    ".quad "
    "0x0000000000000048,0x0000000300000002,0x0000000000000008,"
    "0x0000000000000018\n"
    ".quad "
    "0x7000000100000040,0x0000000000000000,0x0000000000000000,"
    "0x0000000000000160\n"
    ".quad "
    "0x0000000000000020,0x0000000000000005,0x0000000000000004,"
    "0x0000000000000008\n"
    ".quad "
    "0x7000000b0000005c,0x0000000000000000,0x0000000000000000,"
    "0x0000000000000180\n"
    ".quad "
    "0x0000000000000010,0x0000000000000000,0x0000000000000008,"
    "0x0000000000000008\n"
    ".quad "
    "0x0000000500000006,0x0000000000000310,0x0000000000000000,"
    "0x0000000000000000\n"
    ".quad "
    "0x0000000000000070,0x0000000000000070,0x0000000000000008,"
    "0x0000000500000001\n"
    ".quad "
    "0x0000000000000310,0x0000000000000000,0x0000000000000000,"
    "0x0000000000000070\n"
    ".quad 0x0000000000000070, 0x0000000000000008\n"
    ".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[122];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__((aligned(8)))
__attribute__((section(__CUDAFATBINSECTION))) = {
    0x466243b1, 2, fatbinData, (void **)__cudaPrelinkedFatbins};
#ifdef __cplusplus
}
#endif
