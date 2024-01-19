#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/compiler.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module __section(".gnu.linkonce.this_module") = {
    .name = KBUILD_MODNAME,
    .init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
    .exit = cleanup_module,
#endif
    .arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[] __used
    __section("__versions") = {
        {0x46519bf6, "__register_chrdev"},
        {0x6bc3fbc0, "__unregister_chrdev"},
        {0xccb0d112, "kmalloc_caches"},
        {0xc9a6c15a, "kmalloc_trace"},
        {0xcefb0c9f, "__mutex_init"},
        {0xb4c9b5bf, "pcpu_hot"},
        {0x9f4bb615, "address_space_init_once"},
        {0x4dfa8d4b, "mutex_lock"},
        {0x3213f038, "mutex_unlock"},
        {0x8a35b432, "sme_me_mask"},
        {0x59861a0b, "remap_pfn_range"},
        {0x13c49cc2, "_copy_from_user"},
        {0x7b4da6ff, "__init_rwsem"},
        {0x5b3f3e79, "nvidia_p2p_get_pages"},
        {0xd6b33026, "cpu_khz"},
        {0x6b10bee1, "_copy_to_user"},
        {0x668b19a1, "down_read"},
        {0x53b954a2, "up_read"},
        {0x364c23ad, "mutex_is_locked"},
        {0xf0fdf6cb, "__stack_chk_fail"},
        {0x4b323cea, "param_ops_int"},
        {0xbdfb6dbb, "__fentry__"},
        {0x5b8239ca, "__x86_return_thunk"},
        {0x122c3a7e, "_printk"},
        {0xce807a25, "up_write"},
        {0x642487ac, "nvidia_p2p_put_pages"},
        {0x37a0cba, "kfree"},
        {0x57bc19d2, "down_write"},
        {0xf42ca687, "nvidia_p2p_free_page_table"},
        {0x41bc9bb3, "unmap_mapping_range"},
        {0x61958780, "module_layout"},
};

MODULE_INFO(depends, "nv-p2p-dummy");

MODULE_INFO(srcversion, "5FD33EAD8107E1C4EB061FD");

MODULE_INFO(suserelease, "openSUSE Tumbleweed");
