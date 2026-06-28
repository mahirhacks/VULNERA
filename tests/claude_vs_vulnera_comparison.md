# VULNERA vs claude.md — graduated signature boost

Policy: corroboration (omega=0.15) + graduated blend toward plateau (no hard 84.6% jump).

## Summary vs hard plateau

| Metric | Hard plateau | Graduated |
|--------|-------------:|----------:|
| Triage agree | 67.6% | 67.6% |
| Vuln recall (flagged) | 68.8% | 68.8% |
| Safe specificity | 65.2% | 65.2% |
| Mean |Δ| vs claude | 29.4 | 31.1 |

## PrimeVul F1

| Split | ML-only | Plateau+corr | Graduated |
|-------|--------:|-------------:|----------:|
| valid | 0.4811 | 0.4914 | **0.4914** |
| test | 0.4697 | 0.4768 | **0.4768** |

## Per function

| File | Function | Claude | VULNERA | Boost | Match |
|------|----------|-------:|--------:|-------|-------|
| test_1.c | `tls_process_heartbeat` | 97% | 71.5% | graduated | ✓ |
| test_2.c | `bash_exported_function_check` | 95% | 25.4% | nan | ✗ |
| test_3.c | `cow_follow_write` | 88% | 57.0% | graduated | ✓ |
| test_3.c | `cow_madvise_dontneed` | 12% | 12.7% | nan | ✓ |
| test_4.c | `samba_load_pipename_module` | 82% | 25.8% | nan | ✗ |
| test_5.c | `parse_dns_aaaa_rdata` | 96% | 75.0% | graduated | ✓ |
| test_6.c | `nginx_chunked_append` | 84% | 83.6% | graduated | ✓ |
| test_7.c | `imagemagick_delegate_convert` | 94% | 84.3% | graduated | ✓ |
| test_8.c | `smb_trans2_copy_params` | 95% | 74.5% | graduated | ✓ |
| test_9.c | `stagefright_parse_stsc` | 90% | 71.5% | graduated | ✓ |
| test_10.c | `tls_build_heartbeat_response` | 93% | 84.3% | graduated | ✓ |
| test_11.c | `sudoers_unescape_heap` | 91% | 35.1% | nan | ✓ |
| test_12.c | `pkexec_collect_args` | 79% | 84.6% | graduated | ✓ |
| test_13.c | `cls_route_teardown` | 93% | 20.4% | nan | ✗ |
| test_13.c | `cls_route_init_job` | 8% | 29.8% | nan | ✓ |
| test_14.c | `glibc_tunables_parse_stack` | 92% | 84.6% | graduated | ✓ |
| test_15.c | `nft_set_flush_pair` | 85% | 22.1% | nan | ✗ |
| test_15.c | `nft_set_elem_attach` | 70% | 84.3% | graduated | ✓ |
| test_16.c | `legacy_parse_param` | 89% | 80.8% | graduated | ✓ |
| test_16.c | `fs_context_alloc` | 5% | 26.5% | nan | ✓ |
| test_17.c | `nft_payload_copy` | 87% | 84.6% | graduated | ✓ |
| test_18.c | `io_register_pbuf` | 15% | 32.4% | nan | ✗ |
| test_18.c | `io_unregister_pbuf` | 80% | 20.4% | nan | ✗ |
| test_18.c | `io_provide_buffers` | 91% | 32.4% | nan | ✓ |
| test_19.c | `wall_broadcast` | 72% | 25.4% | nan | ✗ |
| test_20.c | `smb_session_lookup` | 10% | 31.8% | nan | ✓ |
| test_20.c | `smb_session_destroy` | 30% | 32.4% | nan | ✗ |
| test_20.c | `smb_handle_request` | 92% | 33.0% | nan | ✓ |
| test_21.c | `pipe_merge_write` | 90% | 84.3% | graduated | ✓ |
| test_22.c | `socks5_set_target` | 95% | 80.8% | graduated | ✓ |
| test_23.c | `zlib_read_extra` | 94% | 84.6% | graduated | ✓ |
| test_24.c | `sudoers_unescape_heap_v2` | 93% | 33.0% | nan | ✓ |
| test_25.c | `bpf_array_update_elem` | 75% | 84.3% | graduated | ✓ |
| test_26.c | `gs_device_write_hex_line` | 92% | 84.6% | graduated | ✓ |
| test_26.c | `hex_nibble` | 3% | 5.2% | nan | ✓ |
| test_27.c | `smtp_header_append` | 83% | 84.6% | graduated | ✓ |
| test_28.c | `vmw_ioctl_destroy_bo` | 81% | 33.0% | nan | ✓ |
| test_28.c | `vmw_ioctl_destroy_bo_alias` | 65% | 28.3% | nan | ✗ |
| test_28.c | `vmw_bo_create_shared` | 20% | 32.4% | nan | ✗ |
| test_29.c | `com_field_read_name` | 95% | 84.3% | graduated | ✓ |
| test_30.c | `pkexec_collect_args_unbounded` | 96% | 84.6% | graduated | ✓ |
| test_31.c | `store_username` | 94% | 68.2% | graduated | ✓ |
| test_32.c | `store_username` | 4% | 31.8% | nan | ✓ |
| test_33.c | `read_user_line` | 98% | 78.1% | graduated | ✓ |
| test_34.c | `read_user_line` | 3% | 76.2% | graduated | ✗ |
| test_35.c | `build_config_path` | 90% | 72.4% | graduated | ✓ |
| test_36.c | `build_config_path` | 4% | 72.4% | graduated | ✗ |
| test_37.c | `log_client_message` | 93% | 60.9% | graduated | ✓ |
| test_38.c | `log_client_message` | 2% | 25.4% | nan | ✓ |
| test_39.c | `alloc_pixel_buffer` | 88% | 25.4% | nan | ✗ |
| test_40.c | `alloc_pixel_buffer` | 3% | 42.4% | graduated | ✗ |
| test_41.c | `validate_session` | 96% | 25.0% | nan | ✗ |
| test_42.c | `validate_session` | 3% | 23.4% | nan | ✓ |
| test_43.c | `cache_release` | 95% | 18.6% | nan | ✗ |
| test_44.c | `cache_release` | 2% | 11.9% | nan | ✓ |
| test_45.c | `node_value` | 80% | 11.9% | nan | ✗ |
| test_46.c | `node_value` | 3% | 7.8% | nan | ✓ |
| test_47.c | `copy_tags` | 91% | 31.8% | nan | ✗ |
| test_48.c | `copy_tags` | 4% | 31.8% | nan | ✓ |
| test_49.c | `pack_header` | 89% | 65.3% | graduated | ✓ |
| test_50.c | `pack_header` | 3% | 74.5% | graduated | ✗ |
| test_51.c | `grow_buffer` | 86% | 26.7% | nan | ✗ |
| test_52.c | `grow_buffer` | 4% | 31.4% | nan | ✓ |
| test_53.c | `parse_token` | 90% | 29.8% | nan | ✗ |
| test_54.c | `parse_token` | 3% | 29.8% | nan | ✓ |
| test_55.c | `append_fragment` | 89% | 71.5% | graduated | ✓ |
| test_56.c | `append_fragment` | 4% | 25.0% | nan | ✓ |
| test_57.c | `copy_payload` | 92% | 72.4% | graduated | ✓ |
| test_58.c | `copy_payload` | 5% | 71.5% | graduated | ✗ |
| test_59.c | `average_count` | 55% | 8.3% | nan | ✗ |
| test_60.c | `average_count` | 3% | 17.4% | nan | ✓ |
