| model | test set | policy | label | threshold | SELD | ER | F | LE (deg) | LR |
|---|---|---|---|---|---|---|---|---|---|
| mixed_bce | starss23 | fixed_configured_threshold | headline | 0.5 | 0.6986 | 0.8980 | 0.1111 | 13.69 | 0.0684 |
| mixed_bce | starss23 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.1 | 0.6959 | 0.8927 | 0.1088 | 13.71 | 0.0765 |
| mixed_focal (reference) | starss23 | fixed_configured_threshold | headline | 0.5 | 0.6658 | 0.8458 | 0.1747 | 13.57 | 0.0832 |
| mixed_focal (reference) | starss23 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.4 | 0.6615 | 0.8357 | 0.1735 | 13.62 | 0.0920 |
| ExpA | starss23 | fixed_configured_threshold | headline | 0.5 | 0.6354 | 0.8160 | 0.2352 | 12.48 | 0.1083 |
| ExpA | starss23 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.3 | 0.6070 | 0.7660 | 0.2470 | 13.05 | 0.1635 |
| ExpB | starss23 | fixed_configured_threshold | headline | 0.5 | 0.6627 | 0.8455 | 0.1964 | 13.74 | 0.0747 |
| ExpB | starss23 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.5 | 0.6627 | 0.8455 | 0.1964 | 13.74 | 0.0747 |
| ExpB-beta.7 | starss23 | fixed_configured_threshold | headline | 0.3 | 0.8225 | 1.5275 | 0.1547 | 12.54 | 0.1523 |
| ExpB-beta.7 | starss23 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.5 | 0.6646 | 0.8560 | 0.1755 | 13.25 | 0.0956 |
| ExpC | starss23 | fixed_configured_threshold | headline | 0.5 | 0.6824 | 0.8849 | 0.1595 | 13.68 | 0.0717 |
| ExpC | starss23 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.4 | 0.6535 | 0.8282 | 0.1797 | 13.47 | 0.1095 |
| ExpA-posw | starss23 | fixed_configured_threshold | headline | 0.3 | 0.9540 | 2.1219 | 0.1517 | 12.54 | 0.2238 |
| ExpA-posw | starss23 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.5 | 0.6483 | 0.8358 | 0.1669 | 12.87 | 0.1473 |
| TAU2019 focal (native) | tau2019 | fixed_configured_threshold | headline | 0.5 | 0.1216 | 0.1449 | 0.8669 | 7.55 | 0.8337 |
| TAU2019 focal (native) | tau2019 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.5 | 0.1216 | 0.1449 | 0.8669 | 7.55 | 0.8337 |
| TAU2019-ExpA (native) | tau2019 | fixed_configured_threshold | headline | 0.5 | 0.1415 | 0.1814 | 0.8597 | 6.55 | 0.7921 |
| TAU2019-ExpA (native) | tau2019 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.4 | 0.1210 | 0.1477 | 0.8581 | 6.65 | 0.8426 |
| NIGENS2021 focal (native) | tau_nigens2021 | fixed_configured_threshold | headline | 0.5 | 0.5510 | 0.7215 | 0.3124 | 12.63 | 0.2753 |
| NIGENS2021 focal (native) | tau_nigens2021 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.4 | 0.5453 | 0.7076 | 0.3081 | 12.65 | 0.2884 |
| NIGENS2021-ExpA (native) | tau_nigens2021 | fixed_configured_threshold | headline | 0.5 | 0.5371 | 0.7178 | 0.3448 | 11.65 | 0.2895 |
| NIGENS2021-ExpA (native) | tau_nigens2021 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.4 | 0.5051 | 0.6586 | 0.3549 | 11.67 | 0.3482 |
| ictx w20 | starss23 | fixed_configured_threshold | headline | 0.5 | 0.7279 | 0.9382 | 0.0745 | 14.10 | 0.0304 |
| ictx w20 | starss23 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.4 | 0.7250 | 0.9319 | 0.0742 | 14.02 | 0.0354 |
| topt BiGRU | starss23 | fixed_configured_threshold | headline | 0.5 | 0.6833 | 0.8781 | 0.1417 | 13.06 | 0.0758 |
| topt BiGRU | starss23 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.4 | 0.6804 | 0.8715 | 0.1397 | 13.03 | 0.0824 |
| topt TT-4 | starss23 | fixed_configured_threshold | headline | 0.5 | 0.7027 | 0.8983 | 0.1133 | 12.74 | 0.0449 |
| topt TT-4 | starss23 | test_optimal_global_threshold | post-hoc (test-optimised) | 0.4 | 0.7005 | 0.8922 | 0.1131 | 12.87 | 0.0486 |
| ExpA | starss23 | fixed_0.5_recomputed | headline | 0.5 | 0.6354 | 0.8160 | 0.2352 | 12.48 | 0.1083 |
| ExpA | starss23 | val_selected_global_macroF1 | validation-selected | 0.35 | 0.6097 | 0.7676 | 0.2488 | 12.49 | 0.1495 |
| ExpA | starss23 | val_selected_class_macroF1 | validation-selected | class-specific (validation) | 0.6113 | 0.7680 | 0.2418 | 12.28 | 0.1491 |
| ExpA | starss23 | class_wise_test_oracle | post-hoc (test-optimised) | class-specific (test oracle) | 0.6796 | 1.0347 | 0.2268 | 12.73 | 0.1601 |
| mixed_focal (reference) | starss23 | fixed_0.5_recomputed | headline | 0.5 | 0.6658 | 0.8458 | 0.1747 | 13.57 | 0.0832 |
| mixed_focal (reference) | starss23 | val_selected_global_macroF1 | validation-selected | 0.45 | 0.6634 | 0.8405 | 0.1747 | 13.59 | 0.0875 |
| mixed_focal (reference) | starss23 | val_selected_class_macroF1 | validation-selected | class-specific (validation) | 0.6652 | 0.8442 | 0.1739 | 13.65 | 0.0852 |
| mixed_focal (reference) | starss23 | class_wise_test_oracle | post-hoc (test-optimised) | class-specific (test oracle) | 0.6600 | 0.8330 | 0.1741 | 13.46 | 0.0936 |
| ExpA | starss23 | matrix:val_selected_global_sed_f1_at_20deg | validation-selected | 0.5 | 0.6354 | 0.8160 | 0.2352 | 12.48 | 0.1083 |
| ExpA | starss23 | matrix:val_selected_global_seld_score_at_20deg | validation-selected | 0.45 | 0.6263 | 0.7999 | 0.2424 | 12.48 | 0.1217 |
| ExpA | starss23 | matrix:val_selected_class_specific_sed | validation-selected | class-specific vector | 0.6294 | 0.8027 | 0.2312 | 12.74 | 0.1245 |
| ExpA | starss23 | matrix:val_selected_class_specific_seld_at_20deg | validation-selected | class-specific vector | 0.6328 | 0.8082 | 0.2288 | 12.71 | 0.1187 |
| TAU2019-ExpA (native) | tau2019 | matrix:val_selected_global_sed_f1_at_20deg | validation-selected | 0.45 | 0.1283 | 0.1577 | 0.8618 | 6.60 | 0.8195 |
| TAU2019-ExpA (native) | tau2019 | matrix:val_selected_global_seld_score_at_20deg | validation-selected | 0.5 | 0.1415 | 0.1814 | 0.8597 | 6.55 | 0.7921 |
| TAU2019-ExpA (native) | tau2019 | matrix:val_selected_class_specific_sed | validation-selected | class-specific vector | 0.1329 | 0.1674 | 0.8628 | 6.58 | 0.8095 |
| TAU2019-ExpA (native) | tau2019 | matrix:val_selected_class_specific_seld_at_20deg | validation-selected | class-specific vector | 0.1420 | 0.1825 | 0.8597 | 6.55 | 0.7913 |
| NIGENS2020-ExpA (native) | tau_nigens2020 | matrix:val_selected_global_sed_f1_at_20deg | validation-selected | 0.35 | 0.7027 | 1.3819 | 0.2664 | 12.02 | 0.3713 |
| NIGENS2020-ExpA (native) | tau_nigens2020 | matrix:val_selected_global_seld_score_at_20deg | validation-selected | 0.45 | 0.5405 | 0.7095 | 0.3248 | 11.91 | 0.2887 |
| NIGENS2020-ExpA (native) | tau_nigens2020 | matrix:val_selected_class_specific_sed | validation-selected | class-specific vector | 0.7433 | 1.5258 | 0.2530 | 12.02 | 0.3664 |
| NIGENS2020-ExpA (native) | tau_nigens2020 | matrix:val_selected_class_specific_seld_at_20deg | validation-selected | class-specific vector | 0.5381 | 0.7071 | 0.3247 | 11.90 | 0.2962 |
| NIGENS2021-ExpA (native) | tau_nigens2021 | matrix:val_selected_global_sed_f1_at_20deg | validation-selected | 0.3 | 0.5751 | 0.9759 | 0.3354 | 11.69 | 0.4050 |
| NIGENS2021-ExpA (native) | tau_nigens2021 | matrix:val_selected_global_seld_score_at_20deg | validation-selected | 0.4 | 0.5051 | 0.6586 | 0.3549 | 11.67 | 0.3482 |
| NIGENS2021-ExpA (native) | tau_nigens2021 | matrix:val_selected_class_specific_sed | validation-selected | class-specific vector | 0.5988 | 1.0450 | 0.3294 | 11.68 | 0.3852 |
| NIGENS2021-ExpA (native) | tau_nigens2021 | matrix:val_selected_class_specific_seld_at_20deg | validation-selected | class-specific vector | 0.4991 | 0.6417 | 0.3595 | 11.66 | 0.3506 |
