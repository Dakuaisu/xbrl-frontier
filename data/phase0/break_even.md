Prices: config/prices_2026-10-01.yaml (observed 2026-10-01). Blank T4 sources dropped: none.

| arm | regime | T4 rate key | util | arm $/filing | break-even f/h | gate-pass f/h | Qwen/Qwen3-0.6B measured f/h | Qwen/Qwen3-1.7B measured f/h | Qwen/Qwen3-0.6B beatable | Qwen/Qwen3-1.7B beatable |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-luna | offline | gpu_per_hour.modal.T4 | 0.90* | 0.0200 | 32.7 | 98.2 | 18.9 | 15.3 | False | False |
| gpt-6-luna | offline | gpu_per_hour.modal.T4 | 0.10 | 0.0200 | 294.7 | 884.2 | 18.9 | 15.3 | False | False |
| gpt-6-luna | offline | gpu_per_hour.modal.T4 | 0.50 | 0.0200 | 58.9 | 176.8 | 18.9 | 15.3 | False | False |
| gpt-6-luna | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.90* | 0.0200 | 27.8 | 83.3 | 18.9 | 15.3 | False | False |
| gpt-6-luna | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.10 | 0.0200 | 249.8 | 749.3 | 18.9 | 15.3 | False | False |
| gpt-6-luna | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.50 | 0.0200 | 50.0 | 149.9 | 18.9 | 15.3 | False | False |
| gpt-6-luna | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.90* | 0.0200 | 29.2 | 87.6 | 18.9 | 15.3 | False | False |
| gpt-6-luna | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.10 | 0.0200 | 262.7 | 788.2 | 18.9 | 15.3 | False | False |
| gpt-6-luna | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.50 | 0.0200 | 52.5 | 157.6 | 18.9 | 15.3 | False | False |
| gpt-6-luna | online | gpu_per_hour.modal.T4 | 0.50* | 0.0400 | 29.5 | 88.4 | 0.0 | 0.0 | False | False |
| gpt-6-luna | online | gpu_per_hour.modal.T4 | 0.10 | 0.0400 | 147.4 | 442.1 | 0.0 | 0.0 | False | False |
| gpt-6-luna | online | gpu_per_hour.modal.T4 | 0.90 | 0.0400 | 16.4 | 49.1 | 0.0 | 0.0 | False | False |
| gpt-6-luna | online | gpu_per_hour.huggingface_endpoints.T4 | 0.50* | 0.0400 | 25.0 | 74.9 | 0.0 | 0.0 | False | False |
| gpt-6-luna | online | gpu_per_hour.huggingface_endpoints.T4 | 0.10 | 0.0400 | 124.9 | 374.6 | 0.0 | 0.0 | False | False |
| gpt-6-luna | online | gpu_per_hour.huggingface_endpoints.T4 | 0.90 | 0.0400 | 13.9 | 41.6 | 0.0 | 0.0 | False | False |
| gpt-6-luna | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.50* | 0.0400 | 26.3 | 78.8 | 0.0 | 0.0 | False | False |
| gpt-6-luna | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.10 | 0.0400 | 131.4 | 394.1 | 0.0 | 0.0 | False | False |
| gpt-6-luna | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.90 | 0.0400 | 14.6 | 43.8 | 0.0 | 0.0 | False | False |
| gemini-3.5-flash-lite | offline | gpu_per_hour.modal.T4 | 0.90* | 0.0778 | 8.4 | 25.3 | 18.9 | 15.3 | False | False |
| gemini-3.5-flash-lite | offline | gpu_per_hour.modal.T4 | 0.10 | 0.0778 | 75.8 | 227.5 | 18.9 | 15.3 | False | False |
| gemini-3.5-flash-lite | offline | gpu_per_hour.modal.T4 | 0.50 | 0.0778 | 15.2 | 45.5 | 18.9 | 15.3 | False | False |
| gemini-3.5-flash-lite | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.90* | 0.0778 | 7.1 | 21.4 | 18.9 | 15.3 | False | False |
| gemini-3.5-flash-lite | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.10 | 0.0778 | 64.3 | 192.8 | 18.9 | 15.3 | False | False |
| gemini-3.5-flash-lite | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.50 | 0.0778 | 12.9 | 38.6 | 18.9 | 15.3 | False | False |
| gemini-3.5-flash-lite | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.90* | 0.0778 | 7.5 | 22.5 | 18.9 | 15.3 | False | False |
| gemini-3.5-flash-lite | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.10 | 0.0778 | 67.6 | 202.8 | 18.9 | 15.3 | False | False |
| gemini-3.5-flash-lite | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.50 | 0.0778 | 13.5 | 40.6 | 18.9 | 15.3 | False | False |
| gemini-3.5-flash-lite | online | gpu_per_hour.modal.T4 | 0.50* | 0.1556 | 7.6 | 22.8 | 0.0 | 0.0 | False | False |
| gemini-3.5-flash-lite | online | gpu_per_hour.modal.T4 | 0.10 | 0.1556 | 37.9 | 113.8 | 0.0 | 0.0 | False | False |
| gemini-3.5-flash-lite | online | gpu_per_hour.modal.T4 | 0.90 | 0.1556 | 4.2 | 12.6 | 0.0 | 0.0 | False | False |
| gemini-3.5-flash-lite | online | gpu_per_hour.huggingface_endpoints.T4 | 0.50* | 0.1556 | 6.4 | 19.3 | 0.0 | 0.0 | False | False |
| gemini-3.5-flash-lite | online | gpu_per_hour.huggingface_endpoints.T4 | 0.10 | 0.1556 | 32.1 | 96.4 | 0.0 | 0.0 | False | False |
| gemini-3.5-flash-lite | online | gpu_per_hour.huggingface_endpoints.T4 | 0.90 | 0.1556 | 3.6 | 10.7 | 0.0 | 0.0 | False | False |
| gemini-3.5-flash-lite | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.50* | 0.1556 | 6.8 | 20.3 | 0.0 | 0.0 | False | False |
| gemini-3.5-flash-lite | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.10 | 0.1556 | 33.8 | 101.4 | 0.0 | 0.0 | False | False |
| gemini-3.5-flash-lite | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.90 | 0.1556 | 3.8 | 11.3 | 0.0 | 0.0 | False | False |
| gpt-5.4-mini-2026-03-17 | offline | gpu_per_hour.modal.T4 | 0.90* | 0.1634 | 4.0 | 12.0 | 18.9 | 15.3 | True | True |
| gpt-5.4-mini-2026-03-17 | offline | gpu_per_hour.modal.T4 | 0.10 | 0.1634 | 36.1 | 108.3 | 18.9 | 15.3 | False | False |
| gpt-5.4-mini-2026-03-17 | offline | gpu_per_hour.modal.T4 | 0.50 | 0.1634 | 7.2 | 21.7 | 18.9 | 15.3 | False | False |
| gpt-5.4-mini-2026-03-17 | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.90* | 0.1634 | 3.4 | 10.2 | 18.9 | 15.3 | True | True |
| gpt-5.4-mini-2026-03-17 | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.10 | 0.1634 | 30.6 | 91.8 | 18.9 | 15.3 | False | False |
| gpt-5.4-mini-2026-03-17 | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.50 | 0.1634 | 6.1 | 18.4 | 18.9 | 15.3 | True | False |
| gpt-5.4-mini-2026-03-17 | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.90* | 0.1634 | 3.6 | 10.7 | 18.9 | 15.3 | True | True |
| gpt-5.4-mini-2026-03-17 | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.10 | 0.1634 | 32.2 | 96.5 | 18.9 | 15.3 | False | False |
| gpt-5.4-mini-2026-03-17 | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.50 | 0.1634 | 6.4 | 19.3 | 18.9 | 15.3 | False | False |
| gpt-5.4-mini-2026-03-17 | online | gpu_per_hour.modal.T4 | 0.50* | 0.3269 | 3.6 | 10.8 | 0.0 | 0.0 | False | False |
| gpt-5.4-mini-2026-03-17 | online | gpu_per_hour.modal.T4 | 0.10 | 0.3269 | 18.0 | 54.1 | 0.0 | 0.0 | False | False |
| gpt-5.4-mini-2026-03-17 | online | gpu_per_hour.modal.T4 | 0.90 | 0.3269 | 2.0 | 6.0 | 0.0 | 0.0 | False | False |
| gpt-5.4-mini-2026-03-17 | online | gpu_per_hour.huggingface_endpoints.T4 | 0.50* | 0.3269 | 3.1 | 9.2 | 0.0 | 0.0 | False | False |
| gpt-5.4-mini-2026-03-17 | online | gpu_per_hour.huggingface_endpoints.T4 | 0.10 | 0.3269 | 15.3 | 45.9 | 0.0 | 0.0 | False | False |
| gpt-5.4-mini-2026-03-17 | online | gpu_per_hour.huggingface_endpoints.T4 | 0.90 | 0.3269 | 1.7 | 5.1 | 0.0 | 0.0 | False | False |
| gpt-5.4-mini-2026-03-17 | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.50* | 0.3269 | 3.2 | 9.7 | 0.0 | 0.0 | False | False |
| gpt-5.4-mini-2026-03-17 | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.10 | 0.3269 | 16.1 | 48.3 | 0.0 | 0.0 | False | False |
| gpt-5.4-mini-2026-03-17 | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.90 | 0.3269 | 1.8 | 5.4 | 0.0 | 0.0 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | offline | gpu_per_hour.modal.T4 | 0.90* | 0.0445 | 14.7 | 44.2 | 18.9 | 15.3 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | offline | gpu_per_hour.modal.T4 | 0.10 | 0.0445 | 132.7 | 398.0 | 18.9 | 15.3 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | offline | gpu_per_hour.modal.T4 | 0.50 | 0.0445 | 26.5 | 79.6 | 18.9 | 15.3 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.90* | 0.0445 | 12.5 | 37.5 | 18.9 | 15.3 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.10 | 0.0445 | 112.4 | 337.3 | 18.9 | 15.3 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | offline | gpu_per_hour.huggingface_endpoints.T4 | 0.50 | 0.0445 | 22.5 | 67.5 | 18.9 | 15.3 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.90* | 0.0445 | 13.1 | 39.4 | 18.9 | 15.3 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.10 | 0.0445 | 118.3 | 354.8 | 18.9 | 15.3 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | offline | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.50 | 0.0445 | 23.7 | 71.0 | 18.9 | 15.3 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | online | gpu_per_hour.modal.T4 | 0.50* | 0.0889 | 13.3 | 39.8 | 0.0 | 0.0 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | online | gpu_per_hour.modal.T4 | 0.10 | 0.0889 | 66.3 | 199.0 | 0.0 | 0.0 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | online | gpu_per_hour.modal.T4 | 0.90 | 0.0889 | 7.4 | 22.1 | 0.0 | 0.0 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | online | gpu_per_hour.huggingface_endpoints.T4 | 0.50* | 0.0889 | 11.2 | 33.7 | 0.0 | 0.0 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | online | gpu_per_hour.huggingface_endpoints.T4 | 0.10 | 0.0889 | 56.2 | 168.6 | 0.0 | 0.0 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | online | gpu_per_hour.huggingface_endpoints.T4 | 0.90 | 0.0889 | 6.2 | 18.7 | 0.0 | 0.0 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.50* | 0.0889 | 11.8 | 35.5 | 0.0 | 0.0 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.10 | 0.0889 | 59.1 | 177.4 | 0.0 | 0.0 | False | False |
| gpt-5.4-nano-2026-03-17 (swap) | online | gpu_per_hour.aws.g4dn_xlarge_T4 | 0.90 | 0.0889 | 6.6 | 19.7 | 0.0 | 0.0 | False | False |

* = the regime's utilization in gates.cost_regimes; other rows are serving.utilization_scenarios.