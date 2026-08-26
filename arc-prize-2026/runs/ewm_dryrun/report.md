# EWM-execute Stage-0 gate dry-run (schema validation on real traces)

Producer: `scripts/ewm_replay_dryrun.py`; schema: `duck_eval/ewm_exec/EVENT_SCHEMA.md`; aggregator: `scripts/ewm_events.py`. Recorded Kaggle action streams replayed through the 12 saturated exec_wm sims; plan proxies = the recorded agent's own action batches. Mismatch rates below are REAL sim-vs-Kaggle-engine fidelity (incl. engine-version drift), measured before any executor build.

## war_eval_v1

Log: 2840 EWMEVT lines, 275,857 bytes, longest line 110 chars, 0 malformed.

| game | plans | steps | step_acc | plan lens | aborts | abort steps | survived | deadlocks | fallbacks |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| ft09 | 49 | 132 | 0.985 | 1x15,2x10,3x8,4x12,5x1,6x2,8x1 | 2 | 0x1,3x1 | 2 | 0 | 2 |
| lf52 | 73 | 73 | 0.301 | 1x73 | 51 | 0x51 | 51 | 0 | 51 |
| lp85 | 71 | 71 | 0.113 | 1x71 | 63 | 0x63 | 63 | 0 | 63 |
| ls20 | 33 | 76 | 0.684 | 1x18,2x7,3x3,4x2,7x1,9x1,26x1 | 24 | 0x22,2x1,3x1 | 24 | 0 | 24 |
| s5i5 | 50 | 50 | 0.360 | 1x46,2x1,6x1,7x2 | 32 | 0x32 | 32 | 0 | 32 |
| sb26 | 79 | 79 | 0.228 | 1x47,2x32 | 61 | 0x61 | 61 | 0 | 61 |
| sp80 | 89 | 89 | 0.067 | 1x55,2x6,3x9,4x2,5x9,6x1,7x1,8x1,.. | 83 | 0x83 | 83 | 0 | 83 |
| su15 | 76 | 76 | 0.382 | 1x74,4x1,16x1 | 47 | 0x47 | 47 | 0 | 47 |
| tn36 | 25 | 54 | 0.833 | 1x20,2x1,10x1,20x1,24x1,39x1 | 9 | 0x9 | 9 | 0 | 9 |
| tr87 | 90 | 113 | 0.832 | 1x80,2x2,3x1,4x5,7x1,13x1 | 19 | 0x17,2x2 | 19 | 0 | 19 |
| tu93 | 32 | 119 | 0.815 | 1x14,2x3,3x5,4x4,5x1,6x1,14x1,16x1,.. | 22 | 0x15,2x1,3x3,4x1,5x1,16x1 | 22 | 0 | 22 |
| vc33 | 43 | 44 | 0.295 | 1x32,2x2,4x4,5x2,6x1,8x1,12x1 | 31 | 0x31 | 31 | 0 | 31 |

Shadow stats (ALL recorded steps verified, not just pre-abort):

| game | held-out state_exact% | steps | exact | shadow acc | first divergence (step#) | med/max diff cells on mismatch | selfdiff | sim_error | done-flag agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft09 | 100.0 | 132 | 130 | 0.985 | 130 | 3600/3600 | 0 | 0 | 131/132 |
| lf52 | 100.0 | 73 | 22 | 0.301 | 0 | 56/1694 | 0 | 0 | 72/73 |
| lp85 | 100.0 | 71 | 8 | 0.113 | 7 | 33/1506 | 0 | 0 | 70/71 |
| ls20 | 100.0 | 91 | 58 | 0.637 | 50 | 2/1455 | 0 | 0 | 90/91 |
| s5i5 | 99.5 | 68 | 18 | 0.265 | 3 | 18/879 | 0 | 0 | 67/68 |
| sb26 | 100.0 | 111 | 18 | 0.162 | 0 | 20/753 | 0 | 0 | 110/111 |
| sp80 | 100.0 | 228 | 6 | 0.026 | 6 | 2/1168 | 0 | 0 | 227/228 |
| su15 | 99.5 | 94 | 29 | 0.309 | 0 | 8/287 | 0 | 0 | 93/94 |
| tn36 | 100.0 | 115 | 61 | 0.530 | 7 | 3/2614 | 0 | 0 | 114/115 |
| tr87 | 100.0 | 127 | 104 | 0.819 | 18 | 11/16 | 0 | 0 | 127/127 |
| tu93 | 100.0 | 145 | 106 | 0.731 | 67 | 1/734 | 0 | 0 | 143/145 |
| vc33 | 99.5 | 88 | 21 | 0.239 | 2 | 140/2920 | 0 | 0 | 86/88 |

`EWM_CANARY games_fired=12 threshold=5 verdict=PASS fired=[ft09,lf52,lp85,ls20,s5i5,sb26,sp80,su15,tn36,tr87,tu93,vc33]`
`EWM_ACTIVATION plans=710 steps=976 outcomes=710 deadlocks=0 verdict=ACTIVE`

## war_eval_v2

Log: 4065 EWMEVT lines, 394,675 bytes, longest line 110 chars, 0 malformed.

| game | plans | steps | step_acc | plan lens | aborts | abort steps | survived | deadlocks | fallbacks |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| ft09 | 21 | 30 | 0.833 | 1x14,2x2,3x2,4x1,6x1,11x1 | 5 | 0x4,3x1 | 5 | 0 | 5 |
| lf52 | 116 | 123 | 0.488 | 1x112,2x1,4x1,5x1,8x1 | 63 | 0x63 | 63 | 0 | 63 |
| lp85 | 17 | 20 | 0.550 | 1x14,2x1,4x2 | 9 | 0x8,3x1 | 9 | 0 | 9 |
| ls20 | 38 | 78 | 0.923 | 1x24,2x3,3x3,4x4,5x2,6x1,7x1 | 6 | 0x5,1x1 | 6 | 0 | 6 |
| s5i5 | 43 | 43 | 0.349 | 1x39,2x2,3x1,4x1 | 28 | 0x28 | 28 | 0 | 28 |
| sb26 | 177 | 177 | 0.153 | 1x100,2x77 | 150 | 0x150 | 150 | 0 | 150 |
| sp80 | 145 | 204 | 0.907 | 1x124,2x7,3x3,4x2,5x4,6x2,8x1,10x2 | 19 | 0x16,2x1,3x1,4x1 | 19 | 0 | 19 |
| su15 | 105 | 119 | 0.218 | 1x95,4x6,6x1,10x1,19x1,20x1 | 93 | 0x92,14x1 | 93 | 0 | 93 |
| tn36 | 48 | 142 | 1.000 | 1x19,2x2,3x9,4x9,5x1,6x6,7x1,8x1 | 0 | - | 0 | 0 | 0 |
| tr87 | 227 | 253 | 0.771 | 1x216,2x2,3x3,4x6 | 58 | 0x58 | 58 | 0 | 58 |
| tu93 | 30 | 112 | 0.830 | 1x19,2x1,3x3,4x3,6x1,10x1,14x1,50x1 | 19 | 0x14,1x2,2x1,3x1,13x1 | 19 | 0 | 19 |
| vc33 | 102 | 138 | 0.725 | 1x92,2x4,3x2,5x1,6x1,13x1,20x1 | 38 | 0x38 | 38 | 0 | 38 |

Shadow stats (ALL recorded steps verified, not just pre-abort):

| game | held-out state_exact% | steps | exact | shadow acc | first divergence (step#) | med/max diff cells on mismatch | selfdiff | sim_error | done-flag agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft09 | 100.0 | 45 | 25 | 0.556 | 25 | 37/3592 | 0 | 0 | 43/45 |
| lf52 | 100.0 | 131 | 65 | 0.496 | 0 | 28/1702 | 0 | 0 | 130/131 |
| lp85 | 100.0 | 24 | 11 | 0.458 | 8 | 93/1506 | 0 | 0 | 23/24 |
| ls20 | 100.0 | 78 | 72 | 0.923 | 48 | 8/10 | 0 | 0 | 78/78 |
| s5i5 | 99.5 | 50 | 15 | 0.300 | 1 | 9/19 | 0 | 0 | 50/50 |
| sb26 | 100.0 | 254 | 27 | 0.106 | 0 | 53/759 | 0 | 0 | 253/254 |
| sp80 | 100.0 | 215 | 189 | 0.879 | 11 | 160/160 | 0 | 0 | 215/215 |
| su15 | 99.5 | 174 | 26 | 0.149 | 0 | 6/293 | 0 | 0 | 173/174 |
| tn36 | 100.0 | 142 | 142 | 1.000 | - | - | 0 | 0 | 142/142 |
| tr87 | 100.0 | 253 | 195 | 0.771 | 21 | 13/16 | 0 | 0 | 253/253 |
| tu93 | 100.0 | 122 | 95 | 0.779 | 73 | 1/742 | 0 | 0 | 120/122 |
| vc33 | 99.5 | 150 | 100 | 0.667 | 2 | 248/2801 | 0 | 0 | 149/150 |

`EWM_CANARY games_fired=12 threshold=5 verdict=PASS fired=[ft09,lf52,lp85,ls20,s5i5,sb26,sp80,su15,tn36,tr87,tu93,vc33]`
`EWM_ACTIVATION plans=1069 steps=1439 outcomes=1069 deadlocks=0 verdict=ACTIVE`

## war_eval_v3

Log: 3591 EWMEVT lines, 348,090 bytes, longest line 111 chars, 0 malformed.

| game | plans | steps | step_acc | plan lens | aborts | abort steps | survived | deadlocks | fallbacks |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| ft09 | 40 | 40 | 1.000 | 1x40 | 0 | - | 0 | 0 | 0 |
| lf52 | 101 | 101 | 0.752 | 1x101 | 25 | 0x25 | 25 | 0 | 25 |
| lp85 | 138 | 138 | 0.087 | 1x138 | 126 | 0x126 | 126 | 0 | 126 |
| ls20 | 28 | 54 | 0.759 | 1x15,2x2,3x4,4x1,5x5,6x1 | 13 | 0x11,2x1,4x1 | 13 | 0 | 13 |
| s5i5 | 47 | 47 | 0.170 | 1x45,3x1,14x1 | 39 | 0x39 | 39 | 0 | 39 |
| sb26 | 51 | 53 | 0.377 | 1x29,2x5,3x2,4x2,6x2,7x1,8x4,10x3,.. | 33 | 0x31,1x2 | 33 | 0 | 33 |
| sp80 | 60 | 67 | 0.179 | 1x29,2x7,3x6,4x3,5x4,6x5,7x1,8x1,.. | 55 | 0x55 | 55 | 0 | 55 |
| su15 | 26 | 26 | 0.808 | 1x26 | 5 | 0x5 | 5 | 0 | 5 |
| tn36 | 44 | 122 | 0.984 | 1x23,2x3,3x2,4x3,5x8,6x2,7x1,8x2 | 2 | 0x2 | 2 | 0 | 2 |
| tr87 | 107 | 159 | 0.824 | 1x97,2x2,3x2,5x1,10x1,11x2,18x1,20x1 | 28 | 0x25,1x1,4x1,12x1 | 28 | 0 | 28 |
| tu93 | 136 | 309 | 0.997 | 1x124,2x2,3x1,4x1,5x1,9x1,11x1,14x2,.. | 1 | 0x1 | 1 | 0 | 1 |
| vc33 | 163 | 163 | 0.368 | 1x163 | 103 | 0x103 | 103 | 0 | 103 |

Shadow stats (ALL recorded steps verified, not just pre-abort):

| game | held-out state_exact% | steps | exact | shadow acc | first divergence (step#) | med/max diff cells on mismatch | selfdiff | sim_error | done-flag agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft09 | 100.0 | 40 | 40 | 1.000 | - | - | 0 | 0 | 40/40 |
| lf52 | 100.0 | 101 | 76 | 0.752 | 0 | 56/206 | 0 | 0 | 101/101 |
| lp85 | 100.0 | 138 | 12 | 0.087 | 5 | 81/1496 | 0 | 0 | 137/138 |
| ls20 | 100.0 | 66 | 53 | 0.803 | 14 | 8/16 | 0 | 0 | 66/66 |
| s5i5 | 99.5 | 62 | 8 | 0.129 | 2 | 8/872 | 0 | 0 | 61/62 |
| sb26 | 100.0 | 173 | 28 | 0.162 | 0 | 20/754 | 0 | 0 | 172/173 |
| sp80 | 100.0 | 180 | 12 | 0.067 | 12 | 2/1168 | 0 | 0 | 179/180 |
| su15 | 99.5 | 26 | 21 | 0.808 | 0 | 2/18 | 0 | 0 | 26/26 |
| tn36 | 100.0 | 122 | 120 | 0.984 | 57 | 3/3 | 0 | 0 | 122/122 |
| tr87 | 100.0 | 182 | 149 | 0.819 | 2 | 13/16 | 0 | 0 | 182/182 |
| tu93 | 100.0 | 309 | 308 | 0.997 | 295 | 770/770 | 0 | 0 | 308/309 |
| vc33 | 99.5 | 163 | 60 | 0.368 | 1 | 140/2870 | 0 | 0 | 161/163 |

`EWM_CANARY games_fired=12 threshold=5 verdict=PASS fired=[ft09,lf52,lp85,ls20,s5i5,sb26,sp80,su15,tn36,tr87,tu93,vc33]`
`EWM_ACTIVATION plans=941 steps=1279 outcomes=941 deadlocks=0 verdict=ACTIVE`

## gpt56_full

Log: 1251 EWMEVT lines, 121,341 bytes, longest line 110 chars, 0 malformed.

| game | plans | steps | step_acc | plan lens | aborts | abort steps | survived | deadlocks | fallbacks |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| ft09 | 11 | 17 | 0.412 | 3x1,4x2,7x1,9x2,10x1,12x2,14x1,16x1 | 10 | 0x9,3x1 | 10 | 0 | 10 |
| lp85 | 61 | 66 | 0.106 | 1x55,4x1,5x2,6x1,8x1,17x1 | 59 | 0x58,5x1 | 59 | 0 | 59 |
| sb26 | 99 | 99 | 0.030 | 1x98,2x1 | 96 | 0x96 | 96 | 0 | 96 |
| su15 | 50 | 51 | 0.176 | 1x48,2x1,3x1 | 42 | 0x42 | 42 | 0 | 42 |
| vc33 | 100 | 100 | 0.310 | 1x100 | 69 | 0x69 | 69 | 0 | 69 |

Shadow stats (ALL recorded steps verified, not just pre-abort):

| game | held-out state_exact% | steps | exact | shadow acc | first divergence (step#) | med/max diff cells on mismatch | selfdiff | sim_error | done-flag agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft09 | 100.0 | 100 | 7 | 0.070 | 7 | 37/3560 | 0 | 0 | 95/100 |
| lp85 | 100.0 | 100 | 7 | 0.070 | 6 | 80/2367 | 0 | 0 | 96/100 |
| sb26 | 100.0 | 100 | 3 | 0.030 | 0 | 40/793 | 0 | 0 | 95/100 |
| su15 | 99.5 | 53 | 9 | 0.170 | 1 | 13/301 | 0 | 0 | 52/53 |
| vc33 | 99.5 | 100 | 31 | 0.310 | 2 | 51/2884 | 0 | 0 | 97/100 |

`EWM_CANARY games_fired=5 threshold=5 verdict=PASS fired=[ft09,lp85,sb26,su15,vc33]`
`EWM_ACTIVATION plans=321 steps=333 outcomes=321 deadlocks=0 verdict=ACTIVE`
