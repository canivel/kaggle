# How We Beat FORGE v16

## Their Bugs We Can Exploit
1. **State hash truncation (64 bits)** - collisions at 500K states. We use full MD5 (128 bits)
2. **`use_counter_priority` hardcoded False** - dead code, never uses lexicographic A*
3. **Transfer only prev level** - we transfer from ALL solved levels
4. **CNN resets every level** - we keep model weights (proven to help)
5. **No ACTION7 (undo) in BFS** - we include undo
6. **Stride-2 misses 1x1 odd-coord sprites** - we use stride-1 on non-bg pixels
7. **Click scan no dedup** - they removed dedup (hurt cd82/sp80) but we can be smarter
8. **`int(1.5)=1` transfer multiplier bug** - we fix to actual 1.5x
9. **ACMD only fires at <100 states** - we run it always as fallback
10. **Background detection by frequency** - we detect by position (edges/corners)

## Our Innovations to Add
1. **Evolved greedy_novelty fallback** (our 0.10 agent) when BFS+CNN both fail
2. **Cross-level knowledge graph** persisted across levels
3. **KAOS-evolved parameters** for CNN fallback (random_explore=0.52, edge clicking)
4. **Multi-level transfer** (try ALL previous solutions, not just prev level)
5. **Parallel BFS + CNN** - start CNN exploring while BFS runs
6. **Warm start CNN from BFS data** - feed BFS-discovered transitions to CNN
