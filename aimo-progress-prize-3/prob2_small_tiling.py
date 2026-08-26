"""
Verify tiling exists for small N cases.
"""

def can_tile(N, rectangles):
    """
    Check if a list of (w, h) rectangles can tile an NxN grid.
    Uses backtracking.
    """
    grid = [[False]*N for _ in range(N)]
    rects = list(rectangles)

    def find_first_empty():
        for y in range(N):
            for x in range(N):
                if not grid[y][x]:
                    return (x, y)
        return None

    def place(x, y, w, h):
        if x + w > N or y + h > N:
            return False
        for dy in range(h):
            for dx in range(w):
                if grid[y+dy][x+dx]:
                    return False
        for dy in range(h):
            for dx in range(w):
                grid[y+dy][x+dx] = True
        return True

    def unplace(x, y, w, h):
        for dy in range(h):
            for dx in range(w):
                grid[y+dy][x+dx] = False

    def solve(remaining_indices):
        pos = find_first_empty()
        if pos is None:
            return len(remaining_indices) == 0

        x, y = pos
        for i in remaining_indices:
            w, h = rects[i]
            new_remaining = [j for j in remaining_indices if j != i]

            # Try (w, h)
            if place(x, y, w, h):
                if solve(new_remaining):
                    return True
                unplace(x, y, w, h)

            # Try (h, w) if different
            if w != h:
                if place(x, y, h, w):
                    if solve(new_remaining):
                        return True
                    unplace(x, y, h, w)

        return False

    return solve(list(range(len(rects))))


# N=3, k=3: half-perimeters (2,3,5)
# s=2: 1x1. s=3: 1x2. s=5: need area to make total = 9.
# 1+2+area5=9 => area5=6. s=5: 2x3 (area 6). ✓
N = 3
rects = [(1,1), (1,2), (2,3)]
print(f"N=3, rects={rects}: {can_tile(N, rects)}")

# N=5, k=6: half-perimeters 2,3,4,5,6,7 (or similar)
# min_areas: 1+2+3+4+5+... let me find valid combo
N = 5
from itertools import combinations

for combo in combinations(range(2, 2*N+1), 6):
    def min_area(s):
        if s <= N+1:
            return s-1
        else:
            return (s-N)*N
    def max_area(s):
        a_lo = max(1, s-N)
        a_hi = min(s-1, N)
        a = min(s//2, a_hi)
        a = max(a, a_lo)
        return a*(s-a)

    tmin = sum(min_area(s) for s in combo)
    tmax = sum(max_area(s) for s in combo)
    if tmin <= N*N <= tmax:
        # Found valid combo, now find specific rectangles
        # Use min area for each, adjust one to meet total
        areas_min = [min_area(s) for s in combo]
        gap = N*N - sum(areas_min)

        rects = []
        for s in combo:
            if s <= N+1:
                rects.append((1, s-1))
            else:
                rects.append((s-N, N))

        # Adjust last rectangle to fill gap
        # For the last s, try different a values
        s_last = combo[-1]
        target_area = min_area(s_last) + gap
        found = False
        for a in range(max(1, s_last-N), min(s_last, N+1)):
            b = s_last - a
            if 1 <= b <= N and a*b == target_area:
                rects[-1] = (a, b)
                found = True
                break

        if found:
            total_area = sum(w*h for w,h in rects)
            if total_area == N*N:
                print(f"\nN=5, combo={combo}")
                print(f"Rectangles: {rects}, total_area={total_area}")
                result = can_tile(N, rects)
                print(f"Can tile: {result}")
                if result:
                    break

# N=4, k=4
N = 4
for combo in combinations(range(2, 2*N+1), 4):
    def min_area4(s):
        if s <= N+1:
            return s-1
        else:
            return (s-N)*N
    def max_area4(s):
        a_lo = max(1, s-N)
        a_hi = min(s-1, N)
        a = min(s//2, a_hi)
        a = max(a, a_lo)
        return a*(s-a)

    tmin = sum(min_area4(s) for s in combo)
    tmax = sum(max_area4(s) for s in combo)
    if tmin <= N*N <= tmax:
        areas_min = [min_area4(s) for s in combo]
        gap = N*N - sum(areas_min)

        rects = []
        for s in combo:
            if s <= N+1:
                rects.append((1, s-1))
            else:
                rects.append((s-N, N))

        # Try adjusting each rectangle
        for idx in range(len(combo)):
            s = combo[idx]
            target = areas_min[idx] + gap
            for a in range(max(1, s-N), min(s, N+1)):
                b = s - a
                if 1 <= b <= N and a*b == target:
                    rects_try = list(rects)
                    rects_try[idx] = (a, b)
                    total = sum(w*h for w,h in rects_try)
                    if total == N*N:
                        print(f"\nN=4, combo={combo}")
                        print(f"Rectangles: {rects_try}, total_area={total}")
                        result = can_tile(N, rects_try)
                        print(f"Can tile: {result}")
                        if result:
                            break
            else:
                continue
            break
        else:
            continue
        break
