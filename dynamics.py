import numpy as np
from scipy.ndimage import label
import random
from mt_class import Mitochondria
from collections import deque
import initialise_and_visualise as iv


def fusion(cell, individual_mt, a, b, condition="healthy"):
    """
    Translate the whole labeled group `b` by exactly ONE pixel toward `a`
    (8-neighbour move). No teleporting, no overlaps. On contact (8-neighbour),
    groups become one connected component; we then do a light exchange.

    Uses _paint(...) and _free_neighbors(...).

    Returns:
        (cell, individual_mt)
    """
    import numpy as np
    from scipy.ndimage import label as _label

    # ---- helpers -------------------------------------
    def _coords_of(label_img, lbl):
        rr, cc = np.where(label_img == lbl)
        return list(zip(rr.tolist(), cc.tolist()))

    # 8-neighbour directions
    STEPS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
             (1, 0), (1, -1), (0, -1), (-1, -1)]
    # --------------------------------------------------
    structure = np.ones((3, 3), dtype=int)

    # Label current frame
    labeled, num = _label(cell, structure=structure)
    if a < 1 or b < 1 or a > num or b > num:
        return cell, individual_mt

    coords_a = _coords_of(labeled, a)
    coords_b = _coords_of(labeled, b)
    if not coords_a or not coords_b:
        return cell, individual_mt

    ca = np.array(coords_a, dtype=int)
    cb = np.array(coords_b, dtype=int)
    centroid_b = cb.mean(axis=0)

    # nearest A pixel to B's centroid
    diffs = ca - centroid_b
    d2 = (diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
    nearest_a = ca[int(np.argmin(d2))]
    base_vec = nearest_a - centroid_b
    base_d2 = float(base_vec[0] ** 2 + base_vec[1] ** 2)

    def step_score(dr, dc):
        trial_centroid = centroid_b + np.array([dr, dc], dtype=float)
        vec = nearest_a - trial_centroid
        return float(vec[0] ** 2 + vec[1] ** 2)

    # Sort candidate steps by how much they reduce distance to A
    candidate_steps = sorted(STEPS, key=lambda d: step_score(d[0], d[1]))

    nrows, ncols = cell.shape
    set_b = set(coords_b)

    # Feasible = stays in bounds and does NOT collide with any 1 (incl. A)
    def feasible(dr, dc):
        for (r, c) in set_b:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                return False
            if (nr, nc) not in set_b and cell[nr, nc] == 1:
                return False
        return True

    # Choose best feasible one-pixel step
    chosen = None
    best_d2 = None
    for (dr, dc) in candidate_steps:
        if feasible(dr, dc):
            d2_step = step_score(dr, dc)
            if best_d2 is None or d2_step < best_d2:
                best_d2 = d2_step
                chosen = (dr, dc)
            if d2_step < base_d2:
                chosen = (dr, dc)
                break

    if chosen is None:
        # Nowhere legal to move this tick
        return cell, individual_mt

    dr, dc = chosen

    # --- move the whole B body by exactly one pixel -------------------------
    # 1) clear old B pixels
    _paint(cell, coords_b, 0)   # adjust to 'value=' if your helper uses that

    # 2) compute new coordinates
    new_coords_b = [(r + dr, c + dc) for (r, c) in coords_b]

    # 3) paint new B pixels
    _paint(cell, new_coords_b, 1)

    # 4) shift identities (avoid dict.pop(...) with 3 args)
    moved = {}
    for (r, c) in coords_b:
        mt = individual_mt.get((r, c))
        if mt is not None:
            try:
                del individual_mt[(r, c)]
            except KeyError:
                pass
            dest = (r + dr, c + dc)
            # shouldn't exist thanks to feasible()
            if dest not in individual_mt:
                moved[dest] = mt
    individual_mt.update(moved)

    # --- free-neighbour query for B's new frontier -----------
    # NOTE: _free_neighbors signature requires 'needed'
    try:
        frontier_zeroes = _free_neighbors(cell, new_coords_b, needed=1)
        _ = frontier_zeroes  # kept for debugging
    except TypeError:
        frontier_zeroes = []

    # --- detect if A and (moved) B are now a single component ----------------
    labeled2, _ = _label(cell, structure=structure)
    # label of the moved B (take any pixel)
    b_lbl2 = labeled2[new_coords_b[0][0], new_coords_b[0][1]]
    merged = any(labeled2[r, c] == b_lbl2 for (r, c) in coords_a)

    if merged:
        rep_a = None
        for (r, c) in coords_a:
            rep_a = individual_mt.get((r, c))
            if rep_a is None:
                break
        rep_b = None
        for (r, c) in new_coords_b:
            rep_b = individual_mt.get((r, c))
            if rep_b is None:
                break
        if rep_a is not None and rep_b is not None:
            try:
                rep_a.fuse_exchange(rep_b, [rep_b], condition)
                rep_b.fuse_exchange(rep_a, [rep_a], condition)
            except Exception:
                pass

    return cell, individual_mt

def fission(cell, individual_mt, a, preferred_coord=None):
    """
    Split one mitochondrion into two by separating an edge position.
    Resources (mtdna, mt_material) are divided proportionally by size.
    
    Args:
        cell (np.ndarray): grid of 0/1
        individual_mt (dict): mapping (x,y) -> Mitochondria object
        a (int): id of mitochondrion to undergo fission
        preferred_coord (tuple): optional (x,y) to try to split off (must be edge)
    """
    cell = cell.copy()
    structure = np.ones((3, 3), dtype=int)
    labeled, num = label(cell, structure)

    # Validate
    if a < 1 or a > num:
        print(f"Invalid mitochondrion ID for fission: {a}")
        return cell, None, None

    # Get coordinates of mitochondrion a
    coords_a = np.argwhere(labeled == a)
    # Total mtDNA before split
    total_mtdna = sum(individual_mt[tuple(c)].mtdna for c in coords_a if tuple(c) in individual_mt)
    
    if len(coords_a) <= 1:
        print(f"Cannot fission: mitochondrion {a} is size 1")
        return cell, None, None

    # Find edge positions (positions with fewer neighbors)
    edge_pos = []
    for x, y in coords_a:
        neighbor_count = 0
        # Check 8-neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < cell.shape[0] and 0 <= ny < cell.shape[1] and 
                    cell[nx, ny] == 1 and labeled[nx, ny] == a):
                    neighbor_count += 1
        
        # Edge positions have fewer than maximum neighbors
        if neighbor_count <= 4:  # Not fully connected
            edge_pos.append((x, y))

    if not edge_pos:
        print("No suitable edge positions found for fission")
        return cell, None, None
    # if preferred_coord is given, check it's valid
    if preferred_coord is not None:
        pc = tuple(preferred_coord)
        if pc in edge_pos:
            split_x, split_y = pc
        else:
            split_x, split_y = random.choice(edge_pos)
    else:
        # Pick an edge pixel to split off
        split_x, split_y = random.choice(edge_pos)
    
    if (split_x, split_y) not in individual_mt:
        print(f"Split position {(split_x, split_y)} not in individual_mt")
        return cell, None, None
        
    split_mt = individual_mt[(split_x, split_y)]
    # Remove from main body
    cell[split_x, split_y] = 0
    del individual_mt[(split_x, split_y)]

    # Find a nearby free spot (within 2-3 steps but not touching)
    possible_positions = []
    
    # Generate positions in concentric circles around split position
    for distance in range(2, 5):  # Distance 2, 3, 4
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                if abs(dx) + abs(dy) == distance or max(abs(dx), abs(dy)) == distance:
                    new_x, new_y = split_x + dx, split_y + dy
                    # Check bounds
                    if not (0 <= new_x < cell.shape[0] and 0 <= new_y < cell.shape[1]):
                        continue
                    # Check if position is empty
                    if cell[new_x, new_y] != 0:
                        continue
                    # Check that it's not adjacent to any existing mitochondria
                    has_neighbor = False
                    for ndx in [-1, 0, 1]:
                        for ndy in [-1, 0, 1]:
                            if ndx == 0 and ndy == 0:
                                continue
                            check_x, check_y = new_x + ndx, new_y + ndy
                            if (0 <= check_x < cell.shape[0] and 0 <= check_y < cell.shape[1] and 
                                cell[check_x, check_y] == 1):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break
                    if not has_neighbor:
                        possible_positions.append((new_x, new_y))

    if not possible_positions:
        # If no isolated position found, place it back and abort
        cell[split_x, split_y] = 1
        individual_mt[(split_x, split_y)] = split_mt
        print("No valid isolated position found for fission product")
        return cell, None, None

    # Choose the closest valid position
    best_pos = min(possible_positions, 
                   key=lambda pos: (pos[0] - split_x)**2 + (pos[1] - split_y)**2)
    new_x, new_y = best_pos

    # Divide resources proportionally
    parent_size = len(coords_a)  # Original size before split
    split_size = 1

    # Place the split mitochondrion
    cell[new_x, new_y] = 1
    split_mt.x_coord, split_mt.y_coord = new_x, new_y
    individual_mt[(new_x, new_y)] = split_mt

    # Proportional division of mtDNA
    if total_mtdna > 0:
        mtdna_split = max(1, int(total_mtdna * split_size / parent_size))
        mtdna_main_body  = max(1, total_mtdna - mtdna_split)
        # CHECK
        if mtdna_main_body + mtdna_split != total_mtdna:
            print(f"mtDNA lost in fission: total {total_mtdna}, main {mtdna_main_body}, split {mtdna_split}")
        else:
            # Update material of split mitochondrion
            split_mt.mtdna = mtdna_split
            split_mt.exchangeable()
            # Update main body mitochondria
            remaining_coords = [tuple(c) for c in coords_a if tuple(c) != (split_x, split_y)]
            for coord in remaining_coords:
                if tuple(coord) in individual_mt:
                    individual_mt[tuple(coord)].mtdna = mtdna_main_body
                    individual_mt[tuple(coord)].exchangeable()   

    # Check health conditions
    def get_condition(mt):
        if mt.mtdna < 10: 
            return "mitophagy"
        elif 10 < mt.mtdna < 90:
            return "sick"
        else:
            return "healthy"

    cond_split = get_condition(split_mt)
    if cond_split == "mitophagy":
        cell[new_x, new_y] = 0
        del individual_mt[(new_x, new_y)]
        print(f"Split mitochondrion removed by mitophagy (mtdna={split_mt.mtdna})")

    # Check main body condition
    cond_main = None
    if remaining_coords and remaining_coords[0] in individual_mt:
        cond_main = get_condition(individual_mt[remaining_coords[0]])
        if cond_main == "mitophagy":
            for coord in remaining_coords:
                if coord in individual_mt:
                    cell[coord] = 0
                    del individual_mt[coord]
            print(f"Main body removed mitophagy")

    return cell, cond_main, cond_split

def hyperfusion(cell, individual_mt, step, ax1, ax2, ax3, ax4, ros, atp, resources,
                clearance_boost=2,
                max_clearance_cap=8,
                animate_each_step=True,
                animate_pause=0.05,
                render_every=3,
                # === Biophysical scaling knobs ===
                k_ros=0.05,          # scales ROS drop per growth step
                k_atp=0.05,          # scales ATP gain per growth step
                min_ros_drop=1,      # ensure at least this many ROS removed if ros>0
                max_atp_gain_step=None  # cap per-step ATP gain (None = no cap)
                ):
    """
    Hyperfusion with 8-neighbour connectivity and biophysically grounded ROS/ATP updates.

    On each *adding* step (0->1):
      ROS drop  ~ k_ros * total_clearance * compactness
      ATP gain  ~ k_atp * area * compactness

    Where:
      total_clearance = sum(mt.ros_clearance)
      compactness     = area / perimeter_8nbr
      area            = total count of 1s
    """
    from scipy.ndimage import label
    import matplotlib.pyplot as plt
    from dynamics import fuse_via_path
    import initialise_and_visualise as iv  # keep your existing visualiser import

    initial_count = int(np.sum(cell))
    ros_val = ros
    atp_val = atp

    structure = np.ones((3,3), dtype=int)
    labeled, n_groups = label(cell, structure)

    count = 0
    plt.ion()

    def _render(tag=None):
        try:
            if ax1 is not None:
                ax1.set_title(f"Hyperfusion – {tag or ''}")
            iv.visualise_step(
                cell, individual_mt, ax1, ax2, ax3, ax4, step,
                ros_val,     
                atp_val,    
                resources)
            if animate_each_step and animate_pause:
                import matplotlib.pyplot as plt
                plt.pause(animate_pause)
        except Exception as e:
            print(f"[hyperfusion] visualisation skipped: {e}")

    safety = 0
    while True:
        labeled, n_groups = label(cell, structure)
        if n_groups <= 1:
            break

        # build all pairs, closest-first (Chebyshev)
        centroids = []
        for gid in range(1, n_groups+1):
            coords = np.argwhere(labeled == gid)
            if len(coords):
                centroids.append((gid, np.mean(coords, axis=0)))
        pairs = []
        for i in range(len(centroids)):
            gi, ci = centroids[i]
            for j in range(i+1, len(centroids)):
                gj, cj = centroids[j]
                d = max(abs(ci[0]-cj[0]), abs(ci[1]-cj[1]))
                pairs.append((d, gi, gj))
        pairs.sort(key=lambda x: x[0])

        progressed = False

        for _, a, b in pairs[:min(8, len(pairs))]:  # try a few nearest pairs
            def _step_hook(adding: bool):
                nonlocal ros_val, atp_val
                if not adding:
                    return
                comp, area, perim = _compactness(cell)
                tot_clear = _total_clearance(individual_mt)
                ros_drop = k_ros * tot_clear * comp
                if ros_val > 0:
                    ros_drop = max(min_ros_drop, ros_drop)
                ros_val = max(0.0, ros_val - ros_drop)
                atp_gain = k_atp * area * comp
                if max_atp_gain_step is not None:
                    atp_gain = min(atp_gain, max_atp_gain_step)
                atp_val += atp_gain

            cell, individual_mt, merged, changed = fuse_via_path(
                cell, individual_mt, a, b,
                render=(lambda tag: _render(tag)) if animate_each_step else None,
                step_hook=_step_hook,
                render_every=render_every,
                max_microsteps=4*cell.size
            )
            count += 1
            # relabel and check
            labeled2, n2 = label(cell, structure)
            if n2 < n_groups or merged or changed:
                progressed = True
                break  # go to next outer round

        if not progressed:
            print("[hyperfusion] no progress this round; applying single-pixel fallback")
            # single-pixel fallback (same as before, borrows from larger group)
            d, a, b = pairs[0]
            coords_a = np.argwhere(labeled == a); coords_b = np.argwhere(labeled == b)
            pa = coords_a[np.argmin([max(abs(p[0]-coords_b[0][0]), abs(p[1]-coords_b[0][1])) for p in coords_a])]
            pb = coords_b[0]
            r = int(round((pa[0] + pb[0]) / 2)); c = int(round((pa[1] + pb[1]) / 2))
            if 0 <= r < cell.shape[0] and 0 <= c < cell.shape[1] and cell[r, c] == 0:
                size_a = (labeled == a).sum(); size_b = (labeled == b).sum()
                donor_label = a if size_a >= size_b else b
                donor_coords = np.argwhere(labeled == donor_label)
                donor = max((tuple(p) for p in donor_coords),
                            key=lambda p: max(abs(p[0]-r), abs(p[1]-c)))
                obj = individual_mt.pop(donor, None)
                cell[donor] = 0
                if obj is not None:
                    obj.x_coord, obj.y_coord = (r, c)
                    individual_mt[(r, c)] = obj
                cell[r, c] = 1
                if animate_each_step:
                    _render("fallback link")

        safety += 1
        if safety > 200:   # lower cap since each round is heavier now
            print("[hyperfusion] safety break (outer loop)"); break

    # fused-state physiology tweak (same as before)
    for mt in individual_mt.values():
        if hasattr(mt, "ros_clearance"):
            mt.ros_clearance = min(max_clearance_cap, mt.ros_clearance + clearance_boost)
        if hasattr(mt, "ros"):
            mt.ros = max(0, int(mt.ros * 0.5))

    _render("complete")

    # sanity
    new_count = int(np.sum(cell))
    if new_count != initial_count:
        print(f"[hyperfusion WARNING] count changed {initial_count} -> {new_count}")

    # IMPORTANT: return updated totals so the caller continues with new state
    return cell, individual_mt, ros_val, atp_val, count


####  ------------- HELPER FUNCTIONS FOR VISUALISATION -------------  ####
def _paint(cell, positions, value):
    if not positions:
        return
    # Force positions -> int ndarray of shape (N, 2)
    arr = np.asarray(list(positions), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"_paint expected (N,2) coords, got shape {arr.shape}")
    rr, cc = arr[:, 0], arr[:, 1]
    cell[rr, cc] = int(value)

def _free_neighbors(cell, start, needed):
    """
    BFS from `start` to collect free in-bounds cells (0s).
    Returns a list of coordinates (len == needed). Guaranteed not
    to touch any currently-occupied pixel.
    """
    H, W = cell.shape
    q = deque([start])
    seen = {start}
    result = []
    while q and len(result) < needed:
        r, c = q.popleft()
        if 0 <= r < H and 0 <= c < W and cell[r, c] == 0 and (r, c) not in result:
            result.append((r, c))
        # 4-neighborhood; add diagonals if you prefer
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return result if len(result) == needed else None

def coords_from_gid(labels, gid):
    rr, cc = np.where(labels == gid)
    return list(zip(rr.tolist(), cc.tolist()))

def fuse_via_path(cell, individual_mt, a, b, render=None, step=None, viz_args=None,
                  step_hook=None, render_every=3, max_microsteps=None):
    """
    8-neighbour path-based fusion with strict pixel conservation + safety:
      - Trim from B when adding bridge pixels; never trim GOAL.
      - If B has nothing left to trim (except GOAL), borrow from A's far side.
      - Throttled rendering (every `render_every` steps).
      - Hard cap on micro-steps to avoid UI 'freezes'.
    Returns
    -------
    cell, individual_mt, merged_bool, changed_bool
    """
    import numpy as np
    from scipy.ndimage import label
    from collections import deque

    H, W = cell.shape
    if max_microsteps is None:
        max_microsteps = 4 * H * W  # generous but finite

    structure = np.ones((3, 3), dtype=int)  # 8-neighbour
    labeled, num = label(cell, structure)
    if a < 1 or b < 1 or a > num or b > num:
        print(f"Invalid mitochondrion IDs: a={a}, b={b}, max={num}")
        return cell, individual_mt, False, False

    coords_a = np.argwhere(labeled == a)
    coords_b = np.argwhere(labeled == b)
    if len(coords_a) == 0 or len(coords_b) == 0:
        return cell, individual_mt, False, False

    A = {tuple(p) for p in map(tuple, coords_a)}
    B = {tuple(p) for p in map(tuple, coords_b)}
    count_before = int(np.sum(cell))

    NEIGH8 = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))

    def boundary(S):
        out = set()
        for r, c in S:
            for dr, dc in NEIGH8:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in S and cell[nr, nc] == 0:
                    out.add((r, c)); break
        return out

    Ab = list(boundary(A))
    Bb = list(boundary(B))
    if not Ab or not Bb:
        return cell, individual_mt, False, False

    # Chebyshev-closest boundary pair
    best, bestd = None, 1e9
    for pa in Ab:
        for pb in Bb:
            d = max(abs(pa[0]-pb[0]), abs(pa[1]-pb[1]))
            if d < bestd:
                bestd, best = d, (pa, pb)
    start, goal = best

    # 8-neighbour BFS through empty cells (goal allowed)
    q = deque([start]); prev = {start: None}; seen = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == goal: break
        for dr, dc in NEIGH8:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen:
                if cell[nr, nc] == 0 or (nr, nc) == goal:
                    seen.add((nr, nc)); prev[(nr, nc)] = (r, c); q.append((nr, nc))

    # Path excludes start, includes goal if reachable; otherwise single empty neighbour
    path = []
    if goal in prev:
        cur = goal
        while cur is not None and cur != start:
            path.append(cur); cur = prev[cur]
        path.reverse()
    else:
        for r, c in Ab:
            for dr, dc in NEIGH8:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and cell[nr, nc] == 0:
                    path = [(nr, nc)]
                    break
            if path: break

    # Trimming pools (exclude goal for B; exclude start for A)
    def cheb_from_start(p):
        return max(abs(p[0]-start[0]), abs(p[1]-start[1]))

    B_no_goal = {p for p in B if p != goal}
    B_ordered = sorted(B_no_goal, key=cheb_from_start, reverse=True)
    A_no_start = {p for p in A if p != start}
    A_ordered = sorted(A_no_start, key=cheb_from_start, reverse=True)

    merged = False
    changed = False
    micro = 0

    for k, px in enumerate(path, start=1):
        micro += 1
        if micro > max_microsteps:
            print("[fuse_via_path] micro-step cap reached; bailing safely")
            break

        on_B = (px in B)          # landing directly on B (already 1)
        adding = (cell[px] == 0)  # will this step add a new 1?

        if adding:
            # refresh pools
            while B_ordered and B_ordered[0] not in B_no_goal:
                B_ordered.pop(0)
            while A_ordered and A_ordered[0] not in A_no_start:
                A_ordered.pop(0)

            # choose trim source
            if B_ordered:
                side, tail = "B", B_ordered.pop(0)
                B_no_goal.discard(tail); B.discard(tail)
            elif A_ordered:
                side, tail = "A", A_ordered.pop(0)
                A_no_start.discard(tail); A.discard(tail)
            else:
                # no place to trim/borrow → cannot add; stop path here
                break

            # add + trim (move identity)
            cell[px] = 1
            A.add(px); A_no_start.add(px)
            obj = individual_mt.pop(tail, None)
            if obj is not None:
                obj.x_coord, obj.y_coord = px
                individual_mt[px] = obj
            cell[tail] = 0
            changed = True

        # step hook (biology update) before drawing
        if step_hook is not None:
            try:
                step_hook(adding)
            except Exception:
                pass

        # throttle rendering
        if render is not None and (k % max(1, int(render_every)) == 0 or on_B or k == len(path)):
            try:
                render(f"merge-path step {k}/{len(path)}")
            except Exception:
                pass

        if on_B:
            merged = True  # connection achieved

    # Extra check: if we didn’t explicitly hit on_B, verify connectivity
    if not merged and changed:
        labeled2, n2 = label(cell, structure)
        # pick any pixel from original sets to compare labels
        la = labeled2[tuple(coords_a[0])]
        lb = labeled2[tuple(coords_b[0])]
        merged = (la == lb)

    # Sanity: pixel count conserved
    count_after = int(np.sum(cell))
    if count_after != count_before:
        print(f"Count mismatch after fusion: before={count_before}, after={count_after}")

    return cell, individual_mt, merged, changed

def _neighbours(cell: np.ndarray) -> int:
    """8-neighbour perimeter: count pixels that touch at least one empty 8-neighbour."""
    H, W = cell.shape
    NEIGH8 = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
    perim = 0
    ones = np.argwhere(cell == 1)
    for r, c in ones:
        for dr, dc in NEIGH8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or cell[nr, nc] == 0:
                perim += 1
                break
    return perim

def _compactness(cell: np.ndarray) -> float:
    """
    Simple compactness proxy in [0, ~inf):
      comp = area / max(1, perimeter)
    Higher = more compact (more reticular), lower = stringy/fragmented.
    """
    area = int(np.sum(cell))
    perim = _neighbours(cell)
    return (area / max(1, perim)), area, perim

def _total_clearance(individual_mt: dict) -> float:
    """Sum of per-mitochondrion ROS clearance parameters (missing -> 0)."""
    tot = 0.0
    for mt in individual_mt.values():
        val = getattr(mt, "ros_clearance", 0.0)
        try:
            tot += float(val)
        except Exception:
            pass
    return tot

