# ---------- Imports ------------ 
import numpy as np
import random
from scipy.ndimage import label
import matplotlib.pyplot as plt

# Other scripts to import:
import dynamics
from mt_class import Mitochondria
from math import exp
from mitochondria_groups import MitochondriaGroup
import initialise_and_visualise as iv

def estimate_local_payoffs(group, groups_dict, gamma=1.0):
    """
    Estimate expected payoff if this group chooses fusion vs fission.
    """
    if not group.members:
        return 0, 0
    size = group.size
    base_atp = group.group_atp()
    base_ros = group.group_ros_proxy()
    base_payoff = base_atp - gamma * base_ros

    # FISSION payoff estimate
    if size <= 1:
        payoff_fission = base_payoff  # cannot split further
    else:
        # main size = size-1, split size = 1
        atp_after = group.members[0].atp_production(size - 1) + group.members[0].atp_production(1)
        ros_after = int(0.5 * (size - 1)) + int(0.5 * 1)
        payoff_fission = atp_after - gamma * ros_after

    # FUSION payoff estimate: consider nearest neighbour
    nearest_gid = None
    nearest_dist = float("inf")
    for other_gid, other_group in groups_dict.items():
        if other_gid == group.gid:
            continue
        d = np.linalg.norm(group.centroid - other_group.centroid)
        if d < nearest_dist:
            nearest_dist = d
            nearest_gid = other_gid
    if nearest_gid is None:
        payoff_fusion = base_payoff
    else:
        other_group = groups_dict[nearest_gid]
        merged_size = size + other_group.size
        # merged ATP estimated as atp_production(merged_size) (single large group)
        atp_merged = group.members[0].atp_production(merged_size)
        ros_merged = int(0.5 * merged_size)
        payoff_fusion = atp_merged - gamma * ros_merged

    return payoff_fusion, payoff_fission

def replicator_logit_step(p, u_fus, u_fis, eta=0.2):
    # move in log-odds space; equivalent to noisy best response
    logit = np.log(p) - np.log(1 - p)
    logit += eta * (u_fus - u_fis)
    p_new = 1.0 / (1.0 + np.exp(-logit))
    return float(np.clip(p_new, 1e-3, 1-1e-3))

def update_group_strategy(group, groups_dict, eta=0.25, gamma=1.0):
    pf_pay, pfis_pay = estimate_local_payoffs(group, groups_dict, gamma=gamma)
    p0   = getattr(group, "p_fusion", 0.5)
    p1   = replicator_logit_step(p0, pf_pay, pfis_pay, eta)
    # persist to members so regrouping preserves state
    for m in group.members:
        m.p_fusion = p1
    group.p_fusion = p1
    return pf_pay, pfis_pay, p1

def _logit(p): return np.log(p) - np.log(1 - p)
def _sigm(z):  return 1.0 / (1.0 + np.exp(-z))

def nudge_group_p(group, beta):
    p = group.p_fusion
    p = _sigm(_logit(p) + beta)  # beta>0 favors fusion; beta<0 favors fission
    p = float(np.clip(p, 1e-3, 1 - 1e-3))
    for m in group.members: m.p_fusion = p
    group.p_fusion = p

def verify_consistency(cell, individual_mt, step_info=""):
    """
    Verify that cell matrix and individual_mt dictionary are consistent
    """
    cell_positions = set(tuple(pos) for pos in np.argwhere(cell == 1))
    dict_positions = set(individual_mt.keys()) 
    if cell_positions != dict_positions:
        print(f"CONSISTENCY ERROR {step_info}:")
        print(f"  Cell has {len(cell_positions)} positions")
        print(f"  Dict has {len(dict_positions)} positions")        
        # 1) Any pixel in the grid that lacks an object -> create one (healthy default)
        for pos in cell_positions - dict_positions:
            x, y = pos
            individual_mt[pos] = Mitochondria(x, y, ca=1, camp=1)  # lightweight default

        # 2) Any object that lacks a pixel -> restore its pixel
        for pos in dict_positions - cell_positions:
            if 0 <= pos[0] < cell.shape[0] and 0 <= pos[1] < cell.shape[1]:
                cell[pos] = 1
    return cell, True

def atp_and_ros_handling(individual_mt, groups, ca_pm, camp_pm, received_gluc:set):
    """ Calculate ROS and ATP for each step with error handling """
    if not individual_mt:
        return 0, 0, {"ca": 0, "camp": 0, "glucose": 0}
    # 1) Build a map to find group size (default 1)
    coord_to_size = {}
    for g in groups.values():
        s = g.size if hasattr(g, "size") else len(getattr(g, "coords", []))
        for c in getattr(g, "coords", []):
            coord_to_size[tuple(c)] = max(1, int(s))
    coord_to_gid = {}
    gid_to_group = {}
    for gid, g in groups.items():
        gid_to_group[gid] = g
        for c in getattr(g, "coords", []):
            coord_to_gid[tuple(c)] = gid
    step_ros = 0
    step_atp = 0
    consumed_ca = 0
    consumed_camp = 0
    try:
       # 2) Per-mito ROS + ATP with group size
        for coord, mt in individual_mt.items():
            # ROS: baseline intracellular + signalling induced
            if hasattr(mt, "baseline_metabolism_ros"):
                step_ros += mt.baseline_metabolism_ros()
            if camp_pm > 0 and hasattr(mt, "use_camp"):
                before = getattr(mt, "camp", 0)
                mt.camp = before + camp_pm
                step_ros += mt.use_camp(camp_pm)
                consumed_camp += max(0, before + camp_pm - mt.camp)
            if ca_pm > 0 and hasattr(mt, "use_ca"):
                before = getattr(mt, "ca", 0)
                mt.ca = before + ca_pm 
                step_ros += mt.use_ca(ca_pm)
                consumed_ca += max(0, before + ca_pm - mt.ca)
            # ROS from stress
            gsize = coord_to_size.get(coord, 1)
            if hasattr(mt, 'use_ros'):
                mt.use_ros(gsize, step_ros) 
            # Only do this immediately if it just took critical damage 
            if getattr(mt, "just_damaged", False) and gsize >= 2:
                gid = coord_to_gid.get(coord)
                # Check if damage was critical
                if getattr(mt, "critically_damaged", False) and gsize >= 2 and getattr(mt, "repair_cooldown", 0) == 0:
                    if 15 <= mt.mtdna < 60:
                        mt.queued_action = "exchange"
                        mt.queued_at_step = iv.current_step if hasattr(iv, "current_step") else 0  # optional
                        mt.queued_coord = coord
                    elif mt.mtdna < 15:
                        mt.queued_action = "bud"
                        mt.queued_at_step = iv.current_step if hasattr(iv, "current_step") else 0
                        mt.queued_coord = coord
            # ATP 
            gsize = coord_to_size.get(coord, 1)
            if coord in received_gluc and hasattr(mt, "atp_production"):
                step_atp += mt.atp_production(gsize)
    except Exception as e:
        print(f"Error in step calculation of ATP and ROS: {e}")
        step_ros = len(individual_mt)*2 # fallback
        step_atp = len(individual_mt)*30 # fallback
    return step_ros, step_atp, {"ca": consumed_ca, "camp": consumed_camp, "glucose": len(received_gluc)}

def resource_recovery(R:float, r:float, K:float, C:float):
    """ Logistic resource recovery after consumption """
    R_nxt = R + r*R*(K - R) - C
    if R_nxt < 0.0:
        return 0.0
    if R_nxt > K:
        return K
    return R_nxt

def glucose_allocation(individual_mt: dict, glucose_pool:int):
    """ Allocate 1 glucose per occupied pixel until pool exhasuted """
    if glucose_pool <= 0 or not individual_mt:
        return set(), 0
    coords = list(individual_mt.keys())
    random.shuffle(coords)
    give = min(glucose_pool, len(coords))
    fed = set(coords[:give])
    return fed, give 

def dynamic_operation(cell, individual_mt, operation_func, *args, **kwargs):
    """Perform operations (fission/fusion) with validation"""
    old_count = np.sum(cell)
    old_cell = cell.copy()
    try:
        result = operation_func(cell, individual_mt, *args, **kwargs)
        if isinstance(result, tuple):
            new_cell = result[0]
            new_individual_mt = result[1] if len(result) > 1 and isinstance(result[1], dict) else individual_mt
            rest = list(result[2:]) if len(result) > 2 else []
        else:
            new_cell = result
            new_individual_mt = individual_mt
            rest = []
        # Verify:
        new_cell, is_consistent = verify_consistency(new_cell, new_individual_mt, "after operation")
        new_count = np.sum(new_cell)
        # Allow for small count change by mitophagy
        if abs(new_count - old_count) > 3: # more than 3 mitochondria lost in 1 step
            print(f"WARNING: Large amount of mitochondria lost: {old_count} -> {new_count}")
        return (new_cell, new_individual_mt, *rest)
    except Exception as e:
        print(f"Operation failed: {e}")
        return (old_cell, individual_mt) 

def mitochondrial_dynamics(cell, ca, camp, glucose, max_steps=50, ros_hyperfusion_threshold=170, gamma=1.2, 
                           r_ca=0.03, r_camp=0.02, r_glucose=0.05,
                           Fusion="ON", Fission="ON"):
    """ Mitochondrial dynamics of a neural cell """
    step = 0
    initial_mt_count = int(np.sum(cell))
    print(f"Starting simulation with {initial_mt_count} mitochondria")

    # Early termination if no mitochondria to begin with
    if initial_mt_count == 0:
        return cell, [], []

    # Global resource pools & carrying capacity:
    pools = {"ca": float(ca), "camp": float(camp), "glucose": float(glucose)}
    K = {"ca": float(ca), "camp": float(camp), "glucose": float(glucose)}

    # initialise individual mitochondria 
    individual_mt = {}
    for x,y in np.argwhere(cell == 1):
        individual_mt[(x,y)] = Mitochondria(x, y, 0, 0)
    # Verify initial consistency 
    cell, _ = verify_consistency(cell, individual_mt, "initial")

    # Track for plotting
    overall_ros = []
    overall_atp = []
    resources = {"ca": [], "camp": [], "glucose": []}
    # Fission:Fusion ratio tracking
    fission_count = 0
    fusion_count = 0
  
    # Clsoe any existing plots and creat figure for real-time plotting 
    plt.close('all')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10))
    plt.ion() # interactive mode

    hyperfusion_mode = False 

    while step < max_steps:
        step += 1
        print(f"\n=== Step {step} ===")
        current_mt_count = np.sum(cell)
        print(f"Current mitochondria count: {current_mt_count}")
        if current_mt_count == 0:
            print("All mitochondria lost")
            break 

        # Checking the condition of the mitochondria: clean up/repair
        to_remove = []
        for coords, mt in list(individual_mt.items()):  
            if mt.mtdna < 10:
                # mitophagy
                to_remove.append(coords)  
                print(f"[step {step}] Mitochondrion at {coords} removed (mitophagy), mtDNA: {mt.mtdna})")
            elif 10 < mt.mtdna < 90:
                if hasattr(mt, 'p_fusion'):
                    mt.p_fusion = 0.99 # force fusion for repair
                print(f"Mitochondrion at {coords} needs repair (mtDNA: {mt.mtdna})")

        # Remove the flagged mitochondria from individual_mt
        for coords in to_remove:
            if coords in individual_mt:
                cell[coords] = 0
                del individual_mt[coords]

        # Re-verify after cleanup
        cell, _ = verify_consistency(cell, individual_mt, f"step {step} after cleanup")
        if len(individual_mt) == 0:
            print("All mitochondria removed.")
            break

        # Label groups and construct group objects after cleanup
        labeled, n_groups = label(cell, np.ones((3,3), dtype=int))
        if n_groups  == 0:
            break
        groups = {}
        for gid in range(1, n_groups+1):
            coords = np.argwhere(labeled == gid)
            if len(coords) > 0:
                groups[gid] = MitochondriaGroup(gid, coords, individual_mt)
        print(f"Found {len(groups)} fused mitochondrial groups")

        # === RESOURCE ALLOCATION ===
        n = len(individual_mt)
        ca_pm = int(pools["ca"]//n) if n > 0 else 0
        camp_pm = int(pools["camp"]//n) if n > 0 else 0
        glucose_given, glucose_consumed = glucose_allocation(individual_mt, int(pools["glucose"]))

        # === ATP & ROS ===
        step_ros, step_atp, consumed = atp_and_ros_handling(individual_mt, groups, ca_pm, camp_pm, glucose_given)

        if hyperfusion_mode and (step_ros <= (ros_hyperfusion_threshold - 50)):
            print(f"[step {step}] ROS {step_ros} <= {ros_hyperfusion_threshold - 50}: fission unlocked")
            hyperfusion_mode = False

        # === RESOURCE RECOVERY ===
        pools["ca"] = resource_recovery(pools["ca"], r_ca, K["ca"], consumed["ca"])
        pools["camp"] = resource_recovery(pools["camp"], r_camp, K["camp"], consumed["camp"])
        pools["glucose"] = resource_recovery(pools["glucose"], r_glucose, K["glucose"], glucose_consumed)

        # === UPDATE SERIES ===
        resources["ca"].append(pools["ca"])
        resources["camp"].append(pools["camp"])
        resources["glucose"].append(pools["glucose"])
        overall_ros.append(step_ros)
        overall_atp.append(step_atp)
        print(f"Step {step}: ROS = {step_ros}, ATP = {step_atp}")

        try:
            # Visualise Cell state
            iv.visualise_step(cell, individual_mt, ax1, ax2, ax3, ax4, step, overall_ros, overall_atp, resources)
        except Exception as e:
            print(f"Visualization error: {e}")
        # Execute queued repair actions first
        try:
            labeled_now, n_now = label(cell, np.ones((3,3), dtype=int))
            queued = [(coord, mt) for coord, mt in individual_mt.items() if getattr(mt, "queued_action", None) in ("exchange", "bud")]
            for coord, mt in queued:
                actions = mt.queued_action
                mt.queued_action = None
                mt.repair_cooldown = 2  # cooldown period
                if coord not in individual_mt:
                    continue  # it was removed
                g_size = 1
                gid_here = None
                if 0 <= coord[0] < labeled_now.shape[0] and 0 <= coord[1] < labeled_now.shape[1]:
                    gid_here = labeled_now[coord[0], coord[1]]
                    if gid_here > 0:
                        g_size = int(np.sum(labeled_now == gid_here))
                    if g_size < 2 or gid_here is None:
                        continue  # not in a group anymore
                coords_gid = np.argwhere(labeled_now == gid_here)
                members = []
                for (xx, yy) in coords_gid:
                    m = individual_mt.get((xx, yy))
                    if m is not None:
                        members.append(m)   
                if not members:
                    continue
                if actions == "exchange" and (15 <= mt.dna < 60):
                    me = coord
                    partner = None
                    bestd = 1e9
                    for other in group.members:
                        if (other.x_coord, other.y_coord) == me:
                            continue
                        d = max(abs(other.x_coord - me[0]), abs(other.y_coord - me[1]))
                        if d < bestd:
                            bestd = d
                            partner = other
                    # Two branches by mtDNA
                    if 15 <= mt.mtdna < 60 and partner is not None:
                        try:
                            mt.fuse_exchange(partner, members, condition="just_damaged")
                        except Exception as _:
                            pass
                elif actions == "bud" and mt.mtdna < 15:
                    # try to bud off THIS pixel from its fused group
                    try:
                        new_cell, cond_main, cond_split = dynamics.fission(
                            cell, individual_mt, gid_here, preferred_coord=coord)
                        fission_count += 1
                        # commit if it succeeded (return type matches your fission)
                        if isinstance(new_cell, np.ndarray):
                            cell[:] = new_cell  # in-place update of the grid
                            # relabel to find the new singletons/groups
                            labeled2, n2 = label(cell, np.ones((3,3), dtype=int))
                            # if the bud survived (not immediately mitophagy’ed), try to fuse for rescue
                            if cond_split != "mitophagy":
                                # Set fusion prob very high for the new singleton
                                new_gid = labeled2[coord[0], coord[1]]
                                if new_gid in groups:
                                    new_group = groups[new_gid]
                                    new_group.p_fusion = 0.99
                    except Exception as _:
                        pass
        except Exception as e:
            print(f"Error executing queued repairs: {e}")
        # Hyperfusion trigger under stress conditions 
        while step_ros >= ros_hyperfusion_threshold and (not hyperfusion_mode) and (Fusion == "ON"):
            print(f"[step {step}] ROS {step_ros} >= {ros_hyperfusion_threshold}: hyperfusion triggered")
            previous_ros = overall_ros[-1] if overall_ros else 0.0
            previous_atp = overall_atp[-1] if overall_atp else 0.0
            cell, individual_mt, r_new, a_new, hypf_count = dynamics.hyperfusion(cell, individual_mt, step, ax1, ax2, ax3, ax4,
                                                                    previous_ros,
                                                                    previous_atp,
                                                                    resources, 
                                                                    animate_each_step=True,
                                                                    animate_pause=0.05,
                                                                    k_ros=0.005, k_atp=0.005, min_ros_drop=1)
            overall_ros.append(r_new)
            overall_atp.append(a_new)
            fusion_count += hypf_count
            cell, _ = verify_consistency(cell, individual_mt, f"Step {step} after hyperfusion")
            hyperfusion_mode = True

        # Update fusion probabilities
        # Rebuild labels & groups because cell/individual_mt may have changed
        labeled, n_groups = label(cell, np.ones((3,3), dtype=int))
        if n_groups == 0:
            break
        groups = {}
        for gid in range(1, n_groups+1):
            coords = np.argwhere(labeled == gid)
            if coords.size:
                groups[gid] = MitochondriaGroup(gid, coords, individual_mt)

        # --- logit dynamics update ---
        for group in groups.values():
            if not group.members:
                continue
            pf_pay, pfis_pay = estimate_local_payoffs(group, groups, gamma=gamma)  
            p0 = group.p_fusion if hasattr(group, "p_fusion") else 0.5
            # logit step
            logit = np.log(p0) - np.log(1 - p0)
            logit += 0.25 * (pf_pay - pfis_pay)    # eta=0.25
            p_new = 1.0 / (1.0 + np.exp(-logit))
            p_new = float(max(1e-3, min(1 - 1e-3, p_new)))
            # persist to members so regrouping preserves state
            for m in group.members:
                m.p_fusion = p_new
            group.p_fusion = p_new

        # Execute dynamics actions
        if len(groups) > 0:
            actions_to_do = max(1, n_groups // 3)
            gid_list = list(groups.keys())
            random.shuffle(gid_list)
            acted = set()

            for gid in gid_list[:actions_to_do]:
                if gid in acted:
                    continue
                # Get group IDs by relabelling
                current_labeled, current_n_groups = label(cell, np.ones((3,3), dtype=int))
                if gid > current_n_groups:
                    continue 
                current_coords = np.argwhere(current_labeled == gid)
                if len(current_coords) == 0:
                    continue
                assert isinstance(individual_mt, dict), type(individual_mt)
                group = MitochondriaGroup(gid, current_coords, individual_mt)
                if not group.members or group.size ==0:
                    continue 

            # Environmental biases:
                # glucose bias 
                if glucose >= 1.5*len(groups):       
                    nudge_group_p(group, beta= -0.6)  # encourage fission
                elif glucose <= 0.8*len(groups):   
                    nudge_group_p(group, beta= 0.4) # encourage fusion

                # signalling bias
                if ca >= 3 * initial_mt_count:  # high Ca2+ load 
                    nudge_group_p(group, beta= -0.6) # encourage fission
                if camp >= 3 * initial_mt_count:  # high cAMP load
                    nudge_group_p(group, beta=0.6) # encourage fusion

                # ROS bias against fission
                if step_ros > 150:
                    nudge_group_p(group, beta=0.90) # encourage fusion

                # Decide action
                fusion_prob = group.p_fusion if hasattr(group, 'p_fusion') else 0.5

                actions_to_do = max(1, len(groups) // 3)
                current_labeled, current_n_groups = label(cell, np.ones((3,3), int))
                current_groups = {
                    g: MitochondriaGroup(g, np.argwhere(current_labeled==g), individual_mt)
                    for g in range(1, current_n_groups+1)
                    if np.any(current_labeled==g)}
                acted = set()
                done = 0
                max_tries = 3 * actions_to_do  # backstop
                gid_pool = list(current_groups.keys())
                random.shuffle(gid_pool)

                while done < actions_to_do and max_tries > 0 and gid_pool:
                    gid = gid_pool.pop()
                    max_tries -= 1
                    if gid in acted: 
                        continue
                    group = current_groups[gid]
                    if not group.members:
                        continue

                    # Stochastic fusion decision:
                    do_fusion = (Fusion != "OFF") and (fusion_prob >= 0.5)
                    did = False 
                    # attempt to fuse: pick best partner (nearest)
                    if do_fusion:
                        best_partner = None
                        best_dist = float("inf")
                        for gid2, group2 in current_groups.items():
                            if gid2 == gid or not group2.members or gid2 in acted:
                                continue
                            d = np.linalg.norm(group.centroid - group2.centroid)
                            if d < best_dist:
                                best_dist = d
                                best_partner = gid2

                        if best_partner is not None:
                            a, b = gid, best_partner
                            # decide fusion condition based on mitochondrial condition
                            avg_mtdna_a = np.mean([m.mtdna for m in current_groups[a].members])
                            avg_mtdna_b = np.mean([m.mtdna for m in current_groups[b].members])
                            if avg_mtdna_a < 100 or avg_mtdna_b < 100:
                                condition = "sick"
                            else:
                                condition = "healthy"
                            print(f"Attempting fusion: {a} + {b} ({condition})")
                            before = int(np.sum(cell))
                            cell, individual_mt, * _ = dynamic_operation(cell, individual_mt, dynamics.fusion, a, b, condition=condition)
                            fusion_count += 1
                            did = True
                            assert cell.dtype != object, "Cell became dtype=object (sequence was assigned into a cell)"
                            u = np.unique(cell)
                            assert set(u).issubset({0, 1}), f"Grid has non-binary values: {u}"
                            after  = int(np.sum(cell))
                            if after != before:
                                print(f"[FUSION COUNT BUG] {before} -> {after} (should never change)")
                            try:
                                iv.visualise_event(cell, individual_mt, "fusion", step=step, ax=ax1, flash_times=2, flash_pause=0.12)
                            except Exception as e:
                                pass
                            acted.add(a)
                            acted.add(b)
                
                # Otehrwise attempt fission
                if (not did) and (not hyperfusion_mode) and (Fission != "OFF") and group.size >= 2:
                    print(f"Attempting fission: group {gid} (size {group.size})")
                    # pre-op: parent group coords and global set of occupied pixels
                    current_labeled, current_n_groups = label(cell, np.ones((3,3), dtype=int))
                    parent_coords = np.argwhere(current_labeled == gid)
                    before_set = set(map(tuple, np.argwhere(cell == 1)))
                    cell, individual_mt, * _ = dynamic_operation(cell, individual_mt, dynamics.fission, gid)
                    fission_count += 1
                    did = True
                    assert cell.dtype != object, "Cell became dtype=object (sequence was assigned into a cell)"
                    u = np.unique(cell)
                    assert set(u).issubset({0, 1}), f"Grid has non-binary values: {u}"
                    after_set = set(map(tuple, np.argwhere(cell == 1)))
                    new_pixels = list(after_set - before_set)
                    split_coords = new_pixels[:1] if new_pixels else []
                    try:
                        iv.visualise_event(cell, individual_mt, "fission", step=step, ax=ax1, flash_times=2, flash_pause=0.12)
                    except Exception as e:
                        pass
                    acted.add(gid)
                
                if did:
                    done += 1
                    # relabel & refresh groups after topology change
                    current_labeled, current_n_groups = label(cell, np.ones((3,3), int))
                    current_groups = {
                        g: MitochondriaGroup(g, np.argwhere(current_labeled==g), individual_mt)
                        for g in range(1, current_n_groups+1)
                        if np.any(current_labeled==g)
                    }
                    gid_pool = [g for g in current_groups if g not in acted]
                    random.shuffle(gid_pool)

    if fusion_count == 0:
        fission_to_fusion = float('inf')
    else:
        fission_to_fusion = fission_count / fusion_count

    plt.ioff() # Turn off interactive mode
    plt.show() # Uncomment for a single simulation, comment out for mutliple, and uncomment next 3 lines
    # plt.show(block=False)
    # plt.pause(1.5)
    # plt.close(fig)
    return cell, overall_ros, overall_atp, fission_to_fusion

# Test the simulation: Comment out if running Simulations file
if __name__ == "__main__":
    test_cell = iv.initialise_cell(20)
    print(f"Initial test cell has {np.sum(test_cell)} mitochondria")
    
    try:
        cell1, ros1, atp1, fis2fus = mitochondrial_dynamics(test_cell, ca=200, camp=200, glucose=200, max_steps=50, Fusion="OFF" )
        print(f"\nSimulation Summary:")
        print(f"Final ROS: {ros1[-1] if ros1 else 0}")
        print(f"Final ATP: {atp1[-1] if atp1 else 0}")
        print(f"Average ROS: {np.mean(ros1) if ros1 else 0:.2f}")
        print(f"Average ATP: {np.mean(atp1) if atp1 else 0:.2f}")
        print(f"Final mitochondria count: {np.sum(cell1)}")
        print(f"Fission to Fusion ratio: {fis2fus}")
        
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()