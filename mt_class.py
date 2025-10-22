import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import random

class Mitochondria:
    def __init__(self, x_coord: int, y_coord: int, ca: int, camp: int):
        self.x_coord = x_coord
        self.y_coord = y_coord

        self.nucleotides = random.randint(2,3)
        self.mtdna = self.nucleotides * 150
        self.mutatedmtdna = 0

        self.ca = max(2, ca)
        self.camp = max(2, camp)

        # track an internal ROS pool (accumulated)
        self.ros = random.randint(1,3) # Baseline production
        # ROS clearance capacity (antioxidant systems)
        self.ros_clearance = 4  # can clear 4 ROS units per step if healthy

        # exchangeable material
        self.mt_material = max(1, self.mtdna // 20) 
        # Fusion probability (50/50 fusion/fission)
        self.p_fusion = 0.5
        
        # ROS damage tracking
        self.just_damaged = False
        self.critically_damaged = False
        self.last_damage_loss = 0
        self.queued_action = None              # None | "exchange" | "bud"
        self.queued_at_step = -1
        self.queued_coord = None               # where this mito sits in the grid
        self.repair_cooldown = 0               # steps until we can queue again
        
    # ---- Resource update functions ----
    def use_camp(self, amount: int) -> int:
        """
        Consume up to `amount` cAMP. Return ROS generated (int).
        For simplicity, small ROS is generated proportional to consumption.
        """
        if amount <= 0:
            return 0
        
        consumed = min(self.camp, amount)
        self.camp = max(0, self.camp - consumed)

        # ROS generation from cAMP metabolism
        ros_generated = max(1, int(consumed * 0.08))
        # Add to internal pool
        self.ros += ros_generated
        
        # Apply clearance (healthy mitochondria can clear more)
        clearance_capacity = self.ros_clearance if self.mtdna >= 100 else max(1, self.ros_clearance // 2)
        ros_cleared = min(self.ros, clearance_capacity)
        self.ros = max(0, self.ros - ros_cleared)
        
        # Return small ROS contribution to cell
        return max(1, ros_generated - ros_cleared // 2)

    def use_ca(self, amount: int) -> int:
        """
        Consume up to `amount` Ca2+. Return ROS generated (integer).
        """
        if amount <= 0:
            return 0
        
        consumed = min(self.ca, amount)
        self.ca = max(0, self.ca - consumed)

        # Ca2+ handling generates more ROS than cAMP
        ros_generated = max(1, int(consumed * 1.2))
        # Add to internal pool
        self.ros += ros_generated
        
        # Apply clearance
        clearance_capacity = self.ros_clearance if self.mtdna >= 100 else max(1, self.ros_clearance // 2)
        ros_cleared = min(self.ros, clearance_capacity)
        self.ros = max(1, self.ros - ros_cleared // 2)
        
        return max(0, ros_generated - ros_cleared)

    def baseline_metabolism_ros(self) -> int:
        """
        Generate baseline ROS from normal respiration.
        Called once per step per mitochondrion.
        """
        # Baseline ROS production (always some level of ROS from respiration)
        baseline_ros = random.randint(2, 4)
        
        # Damaged mitochondria produce more ROS
        if self.mtdna < 50:
            baseline_ros += random.randint(4, 6)  # very damaged
        elif self.mtdna < 90:
            baseline_ros += random.randint(3, 5)  # moderately damaged
            
        # Add to internal pool
        self.ros += baseline_ros
        
        # Apply clearance
        clearance_capacity = self.ros_clearance if self.mtdna >= 100 else max(1, self.ros_clearance // 2)
        ros_cleared = min(self.ros, clearance_capacity)
        self.ros = max(0, self.ros - ros_cleared)
        
        return max(1, baseline_ros - ros_cleared // 2)  # return some ROS to cell
    
    def use_ros(self, group_size, external_ros: int) -> int:
        """
        Apply external ROS stress. This accumulates and causes damage if high.
        - A larger fraction of external ROS now enters the mitochondrial pool.
        - Severe bursts also nick mtDNA directly a tiny bit.
        - Temporary clearance dampening under heavy stress makes damage more likely.
        """
        self.just_damaged = False  # reset damage flag each step
        self.critical_damage = False
        if self.repair_cooldown > 0:
            self.repair_cooldown -= 1

        if external_ros <= 0:
            return 0
        # Check to see if mitochondria are grouped (fused)
        if group_size >= 1: 
            landed = max(1, external_ros//14)
            denom = 1+ 0.6* max(0, int(group_size - 1))
            landed_per_mito = max(1, landed // denom)
            self.ros += landed_per_mito

        # Under heavy stress, transiently reduce clearance (can’t keep up)
        if self.ros > 18:
            self.ros_clearance = max(1, int(self.ros_clearance * 0.8))

        # Severe bursts produce small direct oxidative nicks to mtDNA
        if external_ros >= 120:  # tuneable
            self.mtdna = max(1, self.mtdna - random.randint(2, 4))
            self.exchangeable()

        # High ROS causes damage; severity handled in damage()
        if self.ros > 12:
            # Probability grows with stress, capped at 0.
            damage_chance = min(0.7, 0.05 + 0.04 * (self.ros - 10))
            if random.random() < damage_chance:
                self.damage()

        return external_ros
        

    # ---- Exchange / damage / bookkeeping ----
    def exchangeable(self):
        """Update amount of exchangeable material based on current mtDNA"""
        self.mt_material = max(1, self.mtdna // 20)  # 10 or 15 initially (given 2-3 nucleotides)
        # Reduce clearance capacity if very damaged
        if self.mtdna < 50:
            self.ros_clearance = max(2, 5 - (50 - self.mtdna) // 10)
        elif self.mtdna < 100:
            self.ros_clearance = max(3, 5 - (100 - self.mtdna) // 10)
        else:
            self.ros_clearance = 5
        return self.mt_material
    
    def damage(self):
        """
        Apply ROS-driven mtDNA damage.
        - Severity scales with current ROS pool.
        - Rare strand breaks under very high ROS lop off bigger chunks.
        """
        prev_mtdna = self.mtdna
        # Base severity grows with ROS above threshold
        over = max(0, self.ros - 12)
        # mtDNA lost (stochastic)
        base_loss = 4 + int(over * random.uniform(0.8, 1.25))
        # Occasionally, strand break under very high ROS
        extra_loss = 0
        if self.ros > 30 and random.random() < 0.22:
            extra_loss = random.randint(8, 18)

        total_loss = max(3, base_loss + extra_loss)
        total_loss = min(total_loss, 20)  # cap max loss per hit

        # Very damaged mito are somewhat protected (can't lose below 20 in one go)
        if self.mtdna <= 30:
            total_loss = min(total_loss, 8)

        self.mtdna = max(1, self.mtdna - max(2, total_loss))
        self.just_damaged = True
        self.last_damage_loss = total_loss
        self.exchangeable()

        # Flag critical damage (for special fusion handling)
        crit = 12
        crit_zone = (self.mtdna < 20 and prev_mtdna >= 20)
        if crit_zone or (self.mtdna < crit and prev_mtdna >= crit):
            self.critical_damage = True

        return (self.mtdna, self.mt_material)

    def fuse_exchange(self, other, group_members: list, condition: str): 
        """Exchange materials during fusion based on condition."""
        if not other or not hasattr(other, 'mt_material'):
            return
            
        if condition == "healthy":
            # Balanced exchange during healthy fusion
            received = min(other.mt_material, 1)
            given = min(self.mt_material, 1)
            self.mt_material = max(1, self.mt_material - given + received)
            other.mt_material = max(1, other.mt_material - received + given)        
            # Update mtDNA proportionally
            self.mtdna = max(50, self.mt_material * 20)
            other.mtdna = max(50, other.mt_material * 20)
            # Healthy fusion can also share ROS clearance capacity
            avg_clearance = (self.ros_clearance + other.ros_clearance) // 2
            self.ros_clearance = min(8, avg_clearance + 2)  
            other.ros_clearance = min(8, avg_clearance + 2)
            
        elif condition == "sick":
            # Damaged mitochondrion tries to repair itself without harming healthy mitochondrion
            if self.mtdna < 100:
                needed = max(0, (100 - self.mtdna)//20)
                # Healthy mitochondrion donates but maintains minimun health 
                available = max(0, other.mt_material - 3)
                transfer = min(needed, available, 3)
                if transfer > 0:
                    self.mt_material += transfer + 1
                    other.mt_material = max(2, other.mt_material - transfer)
                    # Update mtDNA 
                    self.mtdna = min(150, self.mt_material * 20)
                    other.mtdna = max(60, other.mt_material * 20)
                    # Repair improves  clearance 
                    self.ros_clearance = min(6, self.ros_clearance + 2)
            elif other.mtdna < 100:
                needed = max(0, (100 - other.mtdna)//20)
                # Healthy mitochondrion donates but maintains minimun health 
                available = max(0, self.mt_material - 2)
                transfer = min(needed, available, 3)
                if transfer > 0:
                    other.mt_material += transfer + 1
                    self.mt_material = max(2, self.mt_material - transfer)
                    # Update mtDNA 
                    other.mtdna = min(150, other.mt_material * 20)
                    self.mtdna = max(60, self.mt_material * 20)
                    # Repair improves  clearance 
                    other.ros_clearance = min(6, other.ros_clearance + 2)
        elif condition == "just_damaged":
            # Recently damaged mitochondrion tries to repair itself but can only take 1 unit
            if self.just_damaged and self.mtdna < 100:
                needed = max(0, (100 - self.mtdna)//20)
                # Healthy mitochondrion donates but maintains minimun health 
                available = 1
                transfer = min(needed, available, 1)
                if transfer > 0:
                    self.mt_material += transfer 
                    other.mt_material = max(2, other.mt_material - transfer)
                    # Update mtDNA 
                    self.mtdna = min(150, self.mt_material * 20)
                    other.mtdna = max(60, other.mt_material * 20)
                    # Repair improves  clearance 
                    self.ros_clearance = min(6, self.ros_clearance + 2)
            elif other.just_damaged and other.mtdna < 100:
                needed = max(0, (100 - other.mtdna)//20)
                # Healthy mitochondrion donates but maintains minimun health 
                available = 1
                transfer = min(needed, available, 1)
                if transfer > 0:
                    other.mt_material += transfer
                    self.mt_material = max(2, self.mt_material - transfer)
                    # Update mtDNA 
                    other.mtdna = min(150, other.mt_material * 20)
                    self.mtdna = max(60, self.mt_material * 20)
                    # Repair improves  clearance 
                    other.ros_clearance = min(6, other.ros_clearance + 2)

    def atp_production(self, mt_size: int):
        """
        ATP production follows a sigmoidal curve with a peak at mt_size = 10.
        High internal ROS reduces ATP yield
        """
        if mt_size <= 0:
            return 0
        
        # Base ATP production
        # Optimised for moderate fusion (size 6-15)
        base_atp = 30 + int(270 / (1 + np.exp(-0.5 * (mt_size - 10))))

                # Add efficiency bonus for optimal sizes
        if 6 <= mt_size <= 15:
            base_atp = int(base_atp * 1.5)
        elif mt_size > 20:  # Penalty for excessive size
            base_atp = int(base_atp * 0.8)

        # ROS stress reduces ATP production efficiency
        efficiency = 1.0
        if self.ros > 20:
            efficiency = max(0.3, 1.0 - (self.ros - 5) * 0.08)
        
        # mtDNA damage reduces efficiency
        if self.mtdna < 70:
            mtdna_efficiency = max(0.2, self.mtdna / 100)
            efficiency *= mtdna_efficiency

        final_atp = max(1, int(base_atp * efficiency))
        return final_atp
    
    def __repr__(self):
        return f"Mitochondria(pos=({self.x_coord},{self.y_coord}), mtdna={self.mtdna}, material={self.mt_material})"
