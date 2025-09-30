import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import random

class Mitochondria:
    def __init__(self, x_coord: int, y_coord: int, ca: int, camp: int):
        self.x_coord = x_coord
        self.y_coord = y_coord

        self.nucleotides = random.randint(2,3)
        self.mtdna = self.nucleotides * 100
        self.mutatedmtdna = 0

        self.ca = max(1, ca)
        self.camp = max(1, camp)

        # track an internal ROS pool (accumulated)
        self.ros = random.randint(1,3) # Baseline production
        # ROS clearance capacity (antioxidant systems)
        self.ros_clearance = 5  # can clear 5 ROS units per step if healthy

        # exchangeable material
        self.mt_material = max(1, self.mtdna // 20) 
        # Fusion probability (50/50 fusion/fission)
        self.p_fusion = 0.5
        
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
        
        # Return net ROS contribution to cell
        return max(0, ros_generated - ros_cleared)

    def use_ca(self, amount: int) -> int:
        """
        Consume up to `amount` Ca2+. Return ROS generated (integer).
        """
        if amount <= 0:
            return 0
        
        consumed = min(self.ca, amount)
        self.ca = max(0, self.ca - consumed)

        # Ca2+ handling generates more ROS than cAMP
        ros_generated = max(1, int(consumed * 0.12))
        # Add to internal pool
        self.ros += ros_generated
        
        # Apply clearance
        clearance_capacity = self.ros_clearance if self.mtdna >= 100 else max(1, self.ros_clearance // 2)
        ros_cleared = min(self.ros, clearance_capacity)
        self.ros = max(0, self.ros - ros_cleared)
        
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
        elif self.mtdna < 100:
            baseline_ros += random.randint(3, 5)  # moderately damaged
            
        # Add to internal pool
        self.ros += baseline_ros
        
        # Apply clearance
        clearance_capacity = self.ros_clearance if self.mtdna >= 100 else max(1, self.ros_clearance // 2)
        ros_cleared = min(self.ros, clearance_capacity)
        self.ros = max(0, self.ros - ros_cleared)
        
        return max(0, baseline_ros - ros_cleared)
    
    def use_ros(self, external_ros: int) -> int:
        """
        Apply external ROS stress. This accumulates and causes damage if high.
        """
        if external_ros <= 0:
            return 0
        
        # Add external ROS to internal pool
        self.ros += external_ros
        
        # High ROS causes damage
        if self.ros > 8:  # damage threshold
            damage_chance = min(0.8, (self.ros - 8) * 0.1)  # higher ROS = more damage
            if random.random() < damage_chance:
                self.damage()
                
        # Apply clearance (reduced if damaged)
        clearance_capacity = self.ros_clearance if self.mtdna >= 100 else max(1, self.ros_clearance // 3)
        ros_cleared = min(self.ros, clearance_capacity)
        self.ros = max(0, self.ros - ros_cleared)
        
        return external_ros  
        

    # ---- Exchange / damage / bookkeeping ----
    def exchangeable(self):
        """Update amount of exchangeable material based on current mtDNA"""
        self.mt_material = max(1, self.mtdna // 20)  # 10 or 15 initially (given 2-3 nucleotides)
        # Reduce clearance capacity if very damaged
        if self.mtdna < 50:
            self.ros_clearance = max(1, 5 - (50 - self.mtdna) // 10)
        elif self.mtdna < 100:
            self.ros_clearance = max(2, 5 - (100 - self.mtdna) // 20)
        else:
            self.ros_clearance = 5
        return self.mt_material

    def damage(self):
        """Apply damage by ROS to mitochondria"""
        d = random.randint(0, 2)
        if d == 0:
            self.mtdna = max(0, self.mtdna - 1)
            self.exchangeable() 
        elif d == 1:
            self.mutatedmtdna += 1
        # Damage reduces ROS clearance capacity
        if self.mtdna < 75:
            self.ros_clearance = max(2, self.ros_clearance - 1)

        return (self.mtdna, self.mt_material)

    def fuse_exchange(self, other, group_members: list, condition: str): 
        """Exchange materials during fusion based on condition."""
        if not other or not hasattr(other, 'mt_material'):
            return
            
        if condition == "healthy":
            # Balanced exchange during healthy fusion
            received = min(other.mt_material, 2)
            given = min(self.mt_material, 2)
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
                available = max(0, other.mt_material - 2)
                transfer = min(needed, available, 3)
                if transfer > 0:
                    self.mt_material += transfer
                    other.mt_material = max(2, other.mt_material - transfer)
                    # Update mtDNA 
                    self.mtdna = min(200, self.mt_material * 20)
                    other.mtdna = max(60, other.material * 20)
                    # Repair improves  clearance 
                    self.ros_clearance = min(6, self.ros_clearance + 2)
            elif other.mtdna < 100:
                needed = max(0, (100 - other.mtdna)//20)
                # Healthy mitochondrion donates but maintains minimun health 
                available = max(0, self.mt_material - 2)
                transfer = min(needed, available, 3)
                if transfer > 0:
                    other.mt_material += transfer
                    self.mt_material = max(2, self.mt_material - transfer)
                    # Update mtDNA 
                    other.mtdna = min(200, other.mt_material * 20)
                    self.mtdna = max(60, self.material * 20)
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
        # Optimised for moderate fusion (size 8-12)
        base_atp = int(300 / (1 + np.exp(-0.5 * (mt_size - 10))))

                # Add efficiency bonus for optimal sizes
        if 6 <= mt_size <= 15:
            base_atp = int(base_atp * 1.2)
        elif mt_size > 20:  # Penalty for excessive size
            base_atp = int(base_atp * 0.8)

        # ROS stress reduces ATP production efficiency
        efficiency = 1.0
        if self.ros > 5:
            efficiency = max(0.3, 1.0 - (self.ros - 5) * 0.08)
        
        # mtDNA damage reduces efficiency
        if self.mtdna < 100:
            mtdna_efficiency = max(0.2, self.mtdna / 100)
            efficiency *= mtdna_efficiency

        final_atp = max(1, int(base_atp * efficiency))
        return final_atp
    
    def __repr__(self):
        return f"Mitochondria(pos=({self.x_coord},{self.y_coord}), mtdna={self.mtdna}, material={self.mt_material})"
