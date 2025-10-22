import numpy as np

class MitochondriaGroup:
    def __init__(self, gid, coords, individual_mt):
        """
        gid: group id (label)
        coords: list/array of pixel coords for this group
        individual_mt: dict mapping (x,y)->Mitochondria objects
        """
        if not isinstance(individual_mt, dict):
            raise TypeError(f"individual_mt must be a dict, got {type(individual_mt).__name__}")

        self.gid = gid
        self.coords = [tuple(int(x) for x in (c.tolist() if hasattr(c, "tolist") else c))
                       for c in coords]
        self.members = [individual_mt[(int(x), int(y))] for x,y in coords if (int(x),int(y)) in individual_mt]

        self.update_derived()
        # strategy probability: p_fusion (fallback = neutral 0.5)
        self.p_fusion = float(np.mean([m.p_fusion for m in self.members])) if self.members else 0.5

    def update_derived(self):
        """Update derived properties based on current state"""
        self.size = len(self.coords)
        if self.size > 0:
            centroid = np.mean(np.array(self.coords), axis=0)
            self.centroid = centroid
        else:
            self.centroid = np.array([0.0, 0.0])

    def recompute_members(self, individual_mt):
        """Recompute members from dictionary"""
        self.members = [individual_mt.get(coord) for coord in self.coords if coord in individual_mt]
        self.members = [m for m in self.members if m is not None]  # Remove None values
        self.update_derived()

    def group_atp(self):
        """Calculate total ATP production for the fused mitochondrion"""
        if not self.members:
            return 0
        # Use the first member as representative for group ATP calculation
        representative = self.members[0]
        
        # Group ATP is more efficient than individual ATP for larger groups
        if hasattr(representative, 'atp_production'):
            return representative.atp_production(self.size)
        else:
            # Fallback calculation
            return self.size * 30  # Base ATP per mitochondrion

    def group_ros_proxy(self):
        """Calculate estimated ROS production for this group."""
        if not self.members:
            return 0
        
        # Simple proxy: ROS increases with group size but not linearly
        # Larger groups are more efficient at ROS management
        base_ros = max(1, int(0.5 * self.size))
        
        # Add penalty for damaged mitochondria
        damaged_penalty = 0
        for member in self.members:
            if hasattr(member, 'mtdna') and member.mtdna < 100:
                damaged_penalty += 1
        
        return base_ros + damaged_penalty

    def average_mtdna(self):
        """Calculate average mtDNA content in the group."""
        if not self.members:
            return 0
        
        mtdna_values = []
        for member in self.members:
            if hasattr(member, 'mtdna'):
                mtdna_values.append(member.mtdna)
        
        return np.mean(mtdna_values) if mtdna_values else 0

    def health_status(self):
        """Determine overall health status of the group."""
        avg_mtdna = self.average_mtdna()
        
        if avg_mtdna < 10:
            return "flagged for mitophagy"
        elif avg_mtdna < 100:
            return "sick"
        else:
            return "healthy"

    def can_fuse_with(self, other_group, max_distance=4):
        """Check if this group can fuse with another group."""
        if not self.members or not other_group.members:
            return False
        
        # Check distance between centroids
        distance = np.linalg.norm(self.centroid - other_group.centroid)
        return distance <= max_distance

    def update_fusion_probability(self, environmental_factors):
        """Update fusion probability based on environmental conditions."""
        # Environmental factors:
        # 'glucose', 'ca', 'camp', 'ros_level'
        
        base_prob = 0.5
        
        # Sick groups strongly prefer fusion for repair
        if self.health_status() == "sick":
            base_prob = 0.9
        elif self.health_status() == "mitophagy":
            base_prob = 0.95  # Desperate attempt to survive
        
        # Environmental modulation (if factors provided)
        if isinstance(environmental_factors, dict):
            glucose = environmental_factors.get('glucose', 100)
            ca = environmental_factors.get('ca', 100)
            camp = environmental_factors.get('camp', 100)
            
            # High glucose encourages fission (reduce fusion probability)
            if glucose > 150:
                base_prob *= 0.8
            elif glucose < 50:
                base_prob = min(0.99, base_prob * 1.3)
            
            # High Ca2+ encourages fission
            if ca > 150:
                base_prob *= 0.7
            
            # High cAMP encourages fusion
            if camp > 150:
                base_prob = min(0.99, base_prob * 1.4)
        
        self.p_fusion = max(0.01, min(0.99, base_prob))
        return self.p_fusion

    def __repr__(self):
        """String representation of the group."""
        health = self.health_status()
        return f"MitGroup(id={self.gid}, size={self.size}, health={health}, p_fus={self.p_fusion:.2f})"

    def __str__(self):
        """Readable string representation."""
        return f"Mitochondrial Group {self.gid}: {self.size} mitochondria ({self.health_status()})"
