import numpy as np
from src.quaternion import Quaternion
from copy import deepcopy

class Tree:
    """
        Main class of the project. Represents a simple bronchial tree surfactant injection model.
    """
    def __init__(
            self,
            surfactant,
            a0,
            n_gen,
            track_ruptures=False,
            coating=True
            ):
        """
        Parameters
        ----------
        surfactant : Surfactant
            Surfactant to use. It essentially gives physical parameters (viscosity,
            density and surface tension).
        a0 : float
            Tracheal radius of the tree. Determines all the children radius.
        n_gen : int
            Number of generations in the tree.
        """
        self.a0 = a0
        self.starting_level = 0
        self.surfactant = surfactant
        self.n_gen = n_gen
        self.first_speed = 0.0
        self.injected_volume = 0.0
        self.coating = coating

        # The bifurcation angle is always set to pi / 4 in our case
        self.theta = np.pi / 4
        # The radius decreasing ratio is always the same
        self.lamb = 2 ** (-1/3)

        # Plug speeds before coating.
        self.speeds = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]
        # Plug volumes before coating.
        self.volumes = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]

        # This allows us to know if a branch has already been coated. It is important to
        # know that each node is attached with its upper branch and the two of them are
        # handled together in this model.
        self.coated = [np.full((2 ** i,), False) for i in range(self.n_gen + 1)]
        # Plug speeds after coating
        self.new_speeds = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]
        # Splitting ratios
        self.ratios = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen)]
        # This list will allow us to know which daughter branch is uphill in each
        # bifurcation.
        self.uphills = []
        # Volumes in terminal nodes.
        self.final_volumes = np.zeros(shape=(2 ** self.n_gen))
        # Roll angles for each bifurcation.
        self.phis = []
        # Pitch angles for each bifurcation.
        self.gammas = []

        self.track_ruptures = track_ruptures
        if self.track_ruptures:
            # Keep track of when a splitting factor vanishes
            self.split_ruptures = [np.ones(shape=(2 ** i,)) * n_gen for i in range(self.n_gen)]
            self.coating_ruptures = [np.ones(shape=(2 ** i,)) * (-1) for i in range(self.n_gen + 1)]
        
        # Splitting factors
        self.alphas = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen)]

        # Deposition ratios
        self.H = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]

        # Performances
        self.efficiency = None
        self.homogeneity = None
        self.distribution = None

    def reset_(self):
        """
        Hard resets the tree. Resets its geometry and all the computed volumes, speeds,
        ratios etc.
        """
        self.theta = np.pi / 4
        self.lamb = 2 ** (-1/3)
        self.speeds = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]
        self.volumes = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]
        self.coated = [np.full((2 ** i,), False) for i in range(self.n_gen + 1)]
        self.new_speeds = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]
        self.ratios = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen)]
        self.uphills = []
        self.final_volumes = np.zeros(shape=(2 ** self.n_gen))
        self.phis = []
        self.gammas = []
        self.vecs = [[self.first_vec]]
        self.alphas = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen)]

        self.first_node_quat = Quaternion(
            0,
            *self.first_vec
        )
        qy = self.quat_exp(
            theta=-self.tree_phi/2,
            vec=np.array((0, 1.0, 0))
        )
        qx = self.quat_exp(
            theta=self.tree_gamma/2,
            vec=np.array((1.0, 0, 0))
        )
        uz = Quaternion(0, 0, 0, 1.0)
        normal_quat = qy@(qx@uz@qx.inv()/qx.squared_norm())@qy.inv()/qy.squared_norm()
        self.first_normal = normal_quat
        self.normals = [
            [np.array([self.first_normal.i, self.first_normal.j, self.first_normal.k])]
            ]

        self.injected_volume = 0.0
        self.efficiency = None
        self.homogeneity = None
        self.distribution = None
    
    def reset_volumes(self):
        """Resets everything but the geometry of the tree."""
        self.efficiency = None
        self.homogeneity = None
        self.distribution = None
        self.coated = [np.full((2 ** i,), False) for i in range(self.n_gen + 1)]
        self.speeds = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]
        self.volumes = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]
        self.final_volumes = np.zeros(shape=(2 ** self.n_gen))
        self.injected_volume = 0.0

    def set_surfactant(
            self,
            mu,
            rho,
            sigma
        ):
        """
        Give new characteristics to the surfactant.

        Parameters
        ----------
        mu : float
            Viscosity.
        rho : float
            Density.
        Sigma : float
            Surface tension.
        """
        self.surfactant.mu = mu
        self.surfactant.rho = rho
        self.surfactant.sigma = sigma

    def set_flow_rate(
            self,
            flow_rate
        ):
        """
        Set a new injection flow rate.

        Parameters
        ----------
        flow_rate : float
            Flow rate to apply to the plug injected in the first node.
        """
        self.flow_rate = flow_rate
        self.first_speed = self.flow_rate / (np.pi * np.power(self.a0, 2))
        self.speeds[0] = np.array([self.first_speed])
    
    def set_volume(
            self,
            volume
        ):
        """
        Set a new injection volume.

        Parameters
        ----------
        volume : float
            Volume to inject.
        """
        self.V0 = volume
        self.injected_volume = volume
        self.volumes[0] = np.array([self.V0])

    def set_angles(
            self,
            gamma,
            phi
        ):
        """
        Set new tree angles. Those angles correspond to the tracheal bifurcation
        pitch and roll angles.

        Parameters
        ----------
        gamma : float
            Pitch angle.
        Phi : float
            Roll angle.
        """
        self.uphills = []
        self.phis = []
        self.gammas = []
        self.first_vec = np.array((0, -np.cos(gamma), -np.sin(gamma)))
        self.tree_gamma = gamma
        self.tree_phi = phi
        self.first_node_quat = Quaternion(
            0,
            *self.first_vec
        )
        qy = self.quat_exp(
            theta=-self.tree_phi/2,
            vec=np.array((0, 1.0, 0))
        )
        qx = self.quat_exp(
            theta=self.tree_gamma/2,
            vec=np.array((1.0, 0, 0))
        )
        uz = Quaternion(0, 0, 0, 1.0)
        normal_quat = qy@(qx@uz@qx.inv()/qx.squared_norm())@qy.inv()/qy.squared_norm()
        self.first_normal = normal_quat
        self.normals = [
            [np.array([self.first_normal.i, self.first_normal.j, self.first_normal.k])]
            ]
        self.vecs = [[self.first_vec]]
        self.full_vecs = [[self.first_vec * 6 * self.a0]]
    
    def quat_exp(
            self,
            theta,
            vec
        ):
        """
        Quaternion exponentiation used in 3D rotations.

        Parameters
        ----------
        theta : float
            Corresponds to half of the desired rotation angle.
        vec : Numpy array
            Corresponds to the desired rotation axis.
        """
        sin_vec = vec * np.sin(theta)
        return Quaternion(
            np.cos(theta),
            *sin_vec
        )

    def generate_next_level(
            self,
            level
            ):
        """
        Generate the next tree level geometry. This is essentially done through
        tridimensional quaternions rotations for a better computation.

        Parameters
        ----------
        level : int
            Level to generate.
        """

        # New level direction vectors. This corresponds to the direction vector of
        # the parent branch of the bifurcation.
        new_level_vecs = []
        # New level normal vectors. This corresponds to the vector that is normal to
        # the plane formed by the bifurcation.
        new_level_normals = []

        # New level roll angles.
        new_level_phis = []
        # New level pitch angles.
        new_level_gammas = []
        # New level uphill daughters indication.
        new_level_uphills = []

        full_level = []

        # For each bifurcation, with its direction vector and normal vector.
        for full_vec, vec, normal in zip(self.full_vecs[-1], self.vecs[-1], self.normals[-1]):
            normal_quat = Quaternion(0, *normal)
            
            # Compute direction vector and normal vector for left and right branch

            # Rotate the parent direction vector of a pi / 4 angle around the parent
            # normal vector to obtain the left branch direction vector.
            qn = self.quat_exp(np.pi/8, normal)
            left_quat = qn@Quaternion(0, *vec)@qn.inv() / qn.squared_norm()

            # Same process for the right branch, but with a -pi / 4 rotation.
            qn = self.quat_exp(-np.pi/8, normal)
            right_quat = qn@Quaternion(0, *vec)@qn.inv() / qn.squared_norm()

            left_vec = np.array((left_quat.i, left_quat.j, left_quat.k))
            right_vec = np.array((right_quat.i, right_quat.j, right_quat.k))
            
            # To get the left branch normal vector, make a -pi / 2 rotation around
            # the parent direction vector, then a pi / 4 rotation around the parent
            # normal vector.
            qnl1 = self.quat_exp(-np.pi / 4, vec)
            qnl2 = self.quat_exp(np.pi / 8, normal)
            left_normal_quat = (
                qnl2@
                (qnl1@normal_quat@qnl1.inv())
                /qnl1.squared_norm()
                @qnl2.inv()
                /qnl2.squared_norm()
                )
            left_normal = np.array(
                (left_normal_quat.i, left_normal_quat.j, left_normal_quat.k)
                )

            # Same process for the right branch but with opposite rotations.
            qnr1 = self.quat_exp(np.pi / 4, vec)
            qnr2 = self.quat_exp(-np.pi / 8, normal)
            right_normal_quat = (
                qnr2@
                (qnr1@normal_quat@qnr1.inv())
                /qnl1.squared_norm()
                @qnr2.inv()
                /qnl2.squared_norm()
                )
            right_normal = np.array(
                (right_normal_quat.i, right_normal_quat.j, right_normal_quat.k)
                )

            # Determine which branch is uphill by looking which direction vector is the
            # highest. Then compute roll and pitch angles.
            if left_vec[2] > right_vec[2]:
                left_uphill = True
                right_uphill = False
                if level == 0:
                    phi = self.tree_phi
                else:
                    phi = np.arcsin(
                        (left_vec - right_vec)[2] / np.linalg.norm(left_vec - right_vec)
                        )
            else:
                right_uphill = True
                left_uphill = False
                if level == 0:
                    phi = -self.tree_phi
                else:
                    phi = np.arcsin(
                        (right_vec - left_vec)[2] / np.linalg.norm(left_vec - right_vec)
                        )
            if level == 0:
                gamma = self.tree_gamma
            else:
                gamma = np.arcsin(
                    vec[2] / np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
                    )
            new_level_phis.append(phi)
            new_level_gammas.append(gamma)
            new_level_uphills.append(left_uphill)
            new_level_uphills.append(right_uphill)
            new_level_vecs.append(left_vec)
            new_level_vecs.append(right_vec)
            new_level_normals.append(left_normal)
            new_level_normals.append(right_normal)
            full_level.append(full_vec + left_vec * (6 * self.a0 * (self.lamb ** (level+1))))
            full_level.append(full_vec + right_vec * (6 * self.a0 * (self.lamb ** (level+1))))

        # Add the computed level geometry to the tree geometry
        self.vecs.append(np.array(new_level_vecs))
        self.normals.append(np.array(new_level_normals))
        self.phis.append(np.array(new_level_phis))
        self.gammas.append(np.array(new_level_gammas))
        self.uphills.append(np.array(new_level_uphills))
        self.full_vecs.append(np.array(full_level))

    def inject(
            self,
            volume,
            flow_rate
            ):
        """
        Make a new injection with given volume and flow rate in the tracheal branch.

        Parameters
        ----------
        volume : float
            Volume to inject.
        flow_rate : float
            Flow rate of the injection.
        """
        self.first_speed = flow_rate / (np.pi * np.power(self.a0, 2))
        self.V0 = volume
        self.injected_volume += volume
        self.flow_rate = flow_rate
        self.speeds = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]
        self.speeds[0] = np.array([self.first_speed])
        self.volumes = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]
        self.volumes[0] = np.array([self.V0])
        self.new_speeds = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen + 1)]
        self.ratios = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen)]
        self.alphas = [np.zeros(shape=(2 ** i,)) for i in range(self.n_gen)]

        self.efficiency = None
        self.homogeneity = None
        self.distribution = None

    def generate_tree(self):
        """Generate the full tree geometry."""
        for level in range(self.n_gen):
            self.generate_next_level(level)

    def loop(self):
        """
        This loop computes the full liquid propagation
        from the statring injection level to the acini by alternating between
        coating and splitting phases. It then finishes with the coating before
        reaching the final nodes.
        """
        for i in range(self.starting_level, self.n_gen):
            self.coat(i)
            self.split(i)
        self.final_coat()

    def coat(
            self,
            level
            ):
        """
        Coating process on a given level.

        Parameters
        ----------
        level : int
            Coating level to compute in the tree.
        """
        self.H[level][np.logical_not(self.coated[level])] = 0.36 * (1 - np.exp(- 2 * np.power((self.surfactant.mu * self.speeds[level][np.logical_not(self.coated[level])] / self.surfactant.sigma), 0.523)))
        # Compute the real coating width with the ratio H.
        if not self.coating:
            self.H[level][np.logical_not(self.coated[level])] = 0
        self.h = self.a0 * 2 ** (-level / 3) * self.H[level]

    def split(
            self,
            level
            ):
        """
        Splitting process on a given level.
        """
        # No need of a mask for the tracheal branch
        if level == 0:
            mask = None
        # Else, create a mask of all nodes of the level that still have surfactant in.
        # This mask is used in the ratio computation, to avoid computing ratios for
        # already empty nodes.
        else:
            mask = (self.volumes[level] > 0)
        g = 9.81
        lamb = 2 ** (-1/3)
        a = self.a0 * lamb ** level
        new_speeds = self.speeds[level] * (1 - self.H[level]) ** 2
        Re = self.surfactant.rho * new_speeds * a / self.surfactant.mu
        Ca = self.surfactant.mu * new_speeds / self.surfactant.sigma
        Rd = 1 - np.power((1 - self.h / a), 2)
        # Airway volume.
        Va = np.pi * np.power(a, 2) * 6 * a
        B0 = self.surfactant.rho * g * np.power(a, 2) / self.surfactant.sigma
        
        # We fix the deposition ratio of already coated branches at 0, so then we do
        # not coat them anymore.
        Rd[self.coated[level]] = 0.0
       
        V1_tilde = (self.volumes[level] - Rd * Va) / (np.pi * np.power(a, 3))
        
        new_mask = V1_tilde > 0.0
        X = (2 * B0 * V1_tilde[new_mask] * (lamb ** 4)) / (Re[new_mask] * Ca[new_mask] * (lamb ** 2) + 16 * V1_tilde[new_mask] * Ca[new_mask])
        self.alphas[level][new_mask] = (1 - (X * np.sin(self.theta) * np.sin(self.phis[level][new_mask])) / (1 - X * np.cos(self.theta) * np.sin(self.gammas[level][new_mask]))) / 2

        self.alphas[level][self.alphas[level] < 0.0] = 0.0
        self.alphas[level][self.alphas[level] > 1.0] = 1.0
        
        self.speeds[level+1][self.uphills[level]] = self.speeds[level] * self.alphas[level] / (lamb ** 2)
        self.speeds[level+1][np.logical_not(self.uphills[level])] = self.speeds[level] * (1 - self.alphas[level]) / (lamb ** 2)

        self.volumes[level+1][self.uphills[level]] = self.alphas[level] * (self.volumes[level] - Rd * Va)
        self.volumes[level+1][np.logical_not(self.uphills[level])] = (1 - self.alphas[level]) * (self.volumes[level] - Rd * Va)
        self.volumes[level+1][self.volumes[level+1] < 0] = 0
        
        self.speeds[level+1][self.speeds[level+1] < 0] = 0
        self.speeds[level+1][self.volumes[level+1] <= 0] = 0
        
        self.alphas[level][np.logical_not(new_mask)] = np.nan

        if self.track_ruptures:
            if level == 0:
                idx_3 = np.full(2 ** level, False)
                idx_3 = (self.volumes[level+1][self.uphills[level]] == 0.0) & (V1_tilde != 0.0)
                self.split_ruptures[level][idx_3] = level
                self.coating_ruptures[level][V1_tilde <= 0.0] = level
            else:
                # First track nodes that were already empty
                idx_1 = np.full(2 ** (level), False)
                idx_1 = (V1_tilde <= 0.0)
                
                # Replace ruptures levels for empty nodes
                self.split_ruptures[level][idx_1] = -1

                # Then track new ruptures and put their rupture levels
                # Split ruptures
                idx_3 = np.full(2 ** level, False)
                idx_3 = (self.volumes[level+1][self.uphills[level]] == 0.0) & ((self.volumes[level] - Rd * Va) > 0.0)
                self.split_ruptures[level][idx_3] = level

                # Coating ruptures
                idx_4 = np.full(2 ** level, False)
                idx_4 = (self.volumes[level] != 0.0) & ((self.volumes[level] - Rd * Va) <= 0.0)
                self.coating_ruptures[level][idx_4] = level

        self.coated[level][self.H[level] > 0] = True

        if level == 0:
            self.coated[level] = np.array([True])
    
    def final_coat(self):
        """Coating of the final branches before the final nodes."""
        lamb = 2 ** (-1/3)
        a = self.a0 * lamb ** self.n_gen
        Va = np.pi * np.power(a, 2) * 6 * a
        self.H[-1][np.logical_not(self.coated[-1])] = 0.36 * (1 - np.exp(- 2 * np.power((self.surfactant.mu * self.speeds[self.n_gen][np.logical_not(self.coated[-1])] / self.surfactant.sigma), 0.523)))
        if not self.coating:
            self.H[-1][np.logical_not(self.coated[-1])] = 0
        self.h = a * self.H[-1]
        Rd = 1 - np.power((1 - self.H[-1]), 2)
        Rd[self.coated[-1]] = 0
        self.final_volumes += self.volumes[self.n_gen] - Rd * Va
        self.final_volumes[self.final_volumes < 0] = 0
        self.coated[-1][self.H[self.n_gen] > 0] = True
        # Coating ruptures
        idx_4 = np.full(2 ** self.n_gen, False)
        idx_4 = (self.volumes[self.n_gen] != 0.0) & (self.final_volumes == 0.0)
        if self.track_ruptures:
            self.coating_ruptures[self.n_gen][idx_4] = self.n_gen

    def compute_efficiency(self):
        """Compute the injection efficiency"""
        self.efficiency = 100 * np.sum(self.final_volumes) / self.injected_volume
        return self.efficiency
    
    def compute_std_inv(self):
        """Compute the injection homogeneity"""
        if self.efficiency == 0.0:
            self.std_inv = 0.0
            self.homogeneity = np.zeros(shape=(2 ** self.n_gen))
        else:
            # Distribution of final volumes fractions
            # with respect to the injected volume.
            self.distribution = 100 * self.final_volumes / (self.injected_volume)
            # Distribution of final volumes with respect to a perfect distribution
            # (same volume in all the nodes), then used to compute 1/SD.
            self.homogeneity = 100 * self.final_volumes / (self.efficiency * self.injected_volume / (2 ** self.n_gen))
            self.std_inv = 1 / np.std(self.homogeneity)
        return self.std_inv