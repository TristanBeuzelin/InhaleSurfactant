import numpy as np
import matplotlib.pyplot as plt
from src.quaternion import Quaternion
from matplotlib import animation
import matplotlib.cm as cm

class Tree_plot:
    def __init__(
        self,
        n_gen,
        first_vec,
        phi,
        gamma
    ):
        self.n_gen = n_gen
        self.first_vec = first_vec
        self.tree_phi = phi
        self.tree_gamma = gamma

        self.first_quat = Quaternion(
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
        first_normal = np.array((normal_quat.i, normal_quat.j, normal_quat.k))
        self.vecs = [[self.first_vec]]
        self.norm_vecs = [[self.first_vec]]
        self.normals = [[first_normal]]

    def quat_exp(
            self,
            theta,
            vec
        ):
        sin_vec = vec * np.sin(theta)
        quat = Quaternion(
            np.cos(theta),
            *sin_vec
        )
        return quat

    def generate_next_level(self, a, fig, ax):
        new_level_norm_vecs = []
        new_level_vecs = []
        new_level_normals = []
        for vec, normed_vec, normal_vec in zip(self.vecs[-1], self.norm_vecs[-1], self.normals[-1]):
            
            # Compute direction vector and normal vector for left and right branch
            normal = Quaternion(0, *normal_vec)

            qn = self.quat_exp(np.pi/8, normal_vec)
            left_quat = qn@Quaternion(0, *normed_vec)@qn.inv() / qn.squared_norm()

            qn = self.quat_exp(-np.pi/8, normal_vec)
            right_quat = qn@Quaternion(0, *normed_vec)@qn.inv() / qn.squared_norm()

            left_vec = np.array((left_quat.i, left_quat.j, left_quat.k))
            right_vec = np.array((right_quat.i, right_quat.j, right_quat.k))
            
            qnl1 = self.quat_exp(-np.pi / 4, normed_vec)
            qnl2 = self.quat_exp(np.pi / 8, normal_vec)
            left_normal = qnl2@qnl1@normal@qnl1.inv()/qnl1.squared_norm()@qnl2.inv() / qnl2.squared_norm()
            left_normal_vec = np.array((left_normal.i, left_normal.j, left_normal.k))
        
            qnr1 = self.quat_exp(np.pi / 4, normed_vec)
            qnr2 = self.quat_exp(-np.pi / 8, normal_vec)
            right_normal = qnr2@qnr1@normal@qnr1.inv()/qnl1.squared_norm()@qnr2.inv() / qnl2.squared_norm()
            right_normal_vec = np.array((right_normal.i, right_normal.j, right_normal.k))

            new_level_norm_vecs.append(left_vec)
            new_level_norm_vecs.append(right_vec)

            new_level_vecs.append(vec + left_vec * a)
            new_level_vecs.append(vec + right_vec * a)

            new_level_normals.append(left_normal_vec)
            new_level_normals.append(right_normal_vec)
            ax.plot(
                [vec[0], vec[0] + left_vec[0] * a],
                [vec[1], vec[1] + left_vec[1] * a],
                [vec[2], vec[2] + left_vec[2] * a],
                linewidth=1,
                color="grey"
            )
            ax.plot(
                [vec[0], vec[0] + right_vec[0] * a],
                [vec[1], vec[1] + right_vec[1] * a],
                [vec[2], vec[2] + right_vec[2] * a],
                linewidth=1,
                color="grey"
            )
        self.vecs.append(new_level_vecs)
        self.norm_vecs.append(new_level_norm_vecs)
        self.normals.append(new_level_normals)

    def loop(self):
        a = 1
        self.fig = plt.figure(dpi=200)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.plot(
            [0, self.first_vec[0]],
            [0, self.first_vec[1]],
            [0, self.first_vec[2]],
            linewidth=1,
            color="grey"
        )
        for _ in range(self.n_gen):
            self.generate_next_level(a, self.fig, self.ax)
            a *= 2 ** (-1/3)
        self.ax.set_aspect('equal', adjustable='box')
        

        #Hide axes ticks
        self.ax.grid(False)
        plt.axis('off')
        self.ax.view_init(elev=10, azim=90)

    def plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d") 
        for vec in self.vecs[-1]:
            self.ax.scatter(vec[0], vec[1], vec[2])
        
    
    def plot_end(
        self,
        volumes,
        path,
        filename
    ):
        x = []
        y = []
        z = []
        vols = []
        for i, coords in enumerate(self.vecs[-1]):
            vol = volumes[i]
            if vol == 0:
                pass
            else:
                x.append(coords[0])
                y.append(coords[1])
                z.append(coords[2])
                vols.append(vol)
        scatter = self.ax.scatter(
            x,
            y,
            z,
            linewidth=1,
            c=vols,
            cmap=cm.plasma,
            alpha=0.5
        )
        self.ax.view_init(elev=0.0, azim=90)

        cbar = self.fig.colorbar(scatter, location="left")
        cbar.ax.set_yticklabels(["{:.2f}".format(i) for i in cbar.get_ticks()])
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Normalized volume", fontsize=20)
        if self.n_gen == 8:
            scatter.set_clim(0.05, 2.5)
        else:
            scatter.set_clim(0.05, 5)
        plt.axis("off")
        plt.grid(visible=False)
        plt.savefig("".join((path, filename)), dpi=500, format="png")
    
    def animate(self, path, filename):
        def init():
            self.ax.view_init(elev=0., azim=90)
            return ()


        def animate(i):
            self.ax.view_init(elev=0., azim=2 * i)
            return ()

        anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                                    frames=180, interval=60, blit=True)
        anim.save("/".join((path, filename)), fps=30)
