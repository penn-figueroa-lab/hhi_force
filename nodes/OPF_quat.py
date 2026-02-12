import numpy as np
from scipy.stats import norm
from scipy.spatial.transform import Rotation as R
import pybullet as pb

def average_quaternions(quaternions, weights):
    """Compute a weighted average of unit quaternions."""
    A = np.zeros((4, 4))
    for q, w in zip(quaternions, weights):
        q = q.reshape(4, 1)
        A += w * (q @ q.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvecs[:, np.argmax(eigvals)]

class OPF_3d:
    def __init__(self, num_particles=5000, name=None, objid=None):
        self.name = name
        self.objid = objid
        self.num_particles = num_particles

        self.particles = np.random.uniform(-1, 1, size=(self.num_particles, 3))
        self.particles1 = R.random(self.num_particles).as_quat()  # (x, y, z, w)
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.weights1 = np.ones(self.num_particles) / self.num_particles

        self.curr_pos = np.average(self.particles, weights=self.weights, axis=0)
        self.curr_quat = average_quaternions(self.particles1, self.weights1)
        self.cov = np.cov(self.particles.T, aweights=self.weights)

        self.trajectory = [np.hstack((self.curr_pos, self.curr_quat))]
        self.R = 0.1
        self.R1 = 0.1

    def predict(self):
        std = 0.1
        if len(self.trajectory) < 2:
            vec = np.zeros(3)
        else:
            vec = (self.trajectory[-1][:3] - self.trajectory[-2][:3]) / 5

        self.particles += vec + np.random.randn(*self.particles.shape) * std

        # Apply random rotation noise to quaternion particles
        rot_noise = R.random(self.num_particles).as_quat()
        self.particles1 = R.from_quat(self.particles1) * R.from_quat(rot_noise)
        self.particles1 = self.particles1.as_quat()
        self.particles1 /= np.linalg.norm(self.particles1, axis=1, keepdims=True)

    def update(self, measurement, hidden=0):
        """measurement = [x, y, z, qx, qy, qz, qw]"""
        self.weights.fill(1.0)
        self.weights1.fill(1.0)

        trans = measurement[:3]
        quat = np.array(measurement[3:])

        dist = np.linalg.norm(self.particles - trans, axis=1)
        dot = np.abs(np.sum(self.particles1 * quat, axis=1))  # cosine of angle
        ang_dist = 1 - dot ** 2

        self.weights *= norm(dist, self.R).pdf(0)
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

        self.weights1 *= norm(ang_dist, self.R1).pdf(0)
        self.weights1 += 1.e-300
        self.weights1 /= np.sum(self.weights1)

        self.curr_pos = np.average(self.particles, weights=self.weights, axis=0)
        self.curr_quat = average_quaternions(self.particles1, self.weights1)
        self.trajectory.append(np.hstack((self.curr_pos, self.curr_quat)))

    def systematic_resample(self):
        N = self.num_particles
        positions = (np.arange(N) + np.random.random()) / N

        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum1 = np.cumsum(self.weights1)

        indexes = np.zeros(N, 'i')
        indexes1 = np.zeros(N, 'i')

        i = j = 0
        while i < N and j < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        i = j = 0
        while i < N and j < N:
            if positions[i] < cumulative_sum1[j]:
                indexes1[i] = j
                i += 1
            else:
                j += 1

        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights /= np.sum(self.weights)

        self.particles1[:] = self.particles1[indexes1]
        self.weights1[:] = self.weights1[indexes1]
        self.weights1 /= np.sum(self.weights1)

        # Ensure unit quaternions
        self.particles1 /= np.linalg.norm(self.particles1, axis=1, keepdims=True)

    def plot(self, image, projectionMatrix, viewMatrix, scale=12):
        import cv2
        # Compute covariance ellipse
        vals, vecs = np.linalg.eigh(5.9915 * self.cov)
        alpha = np.arctan2(vecs[1, 1], vecs[1, 0]) if vals[0] <= vals[1] else np.arctan2(vecs[0, 1], vecs[0, 0])
        mean = np.average(self.particles, weights=self.weights, axis=0)
        var = np.average((self.particles - mean) ** 2, weights=self.weights, axis=0)
        axs = int(np.sqrt(var[0]) * 500), int(np.sqrt(var[1]) * 500)
        angle = int(180. * alpha / np.pi)
        # Project mean position to image
        mean_coord = self._world2camera(projectionMatrix, viewMatrix, mean[:3], scale)
        cv2.ellipse(image, tuple(mean_coord[:2]), axs, angle, 0, 360, (0, 255, 255), 1)
        # Draw 3D orientation axes
        quat = self.curr_quat
        rot = pb.getMatrixFromQuaternion(quat)
        rotation_matrix = np.array(rot).reshape((3, 3))
        axis_x = np.dot(rotation_matrix, np.array([1, 0, 0]))
        axis_y = np.dot(rotation_matrix, np.array([0, 1, 0]))
        axis_z = np.dot(rotation_matrix, np.array([0, 0, 1]))
        axis_length = 30
        cv2.line(image, mean_coord,
                (mean_coord[0] + int(axis_length * axis_x[1]),
                mean_coord[1] + int(axis_length * axis_x[0])), (255, 0, 0), 2)
        cv2.line(image, mean_coord,
                (mean_coord[0] + int(axis_length * axis_y[1]),
                mean_coord[1] + int(axis_length * axis_y[0])), (0, 255, 0), 2)
        cv2.line(image, mean_coord,
                (mean_coord[0] + int(axis_length * axis_z[1]),
                mean_coord[1] + int(axis_length * axis_z[0])), (0, 0, 255), 2)
        # Save overlayed image for debugging
        cv2.imwrite('opf_frame.png', image)
    def _world2camera(self, projectionMatrix, viewMatrix, point3D, scale=12):
        point4D = np.append(point3D, 1)
        pv = np.dot(np.dot(projectionMatrix, viewMatrix), point4D)
        pv = pv / pv[3]
        return [int(pv[0] * scale), int(pv[1] * scale), int(pv[2] * scale)]