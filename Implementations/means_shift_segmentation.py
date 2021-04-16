import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_color(img, title=""):
    plt.imshow(img, vmin=0, vmax=255)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    print("Size :", img.shape)
    print("dtype :", img.dtype)
    print("max:", np.max(img), " min:", np.min(img))


def plot_3d_map(img):
    r, g, b = cv2.split(img)
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(r, g, b)
    plt.show()


def get_rand(start, end):
    a = list(range(start, end))
    np.random.shuffle(a)
    return a


def arr(*args):
    a = []
    for i in args:
        a.append(i)
    return np.array(a).astype(int)


class ImageProcessor:
    def __init__(self, image):
        self.image = image
        self.output_cluster = image.copy()
        self.filtered = image.copy()
        self.shape = image.shape

    @classmethod
    def get_dists(cls, points, point):
        return np.sum(np.square(points - point), axis=1)

    @classmethod
    def get_dist(cls, point1, point2):
        return np.linalg.norm(point2 - point1)

    def cluster(self, k, ITER_LIM):
        img = self.filtered.copy()
        output = img.copy()
        shape = self.shape
        features = np.zeros((shape[0] * shape[1], 5))
        for i in range(shape[0]):
            for j in range(shape[1]):
                temp = img[i][j]
                features[i * shape[1] + j] = arr(i, j, temp[0], temp[1], temp[2])

        np.random.shuffle(features)

        clusters = features[:k]
        # points = np.array([int(k * np.random.randint()) for _ in range(features.shape[0])])
        points = np.random.randint(k, size=features.shape[0])
        for cl_ix in range(k):
            points[cl_ix] = cl_ix

        converged = False
        iteration = 0

        # dist points among the clstrs
        for ftr_idx in range(features.shape[0]):
            ar = ImageProcessor.get_dists(features[ftr_idx], clusters)
            points[ftr_idx] = np.argmin(ar)

        while not converged:
            sys.stdout.write('\r' + 'Iteration ' + str(iteration))
            points_cpy = points.copy()

            for ftr_idx in range(features.shape[0]):
                ar = ImageProcessor.get_dists(features[ftr_idx], clusters)
                points[ftr_idx] = np.argmin(ar)

            if (points_cpy == points).all() or iteration > ITER_LIM:
                converged = True

            for cl_idx in range(k):
                clstr_fts = (points == cl_idx)[..., np.newaxis] * features
                clusters[cl_idx] = (np.sum(clstr_fts, axis=0) / (np.count_nonzero(clstr_fts) / 5)).astype(int)
            iteration += 1

        for ft_id in range(features.shape[0]):
            cid = int(points[ft_id])
            ftr = features[ft_id].astype(int)
            if ((shape[:2] - ftr[:2]) > 0).all():
                output[ftr[0]][ftr[1]] = clusters[cid][2:5]

        self.output_cluster = output
        print('\nLabels : ', points)

    def filter(self, SIGMA_SP, SIGMA_RGB, N, ITER, IMG_UPD_SIZE):
        for _ in range(ITER):
            print('Iteration ', _, ':')
            self.run(SIGMA_SP, SIGMA_RGB, N, IMG_UPD_SIZE)

    def run(self, SIGMA_SP, SIGMA_RGB, N, IMG_UPD_SIZE):
        img = self.filtered.copy()
        shape = self.shape
        output = self.output_cluster
        features = np.zeros((shape[0], shape[1], 5))

        for i in range(shape[0]):
            for j in range(shape[1]):
                temp = img[i][j]
                features[i][j] = arr(i, j, temp[0], temp[1], temp[2])

        eta = 0.01
        w_size = N
        w_size_2 = int(N / 2)
        w_img_upd = IMG_UPD_SIZE
        w_img_upd_2 = int(w_img_upd / 2)
        with tqdm(total=(shape[0] - 2 * w_size_2) * (shape[1] - 2 * w_size_2)) as progress:
            r_px, r_py = get_rand(w_size_2, shape[0] - w_size_2), get_rand(w_size_2, shape[1] - w_size_2)
            avg_steps = conv_count = div_count = 0

            for px in r_px:
                for py in r_py:
                    converged = False
                    cluster_center = features[px][py]
                    steps = 0
                    while not converged:
                        prev_center = cluster_center.copy()

                        win_features = features[px - w_size_2:px + w_size_2 + 1, py - w_size_2:py + w_size_2 + 1, :]
                        win_ft_diff = win_features - cluster_center
                        win_ft_diff_sq = np.square(win_ft_diff)

                        win_ft_sp = np.sum(win_ft_diff_sq[:, :, :2], axis=2)
                        win_ft_rgb = np.sum(win_ft_diff_sq[:, :, 2:5], axis=2)

                        sp_wts = np.exp(-1 * win_ft_sp / (2 * (SIGMA_SP ** 2)))
                        rgb_wts = np.exp(-1 * win_ft_rgb / (2 * (SIGMA_RGB ** 2)))

                        wts = np.multiply(sp_wts, rgb_wts)
                        mean_shift = (wts[..., np.newaxis] * win_features) / np.sum(wts)
                        mean_shift = np.sum(mean_shift.reshape((w_size * w_size, 5)), axis=0)
                        cluster_center = mean_shift.astype(int)
                        # print(cluster_center,end='')
                        steps += 1

                        if np.linalg.norm(cluster_center - prev_center) < eta or steps > 50:
                            converged = True
                            output[px][py] = cluster_center[2:5]
                            temp = cluster_center[2:5]
                            # features[px][py] = arr(px, py, temp[0], temp[1], temp[2])
                            temp = features[px - w_img_upd_2:px + w_img_upd_2 + 1,
                                   py - w_img_upd_2:py + w_img_upd_2 + 1, 2:5].reshape((w_img_upd * w_img_upd, 3))
                            avg_color = np.sum(temp, axis=0) / (w_img_upd * w_img_upd)

                            features[px][py] = arr(px, py, avg_color[0], avg_color[1], avg_color[2])

                            if steps > 50:
                                div_count += 1
                            else:
                                conv_count += 1
                                avg_steps += steps

                    progress.update(1)
        print('converged:', conv_count, ' diverged:', div_count, ' average_shifts:', avg_steps / conv_count)
        self.filtered = output


## PARAMS ##
SIGMA_SP = 9
SIGMA_RGB = 18
BAND_SIZE = 23
ITER = 1
IMG_UPD_SIZE = 3
## END PARAMS ##


flower = cv2.cvtColor(cv2.imread('flower.png'), cv2.COLOR_BGR2RGB)
parrot = cv2.cvtColor(cv2.imread('parrot.png'),cv2.COLOR_BGR2RGB)

flowerShift = ImageProcessor(flower)

flowerImage.filter(SIGMA_SP, SIGMA_RGB, BAND_SIZE, ITER,IMG_UPD_SIZE)
plot_color(flowerImage.filtered)

flowerShift.cluster(100,ITER_LIM=10)
plot_color(flowerShift.output_cluster)
plot_3d_map(flowerShift.filtered)

parrotImage = ImageProcessor(parrot)
parrotImage.filter(SIGMA_SP, SIGMA_RGB, BAND_SIZE, ITER,IMG_UPD_SIZE)
plot_color(parrotImage.filtered)

parrotImage.cluster(100,ITER_LIM=20)
plot_color(parrotImage.output_cluster)
