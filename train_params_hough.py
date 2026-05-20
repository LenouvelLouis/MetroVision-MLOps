import os
import numpy as np
import scipy.io
from myMetroProcessing import processOneMetroImage
from PIL import Image

# Paths and settings
challenge_dir = "BD_CHALLENGE"
gt_mat = "GTCHALLENGETEST.mat"
resize_factor = 1

# List of images to process
image_files = []
for fname in sorted(os.listdir(challenge_dir)):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        # extract image index
        idx = int(fname.split('(')[1].split(')')[0])
        image_files.append((idx, fname))

# Hyperparameter grid for HoughCircles
dp_list       = [1.0, 1.2]
minDist_list  = [20, 25, 30]
param1_list   = [100, 150, 200]
param2_list   = [40, 50, 60]
minRad_list   = [20, 25, 30]
maxRad_list   = [80, 100]

# Function to compute accuracy given BDTEST file
def compute_accuracy(gt_path, test_path, resize_factor):
    BDREF = scipy.io.loadmat(gt_path)['BD']
    BDTEST = scipy.io.loadmat(test_path)['BD']
    # scale ground truth boxes
    BDREF[:,1:5] *= resize_factor

    # centroids and diameters
    I = np.mean(BDREF[:,1:3], axis=1)
    J = np.mean(BDREF[:,3:5], axis=1)
    D = np.round((BDREF[:,2] - BDREF[:,1] + BDREF[:,4] - BDREF[:,3]) / 2)
    max_dec = 0.1

    # init matrices
    conf = np.zeros((14,14), dtype=int)
    plus = np.zeros(14, dtype=int)
    minus = np.zeros(14, dtype=int)
    processed = np.zeros(BDREF.shape[0], dtype=bool)

    for row in BDTEST:
        n, y1, y2, x1, x2, cls = map(int, row)
        inds = np.where(BDREF[:,0] == n)[0]
        i, j = np.mean([y1, y2]), np.mean([x1, x2])
        if len(inds) == 0:
            plus[cls-1] += 1
            continue
        dists = np.sqrt((I[inds]-i)**2 + (J[inds]-j)**2)
        k = inds[np.argmin(dists)]
        if np.min(dists) <= max_dec * D[k]:
            conf[int(BDREF[k,5])-1, cls-1] += 1
            processed[k] = True
        else:
            plus[cls-1] += 1
    for k in np.where(~processed)[0]:
        minus[int(BDREF[k,5])-1] += 1

    TP = np.trace(conf)
    FN = np.sum(minus)
    FP = np.sum(plus)
    accuracy = TP / (TP + FP + FN)
    return accuracy

# Main grid search
def optimize_hough():
    best_acc = 0.0
    best_params = None

    for dp in dp_list:
        for minDist in minDist_list:
            for p1 in param1_list:
                for p2 in param2_list:
                    for rmin in minRad_list:
                        for rmax in maxRad_list:
                            # set Hough parameters
                            from myMetroProcessing import hough_params
                            hough_params.update({
                                'dp': dp,
                                'minDist': minDist,
                                'param1': p1,
                                'param2': p2,
                                'minRadius': rmin,
                                'maxRadius': rmax
                            })

                            # process all images
                            BDTEST = []
                            for idx, fname in image_files:
                                path = os.path.join(challenge_dir, fname)
                                im = np.array(Image.open(path).convert('RGB'))
                                im = (im * 255).astype(np.uint8) if im.max() <= 1.0 else im.astype(np.uint8)
                                _, bd = processOneMetroImage(fname, im, idx, resize_factor)
                                BDTEST.extend(bd.tolist())

                            # save temporary results
                            temp_mat = 'temp_BDTEST.mat'
                            scipy.io.savemat(temp_mat, {'BD': np.array(BDTEST, dtype=int)})

                            # evaluate
                            acc = compute_accuracy(gt_mat, temp_mat, resize_factor)
                            print(f"Params dp={dp}, minDist={minDist}, p1={p1}, p2={p2}, rmin={rmin}, rmax={rmax} -> acc={acc:.4f}")

                            # update best
                            if acc > best_acc:
                                best_acc = acc
                                best_params = dict(dp=dp, minDist=minDist, param1=p1,
                                                   param2=p2, minRadius=rmin, maxRadius=rmax)

    print("\nBest accuracy:", best_acc)
    print("Best Hough params:", best_params)

if __name__ == '__main__':
    optimize_hough()