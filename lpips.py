# Adapted from https://github.com/richzhang/PerceptualSimilarity
#


import argparse
import os
import lpips
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dir', type=str, default='./imgs/ex_dir_pair')
parser.add_argument('-o', '--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v', '--version', type=str, default='0.1')
parser.add_argument('--all-pairs', action='store_true',
                    help='turn on to test all N(N-1)/2 pairs, leave off to just do consecutive pairs (N-1)')
parser.add_argument('-N', type=int, default=None)
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

args = parser.parse_args()

lpips_fn = lpips.LPIPS(net='alex', version=args.version)
if(args.use_gpu):
    lpips_fn.cuda()

f = open(args.out, 'w')
list_avg_dist = []

for i in range(200):
    img_dir = os.path.join(args.dir, str(i))
    files = os.listdir(img_dir)
    if args.N is not None:
        files = files[:args.N]
    F = len(files)

    dists = []
    for (ff, file) in enumerate(files[:-1]):
        img0 = lpips.im2tensor(lpips.load_image(os.path.join(img_dir, file)))
        if(args.use_gpu):
            img0 = img0.cuda()

        files1 = files[ff + 1:]
        for file1 in files1:
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(img_dir, file1)))
            if(args.use_gpu):
                img1 = img1.cuda()

            # Compute distance
            dist01 = lpips.forward(img0, img1)
            f.writelines('(%s,%s): %.6f\n' % (file, file1, dist01))

            dists.append(dist01.item())

    avg_dist = np.mean(np.array(dists))
    stderr_dist = np.std(np.array(dists)) / np.sqrt(len(dists))

    f.writelines('Avg: %.6f +/- %.6f' % (avg_dist, stderr_dist))
    list_avg_dist.append(avg_dist)

print('\n\nAvg across all test sets: %.5f' % np.mean(list_avg_dist))
f.writelines('\n\nAvg across all test sets: %.5f' % np.mean(list_avg_dist))


f.close()