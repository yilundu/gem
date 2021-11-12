from torch.utils.data import DataLoader
# from fid import get_fid_score
import collections
import numpy as np
import torch
from torchvision.utils import make_grid
from torch_fidelity import calculate_metrics
import os.path as osp
from imageio import imwrite
import os
from shutil import rmtree
from tqdm import tqdm


def dict_to_gpu(ob):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()


def compute_fid_nn(dataset, model, rank):
    bs = 32
    data_loader = DataLoader(dataset, shuffle=True, batch_size=bs, pin_memory=False, num_workers=4, drop_last=False)
    real_images = []
    fake_images = []

    torch.cuda.set_device(rank)
    imsize = dataset.im_size

    with torch.no_grad():
        for step, ((fake_model_input, fake_gt), (real_model_input, real_gt)) in enumerate(tqdm(data_loader)):
            fake_model_input = dict_to_gpu(fake_model_input)
            fake_gt = dict_to_gpu(fake_gt)
            real_gt = dict_to_gpu(real_gt)

            # Collate images for nearest neighbor calculation
            if len(fake_images) < 50000:
                closest_idx = (step == 0)
                fake_model_output = model(fake_model_input, prior_sample=True, render=True, closest_idx=closest_idx, manifold_model=False)
                real_image = real_gt['rgb'].detach().cpu().numpy().reshape((-1, imsize, imsize,  3))
                fake_image = fake_model_output['rgb'].detach().cpu().numpy().reshape(-1, imsize, imsize, 3)
                fake_image = (np.clip(fake_image, -1, 1) + 1) /  2.
                real_image = (real_image + 1) / 2.
                fake_image = (fake_image * 255).astype(np.uint8)
                fake_images.extend(list(fake_image))
                real_image = (real_image * 255).astype(np.uint8)
                real_images.extend(list(real_image))

            # Compute nearest neighbor in dataset
            if step == 0:
                real_image = real_gt['rgb'].detach().cpu().numpy().reshape((-1, imsize, imsize,  3))
                real_image = (real_image + 1) / 2.
                real_image = (real_image * 255).astype(np.uint8)
                start_im = fake_image[:32]
                goal_im = np.zeros((5, *start_im.shape))
                embed_dist = np.ones((5, goal_im.shape[1])) * 100000

                goal_im_latent = np.zeros((5, *start_im.shape))
                sort_idx = fake_model_output['sort_idx']
                idx_map = {}
                for i in range(sort_idx.shape[0]):
                    for j in range(sort_idx.shape[1]):
                        idx_map[sort_idx[i, j]] = (i, j)

                idxs = real_gt['idx'].detach().squeeze().cpu().numpy()

                # for count, idx in enumerate(idxs):
                for idx in idx_map.keys():
                    i, j = idx_map[idx]
                    inp = dataset[idx]
                    rgb = inp[1][1]['rgb']
                    rgb = rgb.view(imsize, imsize, 3).detach().cpu().numpy()
                    rgb = (rgb + 1) / 2
                    rgb = (rgb * 255).astype(np.uint8)

                    goal_im_latent[j, i] = rgb

                dist_bulk = np.square(real_image[None, :] / 255. - start_im[:, None] / 255.).mean(axis=2).mean(axis=2).mean(axis=2)
                for i in range(bs):
                    dist_i = dist_bulk[i]

                    for j, dist_im in enumerate(dist_i):
                        if dist_im < embed_dist[:, i].max():
                            replace_idx = embed_dist[:, i].argsort()[-1]
                            embed_dist[replace_idx, i] = dist_im
                            goal_im[replace_idx, i] = real_image[j]

            else:
                real_image = real_gt['rgb'].detach().cpu().numpy().reshape((-1, imsize, imsize,  3))
                real_image = (real_image + 1) / 2.
                real_image = (real_image * 255).astype(np.uint8)
                dist_bulk = np.abs(real_image[None, :] - start_im[:, None]).mean(axis=2).mean(axis=2).mean(axis=2)

                for i in range(bs):
                    dist_i = dist_bulk[i]

                    for j, dist_im in enumerate(dist_i):
                        if dist_im < embed_dist[:, i].max():
                            replace_idx = embed_dist[:, i].argsort()[-1]

                            embed_dist[replace_idx, i] = dist_im
                            goal_im[replace_idx, i] = real_image[j]

                idxs = real_gt['idx'].detach().squeeze().cpu().numpy()

                # for count, idx in enumerate(idxs):
                #     if idx in idx_map:
                #         i, j = idx_map[idx]
                #         goal_im_latent[j, i] = real_image[count]

            # if step > 5:
            #     break


    torch.cuda.empty_cache()

    real_path = "/tmp/real_image"
    fake_path = "/tmp/fake_image"

    if osp.exists(real_path):
        rmtree(real_path)

    if osp.exists(fake_path):
        rmtree(fake_path)

    os.mkdir(real_path)
    os.mkdir(fake_path)

    for i, real_image in enumerate(real_images):
        imwrite(osp.join(real_path, 'real_{}.png'.format(i)), real_image)

    for i, fake_image in enumerate(fake_images):
        imwrite(osp.join(fake_path, 'fake_{}.png'.format(i)), fake_image)

    if imsize == 32:
        metrics_dict = calculate_metrics(fake_path, real_path, cuda=True, isc=True, fid=True, kid=False, verbose=True)
    else:
        metrics_dict = calculate_metrics(fake_path, real_path, cuda=True, isc=False, fid=True, kid=False, verbose=True)

    # fid = get_fid_score(real_images, fake_images)
    # fid = 0.0
    print("FID scores: ", metrics_dict)
    fid = metrics_dict['frechet_inception_distance']
    # assert False

    goal_im = goal_im.transpose((1, 2, 0, 3, 4))
    goal_im_latent = goal_im_latent.transpose((1, 2, 0, 3, 4))
    panel_im = np.concatenate([start_im[:, :, None], goal_im], axis=2)
    panel_im_latent = np.concatenate([start_im[:, :, None], goal_im_latent], axis=2)

    panel_im = torch.Tensor(panel_im).permute(0, 2, 4, 1, 3).reshape((-1, 3, imsize, imsize))
    panel_im_latent = torch.Tensor(panel_im_latent).permute(0, 2, 4, 1, 3).reshape((-1, 3, imsize, imsize))

    panel_im = make_grid(panel_im, nrow=6, scale_each=False, normalize=True).numpy().transpose((1, 2, 0))
    panel_im_latent = make_grid(panel_im_latent, nrow=6, scale_each=False, normalize=True).numpy().transpose((1, 2, 0))
    assert False

    # panel_im = panel_im.reshape((32*128, 128*6, 3)).astype(np.uint8)
    # panel_im_latent = panel_im_latent.reshape((32*128, 128*6, 3)).astype(np.uint8)

    return fid, panel_im, panel_im_latent

    # writer.add_image(prefix + name, make_grid(img, scale_each=False, normalize=True),
    #                  global_step=total_steps)


