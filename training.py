'''Implements a generic training loop.
'''

import loss_functions
import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import collections
from collections import defaultdict
import torch.distributed as dist
from utils import get_mgrid


def dict_to_gpu(ob):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()


def average_gradients(model):
    """Averages gradients across workers"""
    size = float(dist.get_world_size())

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size


def multiscale_training(model, ema_model, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
                        dataloader_callback, dataloader_iters, dataloader_params,
                        val_loss_fn=None, summary_fn=None, iters_til_checkpoint=None, clip_grad=False,
                        overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0):


    model_dir_base = model_dir
    for i in range(1000):
        for params, max_steps in zip(dataloader_params, dataloader_iters):
            train_dataloader, val_dataloader = dataloader_callback(*params)
            model_dir = os.path.join(model_dir_base, '_'.join(map(str, params)))

            model, optimizers = train(model, ema_model, train_dataloader, epochs=10000, lr=lr, steps_til_summary=steps_til_summary,
                                      val_dataloader=val_dataloader, epochs_til_checkpoint=epochs_til_checkpoint, model_dir=model_dir, loss_fn=loss_fn,
                                      val_loss_fn=val_loss_fn, summary_fn=summary_fn, iters_til_checkpoint=iters_til_checkpoint,
                                      clip_grad=clip_grad, overwrite=overwrite, optimizers=optimizers, batches_per_validation=batches_per_validation,
                                      gpus=gpus, rank=rank, max_steps=max_steps)


def train(model, ema_model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None):

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]
    # schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200) for optimizer in optimizers]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if rank == 0:
        # if os.path.exists(model_dir):
        #     if overwrite:
        #         shutil.rmtree(model_dir)
        #     else:
        #         val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        #         if val == 'y' or overwrite:
        #             shutil.rmtree(model_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        utils.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        utils.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    total_steps = 0
    print("len data loader: ", len(train_dataloader), " w/ epochs: ", len(train_dataloader) * epochs)
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = dict_to_gpu(model_input)
                gt = dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    # torch.save(model.state_dict(),
                    #            os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)

                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                ema_model.update(model.parameters())

                # for scheduler in schedulers:
                #     scheduler.step()

                if rank == 0:
                    pbar.update(1)

                # if total_steps % 500 == 0:
                #     optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

                if not total_steps % steps_til_summary and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = dict_to_gpu(model_input)
                                gt = dict_to_gpu(gt)

                                model_output = model(model_input)
                                val_loss = val_loss_fn(model_output, gt)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(loss)
                                summary_fn(model, model_input, gt, model_output, writer, total_steps, 'val_')
                                writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    shadow_params = ema_model.shadow_params
                    ema_dict = {}
                    named_model_params = list(model.named_parameters())

                    for (k, v), param in zip(named_model_params, shadow_params):
                        ema_dict[k] = param

                    # torch.save({'model_dict': model.state_dict(),  'optimizer_dict': optimizers[0].state_dict()},
                    #            os.path.join(checkpoints_dir, 'model_{}_latest.pth'.format(total_steps)))
                    torch.save({'model_dict': model.state_dict()},
                               os.path.join(checkpoints_dir, 'model_{}.pth'.format(total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break

        # if rank == 0:
        #     torch.save(model.state_dict(),
        #                os.path.join(checkpoints_dir, 'model_final.pth'))
        #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
        #                np.array(train_losses))

        return model, optimizers


def train_autodecoder_gan(model, discriminator, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None):

    model_optim, disc_optim = [torch.optim.Adam(lr=lr, params=model.parameters()),
            torch.optim.Adam(lr=lr, params=discriminator.parameters(), betas=(0.0, 0.9))]

    # schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200) for optimizer in optimizers]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        utils.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        utils.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    total_steps = 0
    print("len data loader: ", len(train_dataloader), " w/ epochs: ", len(train_dataloader) * epochs)
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = dict_to_gpu(model_input)
                gt = dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    # torch.save(model.state_dict(),
                    #            os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)


                # Fake forward pass
                fake_model_output = model(model_input, prior_sample=True, render=False, manifold_model=False)

                # Real forward pass
                real_model_output = model(model_input, render=False, manifold_model=False)

                pred_fake = discriminator(fake_model_output, detach=True)  # Detach to make sure no gradients go into generator
                loss_d_fake = loss_functions.gan_loss(pred_fake, False)

                # Real forward step
                pred_real = discriminator(real_model_output, detach=True)
                loss_d_real = loss_functions.gan_loss(pred_real, True)

                disc_loss = 0.5 * (loss_d_real + loss_d_fake)

                if rank == 0:
                    writer.add_scalar('disc_fake', loss_d_fake, total_steps)
                    writer.add_scalar('disc_real', loss_d_real, total_steps)
                    writer.add_scalar('discriminator_loss', disc_loss, total_steps)

                discriminator.requires_grad_(False)
                pred_fake = discriminator(fake_model_output, detach=False)
                discriminator.requires_grad_(True)
                loss_g_gan_fake = loss_functions.gan_loss(pred_fake, True)

                if rank == 0:
                    writer.add_scalar('gan_gen', loss_g_gan_fake, total_steps)

                disc_loss = disc_loss + 0.01 * loss_g_gan_fake

                batch_size = real_model_output['representation'].shape[0]
                alpha = torch.rand(batch_size, 1).cuda()
                interpolated = alpha * real_model_output['representation'].data + (1 - alpha) * fake_model_output['representation'].data
                interpolated = interpolated.requires_grad_(True)

                # Calculate probability of interpolated examples
                prob_interpolated = discriminator({'representation':interpolated})
                input_list = [interpolated]

                # Calculate gradients of probabilities with respect to examples
                gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=input_list,
                                                grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                                                create_graph=True, retain_graph=True, allow_unused=True)

                gradients = [g for g in gradients if g is not None]
                gradients = torch.cat(gradients, dim=-1)
                # Gradients have shape (batch_size, num_channels, img_width, img_height),
                # so flatten to easily take norm per example in batch
                gradients = gradients.view(batch_size, -1)

                # Derivatives of the gradient close to 0 can cause problems because of
                # the square root, so manually calculate norm and add epsilon
                gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                gradient_penalty = ((gradients_norm - 1) ** 2).mean()

                disc_loss += gradient_penalty

                if rank == 0:
                    writer.add_scalar('gradient_penalty', gradient_penalty, total_steps)

                model_optim.zero_grad()
                disc_optim.zero_grad()

                disc_loss.backward()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)
                    average_gradients(discriminator)

                disc_optim.step()
                model_optim.step()

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_summary and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = dict_to_gpu(model_input)
                                gt = dict_to_gpu(gt)

                                model_output = model(model_input)
                                val_loss = val_loss_fn(model_output, gt)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(loss)
                                summary_fn(model, model_input, gt, model_output, writer, total_steps, 'val_')
                                writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    torch.save({'model_dict': model.state_dict(), 'disc_dict': discriminator.state_dict()},
                               os.path.join(checkpoints_dir, 'model_{}.pth'.format(total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break

        # if rank == 0:
        #     torch.save(model.state_dict(),
        #                os.path.join(checkpoints_dir, 'model_final.pth'))
        #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
        #                np.array(train_losses))

        return model, optimizers


def train_latent_gan(model, discriminator, data_loader, epochs, lr, steps_til_summary, model_dir,
                     loss_fn, summary_fn=None, iters_til_checkpoint=None, overwrite=True, optimizers=None, val_loader=None,
                     gt_model=None, gradient_penalty=False, r1_loss=True, real_reconstruction_loss=True, gpus=1, rank=0, val_dataset=None,
                     multimodal=False):
    if optimizers is None:
        model_optim, disc_optim = [torch.optim.Adam(lr=lr, params=model.generator.parameters(), betas=(0.0, 0.9)),
                                   torch.optim.Adam(lr=lr, params=discriminator.parameters(), betas=(0.0, 0.9))]

    if gt_model is None:
        gt_model = model


    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        utils.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        utils.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    total_steps = 0

    size = 64
    mgrid = utils.get_mgrid((size, size), dim=2)
    print("pre img mgrid: ", mgrid.shape)
    mgrid = mgrid[None, :, :].repeat(16, 1, 1)
    print("post img mgrid: ", mgrid.shape)

    if multimodal:
        img_mgrid = data_loader.dataset.real_dataset.dataset.img_mgrid
        img_coords = np.concatenate([img_mgrid, torch.zeros(img_mgrid.shape[0], 1)], axis=1)

        audio_mgrid = data_loader.dataset.real_dataset.dataset.audio_mgrid
        audio_coords = np.concatenate([torch.zeros(audio_mgrid.shape[0],2), audio_mgrid], axis=1) * 100  # account for diff in frequency

        mgrid = torch.from_numpy(np.vstack([img_coords, audio_coords]))

        print("mgrid: ", mgrid.shape)

        mgrid = mgrid[None, :, :].repeat(16, 1, 1)
    else:
        mgrid = utils.get_mgrid((size, size), dim=2)
        mgrid = mgrid[None, :, :].repeat(16, 1, 1)
    mgrid = mgrid.cuda()

    with tqdm(total=len(data_loader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            for step, ((fake_model_input, fake_gt), (real_model_input, real_gt)) in enumerate(data_loader):
                if ((total_steps % 10000) == 0) and rank == 0:
                    from metrics import compute_fid_nn
                    if not multimodal:
                        fid_score, panel_im, panel_im_latent = compute_fid_nn(val_dataset, model, rank)
                        writer.add_scalar('fid_score', fid_score, total_steps)
                        writer.add_image('nn', panel_im,
                                         global_step=total_steps, dataformats="HWC")
                        writer.add_image('nn_latent', panel_im_latent,
                                         global_step=total_steps, dataformats="HWC")
                model_loss = 0.

                fake_model_input = dict_to_gpu(fake_model_input)
                real_model_input = dict_to_gpu(real_model_input)
                real_gt = dict_to_gpu(real_gt)
                fake_gt = dict_to_gpu(fake_gt)

                # Fake forward pass
                fake_model_output = model(fake_model_input, prior_sample=True, render=False, manifold_model=False)

                # Real forward pass
                real_model_output = gt_model(real_model_input, render=real_reconstruction_loss, manifold_model=False)

                if real_reconstruction_loss:
                    real_losses = loss_fn(real_model_output, real_gt)

                    for loss_name, loss in real_losses.items():
                        single_loss = loss.mean()

                        writer.add_scalar("real_" + loss_name, single_loss, total_steps)
                        model_loss += single_loss

                # Discriminator forward passes
                # Fake forward step
                pred_fake = discriminator(fake_model_output, detach=True)  # Detach to make sure no gradients go into generator
                loss_d_fake = loss_functions.gan_loss(pred_fake, False)

                # Real forward step
                pred_real = discriminator(real_model_output, detach=True)
                loss_d_real = loss_functions.gan_loss(pred_real, True)

                disc_loss = 0.5 * (loss_d_real + loss_d_fake)

                # Gradient penalty
                if gradient_penalty:
                    # Calculate interpolation
                    batch_size = real_model_output['representation'].shape[0]
                    alpha = torch.rand(batch_size, 1).cuda()
                    interpolated = alpha * real_model_output['representation'].data + (1 - alpha) * fake_model_output['representation'].data
                    interpolated = interpolated.requires_grad_(True)

                    representations = [alpha * rep.clone().detach() + (1-alpha) * rep_neg.clone().detach() for rep, rep_neg in zip(real_model_output['representations'], fake_model_output['representations'])]
                    representations = [rep.requires_grad_(True) for rep in representations]

                    # Calculate probability of interpolated examples
                    prob_interpolated = discriminator({'representation':interpolated, 'representations': representations})
                    input_list = [interpolated] + representations

                    # Calculate gradients of probabilities with respect to examples
                    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=input_list,
                                                    grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                                                    create_graph=True, retain_graph=True, allow_unused=True)

                    gradients = [g for g in gradients if g is not None]
                    gradients = torch.cat(gradients, dim=-1)
                    # Gradients have shape (batch_size, num_channels, img_width, img_height),
                    # so flatten to easily take norm per example in batch
                    gradients = gradients.view(batch_size, -1)

                    # Derivatives of the gradient close to 0 can cause problems because of
                    # the square root, so manually calculate norm and add epsilon
                    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                    gradient_penalty = ((gradients_norm - 1) ** 2).mean()

                    disc_loss += gradient_penalty

                    if rank == 0:
                        writer.add_scalar('gradient_penalty', gradient_penalty, total_steps)

                if r1_loss:
                    # Calculate interpolation
                    batch_size = real_model_output['representation'].shape[0]
                    alpha = torch.rand(batch_size, 1).cuda()
                    interpolated = real_model_output['representation'].clone().detach().requires_grad_()
                    representations = [rep.clone().detach().requires_grad_() for rep in real_model_output['representations']]

                    # Calculate probability of interpolated examples
                    prob_interpolated = discriminator({'representation':interpolated, 'representations': representations})
                    input_list = [interpolated] + representations

                    # Calculate gradients of probabilities with respect to examples
                    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=input_list,
                                                    grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                                                    create_graph=True, retain_graph=True)

                    gradients = torch.cat(gradients, dim=-1)

                    # Gradients have shape (batch_size, num_channels, img_width, img_height),
                    # so flatten to easily take norm per example in batch
                    gradients = gradients.view(batch_size, -1)
                    r1_loss = gradients.pow(2).sum(dim=1).mean()

                    disc_loss += r1_loss

                    if rank == 0:
                        writer.add_scalar('rl_loss', r1_loss, total_steps)

                if rank == 0:
                    writer.add_scalar('disc_fake', loss_d_fake, total_steps)
                    writer.add_scalar('disc_real', loss_d_real, total_steps)
                    writer.add_scalar('discriminator_loss', disc_loss, total_steps)


                disc_loss.backward()

                if gpus > 1:
                    average_gradients(discriminator)
                #     # average_gradients(discriminator)

                disc_optim.step()
                disc_optim.zero_grad()

                # Generator forward pass
                # Try to fake discriminator
                pred_fake = discriminator(fake_model_output, detach=False)
                loss_g_gan_fake = loss_functions.gan_loss(pred_fake, True)

                if rank == 0:
                    writer.add_scalar('generator_loss_fake', loss_g_gan_fake, total_steps)

                model_loss += loss_g_gan_fake

                model_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                model_optim.step()
                model_optim.zero_grad()

                if not total_steps % steps_til_summary and rank == 0:
                    with torch.no_grad():
                        fake_model_input['context']['x'] = mgrid
                        fake_model_input['context']['idx'] = fake_model_input['context']['idx'][:16]
                        real_model_input['context']['x'] = mgrid
                        real_model_input['context']['idx'] = real_model_input['context']['idx'][:16]
                        fake_model_output = model(fake_model_input, prior_sample=True, render=True)
                        real_model_output = gt_model(real_model_input, render=True)
                        torch.save({'model_state_dict': model.state_dict(), 'discriminator_state_dict': discriminator.state_dict},
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, fake_model_input, fake_gt, fake_model_output, writer, total_steps, prefix='fake_')

                    writer.add_histogram('fake_representation', fake_model_output['representation'], total_steps)
                    writer.add_histogram('real_representation', real_model_output['representation'], total_steps)

                    summary_fn(model, real_model_input, real_gt, real_model_output, writer, total_steps, prefix='real_')

                if rank == 0:
                    pbar.update(1)

                print("done with step")

                if not total_steps % steps_til_summary and rank == 0:
                    print("Epoch %d, Total loss %0.6f" % (epoch, model_loss+disc_loss))

                    if val_loader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_loader):
                                model_input = dict_to_gpu(model_input)
                                gt = dict_to_gpu(gt)

                                model_output = model(model_input, prior_sample=True)
                                val_loss = loss_fn(model_output, gt)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())
                                break

                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(loss)
                                summary_fn(model, model_input, gt, model_output, writer, total_steps, 'val_')
                                writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    # pass
                    torch.save({'model_state_dict': model.state_dict(), 'discriminator_state_dict': discriminator.state_dict},
                               os.path.join(checkpoints_dir, 'model_latest.pth' % (epoch, total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1

        if rank == 0:
            torch.save({'model_state_dict': model.state_dict(), 'discriminator_state_dict': discriminator.state_dict},
                       os.path.join(checkpoints_dir, 'model_final.pth'))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                       np.array(train_losses))


def train_conv_gan(model, discriminator, data_loader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir,
                   summary_fn=None, iters_til_checkpoint=None, overwrite=True, optimizers=None, val_loader=None):
    if optimizers is None:
        model_optim, disc_optim = [torch.optim.Adam(lr=lr, betas=(0., 0.9), params=model.parameters()),
                                   torch.optim.Adam(lr=lr, betas=(0., 0.9), params=discriminator.parameters())]

    if os.path.exists(model_dir):
        if overwrite:
            shutil.rmtree(model_dir)
        else:
            val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
            if val == 'y' or overwrite:
                shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(data_loader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(data_loader):
                model_loss = 0.

                # Fake forward pass
                model_input = dict_to_gpu(model_input)
                gt = dict_to_gpu(gt)

                fake_model_output = model(model_input)

                # Discriminator forward passes
                # Fake forward step
                pred_fake = discriminator(fake_model_output['rgb'], detach=True)  # Detach to make sure no gradients go into generator
                loss_d_fake = loss_functions.gan_loss(pred_fake, False)

                # Real forward step
                pred_real = discriminator(gt['rgb'], detach=True)
                loss_d_real = loss_functions.gan_loss(pred_real, True)

                disc_loss = 0.5 * (loss_d_real + loss_d_fake)
                writer.add_scalar('discriminator_loss', disc_loss, total_steps)

                disc_loss.backward()
                disc_optim.step()
                disc_optim.zero_grad()

                # Generator forward pass
                # Try to fake discriminator
                # fake_model_output_det_z = model(fake_model_input, real=False, detach_z=True, render=False)
                pred_fake = discriminator(fake_model_output['rgb'])
                loss_g_gan_fake = loss_functions.gan_loss(pred_fake, True)
                writer.add_scalar('generator_loss_fake', loss_g_gan_fake, total_steps)

                generator_loss = loss_g_gan_fake

                generator_loss.backward()
                model_optim.step()
                model_optim.zero_grad()
                disc_optim.zero_grad()

                writer.add_scalar("gen_loss", generator_loss, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, fake_model_output, writer, total_steps, prefix='fake_')

                pbar.update(1)

                print("done with step")

                if not total_steps % steps_til_summary:
                    print("Epoch %d, Total loss %0.6f" % (epoch, model_loss+disc_loss))

                    if val_loader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_loader):
                                model_input = dict_to_gpu(model_input)
                                gt = dict_to_gpu(gt)

                                model_output = model(model_input, prior_sample=True, real=False)
                                break

                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(loss)
                                summary_fn(model, model_input, gt, model_output, writer, total_steps, 'val_')
                                writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint):
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
