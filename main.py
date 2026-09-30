import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as L
from lightning.pytorch.cli import LightningCLI

from diffusion import DiffusionBridge
from backbones.ncsnpp import NCSNpp
from backbones.discriminator import Discriminator_large
from backbones.fusion import AdaptiveFusionNetwork
from datasets import DataModule
from utils import compute_metrics, save_image_pair, save_preds, save_eval_images
from dct import dct2_torch, idct2_torch


class BridgeRunner(L.LightningModule):
    """Dynamic Dual-Domain Diffusion Bridge (D3Bridge).

    Two domain-specific diffusion bridge experts operate in the spatial
    domain and on DCT coefficients (frequency domain). A step-wise dynamic
    gating network g_psi coordinates the estimates of the two experts
    throughout recursive diffusion refinement, and an adaptive fusion
    network F_omega integrates the dual-domain predictions into the final
    synthesized image.
    """

    def __init__(
        self,
        generator_params,
        discriminator_params,
        diffusion_params,
        lr_g,
        lr_d,
        disc_grad_penalty_freq,
        disc_grad_penalty_weight,
        lambda_rec_loss,
        optim_betas,
        eval_mask,
        eval_subject,
        generator_params_freq=None,
        lambda_con=0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.disc_grad_penalty_freq = disc_grad_penalty_freq
        self.disc_grad_penalty_weight = disc_grad_penalty_weight
        self.lambda_rec_loss = lambda_rec_loss
        self.optim_betas = optim_betas
        self.eval_mask = eval_mask
        self.eval_subject = eval_subject
        self.n_steps = diffusion_params['n_steps']
        self.n_recursions = diffusion_params['n_recursions']

        # Spatial expert: generator and discriminator
        self.generator = NCSNpp(**generator_params)
        self.discriminator = Discriminator_large(**discriminator_params)

        # Frequency expert: operates on DCT coefficients of the images.
        # Reuses the same architecture; can be made smaller via `generator_params_freq`.
        if generator_params_freq is None:
            generator_params_freq = dict(generator_params)
        self.generator_freq = NCSNpp(**generator_params_freq)

        disc_freq_params = dict(discriminator_params)
        self.discriminator_freq = Discriminator_large(**disc_freq_params)

        # Step-wise dynamic gating network g_psi (Fig. 1B): outputs
        # complementary spatial/frequency weights for the two estimates.
        self.gate_net = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 2, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

        # Adaptive fusion network F_omega (Fig. 1C)
        out_ch = generator_params.get("in_ch", 1) if "in_ch" in generator_params else 1
        self.fusion_net = AdaptiveFusionNetwork(in_channels=out_ch)

        # Diffusion bridges for the spatial and frequency domains
        self.diffusion = DiffusionBridge(**diffusion_params)
        self.diffusion_freq = DiffusionBridge(**diffusion_params)

        self.lambda_con = lambda_con

    def training_step(self, batch):
        # Dataset returns (target, source, idx). The frequency (DCT)
        # representations are computed here and fed to the frequency expert.
        x0, y, _ = batch
        device = x0.device

        x0 = x0.float()
        y = y.float()

        # DCT representations: (B, C, H, W) -> (B, C, H, W) coefficients
        x0_freq = dct2_torch(x0)
        y_freq = dct2_torch(y)

        # Optimizer/scheduler layout:
        # [spatial G, spatial D, freq G, freq D, fusion]
        optimizers = self.optimizers()
        optimizer_g = optimizers[0]
        optimizer_d = optimizers[1]
        optimizer_gf = optimizers[2]
        optimizer_df = optimizers[3]
        optimizer_fusion = optimizers[4]

        schedulers = self.lr_schedulers()
        scheduler_g = schedulers[0]
        scheduler_d = schedulers[1]
        scheduler_gf = schedulers[2]
        scheduler_df = schedulers[3]
        scheduler_fusion = schedulers[4]

        # ---------------------------
        # Part 1: Train spatial discriminator
        # ---------------------------
        self.toggle_optimizer(optimizer_d)

        # Real samples
        t = torch.randint(1, self.n_steps+1, (x0.shape[0],)).to(x0.device)
        x_tm1 = self.diffusion.q_sample(t - 1, x0, y)
        x_t = self.diffusion.q_sample(t, x0, y)
        x_t.requires_grad = True

        disc_out = self.discriminator(x_tm1, x_t, t)
        real_loss = self.adversarial_loss(disc_out, is_real=True)

        if self.global_step % self.disc_grad_penalty_freq == 0:
            grads = torch.autograd.grad(outputs=disc_out.sum(), inputs=x_t, create_graph=True)[0]
            grad_penalty = (grads.view(grads.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = grad_penalty * self.disc_grad_penalty_weight
            real_loss += grad_penalty

        # Fake samples via the spatial generator
        x0_r = torch.zeros_like(x_t)
        for _ in range(self.n_recursions):
            x0_r = self.generator(torch.cat((x_t.detach(), y), axis=1), t, x_r=x0_r)
        x0_pred_spatial = x0_r

        x_tm1_pred = self.diffusion.q_posterior(t, x_t, x0_pred_spatial, y)
        disc_out = self.discriminator(x_tm1_pred, x_t, t)
        fake_loss = self.adversarial_loss(disc_out, is_real=False)

        d_loss = real_loss + fake_loss

        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad(set_to_none=True)
        self.untoggle_optimizer(optimizer_d)

        # ---------------------------
        # Part 1.1: Train frequency discriminator
        # ---------------------------
        self.toggle_optimizer(optimizer_df)

        t_f = torch.randint(1, self.n_steps+1, (x0_freq.shape[0],)).to(x0_freq.device)

        x_tm1_f = self.diffusion_freq.q_sample(t_f-1, x0_freq, y_freq)
        x_t_f = self.diffusion_freq.q_sample(t_f, x0_freq, y_freq)
        x_t_f.requires_grad = True

        disc_out_real_f = self.discriminator_freq(x_tm1_f, x_t_f, t_f)
        real_loss_f = self.adversarial_loss(disc_out_real_f, True)

        if self.global_step % self.disc_grad_penalty_freq == 0:
            grads_f = torch.autograd.grad(outputs=disc_out_real_f.sum(), inputs=x_t_f, create_graph=True)[0]
            grad_penalty_f = (grads_f.view(grads_f.size(0), -1).norm(2, dim=1) ** 2).mean() * self.disc_grad_penalty_weight
            real_loss_f = real_loss_f + grad_penalty_f

        # Fake samples via the frequency generator (detached)
        x0_r_f = torch.zeros_like(x_t_f)
        for _ in range(self.n_recursions):
            x0_r_f = self.generator_freq(torch.cat((x_t_f.detach(), y_freq), axis=1), t_f, x_r=x0_r_f)

        x_tm1_pred_f = self.diffusion_freq.q_posterior(t_f, x_t_f, x0_r_f, y_freq)
        disc_out_fake_f = self.discriminator_freq(x_tm1_pred_f.detach(), x_t_f, t_f)
        fake_loss_f = self.adversarial_loss(disc_out_fake_f, False)

        d_loss_f = real_loss_f + fake_loss_f

        self.manual_backward(d_loss_f)
        optimizer_df.step()
        optimizer_df.zero_grad(set_to_none=True)
        self.untoggle_optimizer(optimizer_df)

        # ---------------------------
        # Part 2: Train frequency expert
        # ---------------------------
        self.toggle_optimizer(optimizer_gf)

        t_f = torch.randint(1, self.n_steps+1, (x0_freq.shape[0],)).to(device)
        x_t_f = self.diffusion_freq.q_sample(t_f, x0_freq, y_freq)

        # Spatial guidance for the frequency expert (no gradients to the
        # spatial generator)
        with torch.no_grad():
            t_s_tmp = torch.randint(1, self.n_steps+1, (x0.shape[0],)).to(device)
            x_t_s_tmp = self.diffusion.q_sample(t_s_tmp, x0, y)
            x0_r_s_tmp = torch.zeros_like(x_t_s_tmp)
            for _ in range(self.n_recursions):
                x0_r_s_tmp = self.generator(torch.cat((x_t_s_tmp.detach(), y), dim=1), t_s_tmp, x_r=x0_r_s_tmp)
            sp_guide_freq = dct2_torch(x0_r_s_tmp)

        # Recursive refinement in the frequency domain with step-wise
        # dynamic gating against the spatial guidance. Gate input convention:
        # [own estimate, cross-domain guidance], channel 0 weights the own
        # estimate, channel 1 the guidance (softmax weights sum to 1).
        x0_r_f = torch.zeros_like(x_t_f)
        for _ in range(self.n_recursions):
            x0_r_f = self.generator_freq(torch.cat((x_t_f.detach(), y_freq), axis=1), t_f, x_r=x0_r_f)

            weights = self.gate_net(torch.cat([x0_r_f, sp_guide_freq], dim=1))
            w_f = weights[:, 0:1, ...]
            w_s = weights[:, 1:2, ...]
            x0_r_f = w_f * x0_r_f + w_s * sp_guide_freq

        x0_pred_freq = x0_r_f
        x_tm1_pred_f = self.diffusion_freq.q_posterior(t_f, x_t_f, x0_pred_freq, y_freq)
        x0_pred_freq_ifft = idct2_torch(x0_pred_freq)

        rec_loss_freq = F.l1_loss(x0_pred_freq, x0_freq, reduction='mean')
        adv_loss_f = self.adversarial_loss(self.discriminator_freq(x_tm1_pred_f, x_t_f, t_f), True)

        # Cross-domain consistency L_con (Eq. 12): gradient to the frequency
        # expert only (spatial estimate detached)
        con_loss_f = F.l1_loss(x0_pred_spatial.detach(), x0_pred_freq_ifft, reduction='mean')

        gf_loss = self.lambda_rec_loss * rec_loss_freq + adv_loss_f + self.lambda_con * con_loss_f

        self.manual_backward(gf_loss)
        optimizer_gf.step()
        optimizer_gf.zero_grad(set_to_none=True)
        self.untoggle_optimizer(optimizer_gf)

        # ---------------------------
        # Part 2.1: Train spatial expert
        # ---------------------------
        self.toggle_optimizer(optimizer_g)

        t = torch.randint(1, self.n_steps+1, (x0.shape[0],)).to(x0.device)
        x_t = self.diffusion.q_sample(t, x0, y)

        # Frequency guidance for the spatial expert (no gradients to the
        # frequency generator)
        with torch.no_grad():
            freq_guide_sp = idct2_torch(x0_pred_freq.detach())

        # Recursive refinement in the spatial domain with step-wise dynamic
        # gating against the frequency guidance
        x0_r = torch.zeros_like(x_t)
        for _ in range(self.n_recursions):
            x0_r = self.generator(torch.cat((x_t.detach(), y), axis=1), t, x_r=x0_r)

            weights = self.gate_net(torch.cat([x0_r, freq_guide_sp], dim=1))
            w_s = weights[:, 0:1, ...]
            w_f = weights[:, 1:2, ...]
            x0_r = w_s * x0_r + w_f * freq_guide_sp

        x0_pred_spatial = x0_r

        x_tm1_pred = self.diffusion.q_posterior(t, x_t, x0_pred_spatial, y)
        rec_loss_spatial = F.l1_loss(x0_pred_spatial, x0, reduction="sum")
        adv_loss = self.adversarial_loss(self.discriminator(x_tm1_pred, x_t, t), is_real=True)

        # Cross-domain consistency L_con (Eq. 12): gradient to the spatial
        # expert only (frequency estimate detached)
        con_loss_s = F.l1_loss(x0_pred_spatial, x0_pred_freq_ifft.detach(), reduction='mean')

        g_loss = self.lambda_rec_loss*rec_loss_spatial + adv_loss + self.lambda_con * con_loss_s

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad(set_to_none=True)
        self.untoggle_optimizer(optimizer_g)

        # ---------------------------
        # Part 3: Train adaptive fusion network
        # ---------------------------
        self.toggle_optimizer(optimizer_fusion)
        x0_pred_freq_ifft = x0_pred_freq_ifft.to(device)
        x0_fused = self.fusion_net(x0_pred_spatial.detach(), x0_pred_freq_ifft.detach())

        fusion_loss = F.l1_loss(x0_fused, x0, reduction='mean')

        self.manual_backward(fusion_loss)
        optimizer_fusion.step()
        optimizer_fusion.zero_grad(set_to_none=True)
        self.untoggle_optimizer(optimizer_fusion)

        # Take lr scheduler step
        scheduler_g.step()
        scheduler_d.step()
        scheduler_gf.step()
        scheduler_df.step()
        scheduler_fusion.step()

        # Log losses
        self.log("d_loss", d_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/rec", rec_loss_spatial, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/adv", adv_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/total", g_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_freq/rec", rec_loss_freq, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_freq/adv", adv_loss_f, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_freq/total", gf_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss/con", con_loss_s, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("fusion/loss", fusion_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x0, y, _ = batch
        x0 = x0.float()
        y = y.float()

        # Dual-domain sampling: each expert is coordinated with the other
        # domain's prediction through step-wise dynamic gating.
        y_freq = dct2_torch(y)
        x0_pred_f = self.diffusion_freq.sample_x0(
            y_freq, self.generator_freq, self.gate_net,
            cross_guidance=self.diffusion.sample_x0(y, self.generator))
        x0_pred_f_ifft = idct2_torch(x0_pred_f)
        x0_pred_s = self.diffusion.sample_x0(
            y, self.generator, self.gate_net, cross_guidance=x0_pred_f_ifft)

        # Fused output
        x0_fused = self.fusion_net(x0_pred_s, x0_pred_f_ifft)

        loss = F.mse_loss(x0_fused, x0)
        metrics = compute_metrics(x0, x0_fused)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_psnr", metrics["psnr_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ssim", metrics["ssim_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.global_rank == 0:
            path = os.path.join(self.logger.log_dir, "val_samples", f"epoch_{self.current_epoch}.png")
            save_image_pair(x0, x0_fused, path)

    def on_test_start(self):
        self.test_samples = []
        self.mask = None
        self.subject_ids = None

        if self.eval_mask:
            self.mask = self.trainer.datamodule.test_dataset._load_data('mask')
        if self.eval_subject:
            self.subject_ids = self.trainer.datamodule.test_dataset.subject_ids

    def test_step(self, batch, batch_idx):
        x0, y, slice_idx = batch

        # Dual-domain sampling: each expert is coordinated with the other
        # domain's prediction through step-wise dynamic gating.
        y_freq = dct2_torch(y)
        x0_pred_f = self.diffusion_freq.sample_x0(
            y_freq, self.generator_freq, self.gate_net,
            cross_guidance=self.diffusion.sample_x0(y, self.generator))
        x0_pred_f_ifft = idct2_torch(x0_pred_f)
        x0_pred_s = self.diffusion.sample_x0(
            y, self.generator, self.gate_net, cross_guidance=x0_pred_f_ifft)

        # Fused output
        x0_fused = self.fusion_net(x0_pred_s, x0_pred_f_ifft)

        # Gather fused predictions across ranks
        all_pred = self.all_gather(x0_fused)
        slice_indices = self.all_gather(slice_idx)

        if self.global_rank == 0:
            h, w = x0.shape[-2:]
            self.test_samples.extend(list(zip(
                slice_indices.flatten().tolist(),
                all_pred.reshape(-1, h, w).cpu().numpy())))

    def on_test_end(self):
        if self.global_rank == 0:
            self.test_samples.sort(key=lambda x: x[0])
            pred = np.array([x[1] for x in self.test_samples])
            slice_indices = np.array([x[0] for x in self.test_samples])
            _, locs = np.unique(slice_indices, return_index=True)
            pred = pred[locs]
            dataset = self.trainer.datamodule.test_dataset
            source = dataset.source
            target = dataset.target
            path = os.path.join(self.logger.log_dir, "test_samples", "pred.npy")
            save_preds(pred, path)
            metrics = compute_metrics(
                gt_images=target,
                pred_images=pred,
                mask=self.mask,
                subject_ids=self.subject_ids,
                report_path=os.path.join(self.logger.log_dir, "test_samples", "report.txt")
            )
            print(f"PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
            print(f"SSIM: {metrics['ssim_mean']:.2f} ± {metrics['ssim_std']:.2f}")
            indices = np.random.choice(len(dataset), 10)
            save_eval_images(
                source_images=source[indices],
                target_images=target[indices],
                pred_images=pred[indices],
                psnrs=metrics["psnrs"][indices],
                ssims=metrics["ssims"][indices],
                save_path=os.path.join(self.logger.log_dir, "test_samples")
            )

    def adversarial_loss(self, pred, is_real):
        loss = F.softplus(-pred) if is_real else F.softplus(pred)
        return loss.mean()

    def configure_optimizers(self):
        # Spatial expert
        optimizer_g = Adam(
            list(self.generator.parameters()) + list(self.gate_net.parameters()),
            lr=self.lr_g, betas=self.optim_betas
        )
        optimizer_d = Adam(self.discriminator.parameters(), lr=self.lr_d, betas=self.optim_betas)

        # Frequency expert
        optimizer_gf = Adam(
            list(self.generator_freq.parameters()) + list(self.gate_net.parameters()),
            lr=self.lr_g, betas=self.optim_betas
        )
        optimizer_df = Adam(self.discriminator_freq.parameters(), lr=self.lr_d, betas=self.optim_betas)

        # Fusion network
        optimizer_fusion = Adam(self.fusion_net.parameters(), lr=self.lr_g, betas=self.optim_betas)

        scheduler_g = CosineAnnealingLR(optimizer_g, T_max=self.trainer.max_epochs, eta_min=1e-5)
        scheduler_d = CosineAnnealingLR(optimizer_d, T_max=self.trainer.max_epochs, eta_min=1e-5)
        scheduler_gf = CosineAnnealingLR(optimizer_gf, T_max=self.trainer.max_epochs, eta_min=1e-5)
        scheduler_df = CosineAnnealingLR(optimizer_df, T_max=self.trainer.max_epochs, eta_min=1e-5)
        scheduler_fusion = CosineAnnealingLR(optimizer_fusion, T_max=self.trainer.max_epochs, eta_min=1e-5)

        # Return optimizers and schedulers in the order used by training_step
        return [optimizer_g, optimizer_d, optimizer_gf, optimizer_df, optimizer_fusion], \
            [scheduler_g, scheduler_d, scheduler_gf, scheduler_df, scheduler_fusion]


class _LightningCLI(LightningCLI):
    def instantiate_classes(self):
        if 'test' in self.parser.args and 'CSVLogger' in self.config.test.trainer.logger[0].class_path:
            exp_dir = os.path.dirname(os.path.dirname(self.config.test.ckpt_path))
            logger = self.config.test.trainer.logger[0]
            logger.init_args.save_dir = os.path.dirname(exp_dir)
            logger.init_args.name = os.path.basename(exp_dir)
            logger.init_args.version = "test"

        super().instantiate_classes()


def cli_main():
    cli = _LightningCLI(
        BridgeRunner,
        DataModule,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"}
    )


if __name__ == "__main__":
    cli_main()
