#!/usr/bin/env python

"""
conda create -n ppo python=3.7 numpy ipython matplotlib swig termcolor tqdm scipy tensorboard
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install gym[box2d]
pip install plotille
"""

import logging
import threading

import gym
import numpy as np
import toolbox.config_loader as config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from IPython import embed
from toolbox.log_writer import LogWriter

from .pop_art import PopArtLayer

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, n_features, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, 32), nn.LayerNorm(32), nn.ReLU(),
        )
        self.policy = nn.Linear(32, n_actions)
        # self.vf = nn.Linear(32, 1)
        self.vf = PopArtLayer(32, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, all_v=False):
        x = self.shared(x)
        v = self.vf(x)
        logit = self.policy(x)
        return v if all_v else v[1], F.log_softmax(logit, -1), F.softmax(logit, -1)


if __name__ == "__main__":

    args = config.get()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    writer = LogWriter(args.logdir)

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"use {device}")

    # 환경 생성
    env = gym.vector.make("LunarLander-v2", num_envs=args.n_envs)
    env.seed(args.seed)
    n_features = env.observation_space.shape[1]
    n_actions = env.action_space[0].n

    # 모델 & 옵티마이저 생성
    model = Model(n_features, n_actions)
    model_old = Model(n_features, n_actions)
    model_old.load_state_dict(model.state_dict())
    eta = torch.tensor(1.0, requires_grad=True)
    alpha = torch.tensor(5.0, requires_grad=True)
    optimizer = optim.Adam(
        [*model.parameters(), eta, alpha], lr=args.lr, weight_decay=args.weight_decay
    )
    # 테스트 게임 시작
    def run_test_game():
        test_env = gym.make("LunarLander-v2")
        test_env.seed(args.seed)

        while True:
            test_state = test_env.reset()
            score = 0
            while True:
                with torch.no_grad():
                    _, _, prob = model_old(
                        torch.from_numpy(test_state).float().view(1, -1)
                    )
                action = prob.multinomial(1).item()
                test_state, r, done, _ = test_env.step(action)
                score += r
                test_env.render()
                if done:
                    score = 0
                    break

    if args.eval:
        threading.Thread(target=run_test_game, daemon=True).start()

    # 버퍼 생성
    D_obs = np.zeros((args.horizon, args.n_envs, n_features))
    D_action = np.zeros((args.horizon, args.n_envs), dtype=np.int64)
    D_reward = np.zeros((args.horizon, args.n_envs))
    D_done = np.zeros((args.horizon, args.n_envs), dtype=np.bool_)
    D_value = np.zeros((args.horizon, args.n_envs))
    D_norm_value = np.zeros((args.horizon, args.n_envs))
    D_logp_a = np.zeros((args.horizon, args.n_envs))
    D_task = np.zeros((args.horizon, args.n_envs, args.n_tasks))

    # 학습 시작
    frames = 0
    score = np.zeros(args.n_envs)
    obs_prime = env.reset()

    while True:
        obs = obs_prime
        for D_i in tqdm.trange(args.horizon, desc="Rollout"):
            # 게임 플레이 & 데이터 수집
            with torch.no_grad():
                (value, norm_value), logp, prob = model_old(
                    torch.tensor(obs).float(), all_v=True
                )
                action = prob.multinomial(num_samples=1).numpy().reshape(-1)
            obs_prime, reward, done, info = env.step(action)

            # 점수 기록
            score += reward
            writer.add_stat("score/score", score[done])
            score[done] = 0

            # 데이터 저장
            D_obs[D_i] = obs
            D_action[D_i] = action
            D_reward[D_i] = reward / 100.0
            D_done[D_i] = done
            D_value[D_i] = value.view(-1).numpy()
            D_norm_value[D_i] = norm_value.view(-1).numpy()
            D_logp_a[D_i] = logp.numpy()[range(args.n_envs), action]
            task = np.array([0] * args.n_envs)
            D_task[D_i, np.arange(args.n_envs), task] = 1.0  # task!!

            obs = obs_prime
            frames += args.n_envs

        # 데이터 수집 완료

        # popart 파라미터 준비
        mu = model.vf.mu[None, :].cpu().numpy()
        sigma = model.vf.sigma[None, :].cpu().numpy()
        # gamma
        gamma = args.gamma * (1 - D_done)
        # return 계산
        D_ret = np.zeros((args.horizon + 1, args.n_envs))
        with torch.no_grad():
            (value, norm_value), _, _ = model_old(
                torch.tensor(D_obs[-1]).float(), all_v=True
            )
        D_ret[-1] = value.view(-1).numpy()
        for t in reversed(range(args.horizon)):
            D_ret[t] = D_reward[t] + gamma[t] * D_ret[t + 1]
        D_ret = D_ret[:-1]
        writer.add_stat("score/ret", D_ret.mean())
        D_ret = (D_ret - mu) / sigma  # popart value

        # adv 계산
        value_ = np.vstack([D_value, value.numpy().transpose(1, 0)])
        norm_value_ = np.vstack([D_norm_value, norm_value.numpy().transpose(1, 0)])
        # delta = D_reward + gamma * value_[1:] - value_[:-1]
        # popart pg
        delta = ((D_reward + gamma * value_[1:]) - mu) / sigma - norm_value_[:-1]
        D_adv = np.zeros((args.horizon, args.n_envs))
        gae = 0
        for t in reversed(range(args.horizon)):
            gae = gae * gamma[t] * args.lam + delta[t]
            D_adv[t] = gae

        # batch 차원 제거
        FD_obs = D_obs.reshape(-1, n_features)
        FD_action = D_action.reshape(-1)
        FD_logp_a = D_logp_a.reshape(-1)
        FD_ret = D_ret.reshape(-1)
        FD_adv = D_adv.reshape(-1)
        # top_k (상위 50% advantage 위치)
        FD_top_k = FD_adv > np.median(FD_adv)
        # adv 정규화
        adv_mean, adv_std = FD_adv.mean(), FD_adv.std()
        writer.add_stat("info/adv_mean", adv_mean)
        writer.add_stat("info/adv_std", adv_std)
        # FD_task = D_task.reshape(-1)

        # 미니배치 index 준비
        idx = np.arange(args.horizon * args.n_envs)
        np.random.shuffle(idx)
        n_mini_batchs = args.horizon * args.n_envs // args.mini_batch_size
        for mb_i in tqdm.trange(n_mini_batchs, desc="Fit"):
            # 미니배치 준비
            sel = idx[mb_i * args.mini_batch_size : (mb_i + 1) * args.mini_batch_size]
            obs = torch.tensor(FD_obs[sel], device=device).float()
            action = torch.tensor(FD_action[sel], device=device).long()
            ret = torch.tensor(FD_ret[sel], device=device).float()
            adv = torch.tensor(FD_adv[sel], device=device).float()
            logp_a_old = torch.tensor(FD_logp_a[sel], device=device).float()
            top_k = torch.tensor(FD_top_k[sel], device=device).bool()

            # 그래프 생성
            value, logp, prob = model(obs)
            writer.add_stat("score/value", value.mean())
            logp_a = logp.gather(1, action.view(-1, 1)).view(-1)
            entropy = -(prob * logp).sum(-1).mean()
            writer.add_stat("loss/ent", entropy.item())

            # loss_v
            loss_v = F.mse_loss(value, ret.view(value.shape))  # * task
            writer.add_stat("loss/value", loss_v.item())
            # loss_pi
            with torch.no_grad():
                aug_adv_max = (adv[top_k] / eta).max()
                aug_adv = (adv[top_k] / eta - aug_adv_max).exp()
                norm_aug_adv = aug_adv / aug_adv.sum()
            loss_pi = -(norm_aug_adv * logp_a[top_k]).sum()
            writer.add_stat("loss/pi", loss_pi.item())
            # loss_eta (dual func.)
            loss_eta = (
                eta * args.eps_eta
                + aug_adv_max
                + eta * (adv[top_k] / eta - aug_adv_max).exp().mean().log()
            )
            writer.add_stat("loss/eta", loss_eta.item())
            # loss_alpha
            prob_a_old, prob_a = logp_a_old.exp(), logp_a.exp()
            kld = prob_a_old * (logp_a_old - logp_a) - prob_a_old + prob_a
            writer.add_stat("info/kld", kld.mean().item())
            loss_alpha = (
                alpha * (args.eps_alpha - kld.detach()) + alpha.detach() * kld
            ).mean()
            writer.add_stat("loss/alpha", loss_alpha.item())
            # total_loss
            loss = loss_v * args.value_coef + loss_pi + loss_eta + loss_alpha

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
            optimizer.step()

            with torch.no_grad():
                # eta와 alpha는 반드시 0 이상
                eta.data.copy_(eta.clamp(min=1e-6, max=1e6))
                alpha.data.copy_(alpha.clamp(min=1e-6, max=1e6))

        # popart
        model.vf.update_parameters(
            torch.tensor(D_ret[:, :, None], dtype=torch.float32, device=device),
            torch.tensor(D_task, dtype=torch.float32, device=device),
        )

        # target 모델 교체
        model_old.load_state_dict(model.state_dict())

        # 학습결과 출력
        if writer.checkpoint(frames):
            epoch, mean_score = writer.epoch, writer.mean_score
            writer.save_model(f"model-{epoch}-{mean_score}", model)

        if writer.esc_pressed():
            embed()

        if args.max_frames > 0 and frames > args.max_frames:
            writer.add_hparams(dict(algo="ppg"), dict(final_score=writer.mean_score()))
            break
