#!/usr/bin/env python

"""
conda create -n ppo python=3.7 numpy ipython matplotlib swig termcolor tqdm scipy tensorboard
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install gym[box2d]
pip install plotille
"""

import threading

import gym
import numpy as np
import toolbox.config_loader as config
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.optim as optim
import tqdm
from IPython import embed
from toolbox.log_writer import LogWriter

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, n_features, n_actions):
        super().__init__()
        self.vf = nn.Sequential(
            nn.Linear(n_features, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, 1),
        )
        self.shared = nn.Sequential(
            nn.Linear(n_features, 32), nn.LayerNorm(32), nn.ReLU(),
        )
        self.policy = nn.Linear(32, n_actions)
        self.aux = nn.Linear(32, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        v = self.vf(x)
        logit = self.policy(self.shared(x))
        return v, F.log_softmax(logit, -1), F.softmax(logit, -1)

    def aux_forward(self, x):
        v = self.vf(x)
        x = self.shared(x)
        aux_v = self.aux(x)
        logit = self.policy(x)
        return v, aux_v, F.log_softmax(logit, -1), F.softmax(logit, -1)


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
    policy_optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    aux_optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
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
    D_logp = np.zeros((args.horizon, args.n_envs, n_actions))

    # 학습 시작
    frames = 0
    score = np.zeros(args.n_envs)
    obs_prime = env.reset()

    while True:
        obs = obs_prime
        for D_i in tqdm.trange(args.horizon, desc="Rollout"):
            # 게임 플레이 & 데이터 수집
            with torch.no_grad():
                value, logp, prob = model_old(torch.from_numpy(obs).float())
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
            D_logp[D_i] = logp.numpy()

            obs = obs_prime
            frames += args.n_envs

        # 데이터 수집 완료
        D_i = 0
        # gamma
        gamma = args.gamma * (1 - D_done)
        # return 계산
        D_ret = np.zeros((args.horizon + 1, args.n_envs))
        with torch.no_grad():
            value, _, _ = model_old(torch.from_numpy(D_obs[-1]).float())
        D_ret[-1] = value.view(-1).numpy()
        for t in reversed(range(args.horizon)):
            D_ret[t] = D_reward[t] + gamma[t] * D_ret[t + 1]
        D_ret = D_ret[:-1]
        writer.add_stat("score/ret", D_ret.mean())
        # adv 계산
        value_ = np.vstack([D_value, value.numpy().transpose(1, 0)])
        delta = D_reward + gamma * value_[1:] - value_[:-1]
        D_adv = np.zeros((args.horizon, args.n_envs))
        gae = 0
        for t in reversed(range(args.horizon)):
            gae = gae * gamma[t] * args.lam + delta[t]
            D_adv[t] = gae

        # batch 차원 제거
        FD_obs = D_obs.reshape(-1, n_features)
        FD_action = D_action.reshape(-1)
        FD_logp = D_logp.reshape(-1, n_actions)
        FD_ret = D_ret.reshape(-1)
        FD_adv = D_adv.reshape(-1)
        # adv 정규화
        adv_mean, adv_std = FD_adv.mean(), FD_adv.std()
        FD_adv = (FD_adv - adv_mean) / (adv_std + 1e-8)
        writer.add_stat("info/adv_mean", adv_mean)
        writer.add_stat("info/adv_std", adv_std)

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
            logp_old = torch.tensor(FD_logp[sel], device=device).float()

            # 그래프 생성
            value, logp, prob = model(obs)
            writer.add_stat("score/value", value.mean())
            logp_a = logp.gather(1, action.view(-1, 1)).view(-1)
            logp_old_a = logp_old.gather(1, action.view(-1, 1)).view(-1)

            # loss_v
            loss_v = F.mse_loss(value, ret.view(value.shape))
            writer.add_stat("loss/value", loss_v.item())
            # loss_pi
            ratios = torch.exp(logp_a - logp_old_a)
            policy_loss1 = -adv * ratios
            cr = args.clip_range
            policy_loss2 = -adv * torch.clamp(ratios, min=1.0 - cr, max=1.0 + cr)
            frac_ratio = ((ratios < (1.0 - cr)) | (ratios > (1.0 + cr))).float().mean()
            writer.add_stat("info/frac_ratio", frac_ratio)
            loss_pi = torch.mean(torch.max(policy_loss1, policy_loss2))
            writer.add_stat("loss/pi", loss_pi.item())
            # entropy
            entropy = -(prob * logp).sum(-1).mean()
            writer.add_stat("loss/ent", entropy)
            loss = loss_v * args.value_coef + loss_pi - args.ent_coef * entropy

            policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
            policy_optimizer.step()

        # target 모델 교체
        model_old.load_state_dict(model.state_dict())

        # Aux. phase
        for mb_i in tqdm.trange(args.aux_epochs * n_mini_batchs, desc="Aux. fit"):
            # 미니배치 준비
            mb_j = mb_i % n_mini_batchs
            sel = idx[mb_j * args.mini_batch_size : (mb_j + 1) * args.mini_batch_size]
            obs = torch.tensor(FD_obs[sel], device=device).float()
            ret = torch.tensor(FD_ret[sel], device=device).float()
            with torch.no_grad():
                _, logp_old, prob_old = model_old(obs)

            # 그래프 생성
            value, aux_value, logp, prob = model.aux_forward(obs)

            # loss_v
            loss_v = F.mse_loss(value, ret.view(value.shape))
            # loss_aux_v
            loss_aux_v = F.mse_loss(aux_value, ret.view(value.shape))
            writer.add_stat("aux_loss/aux_value", loss_aux_v.item())
            # loss_kld
            kld = (prob_old * (logp_old - logp) - prob_old + prob).sum(1).mean()
            writer.add_stat("aux_loss/kld", kld.item())
            loss = loss_v + loss_aux_v + args.clone_coef * kld

            aux_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
            aux_optimizer.step()

        # target 모델 교체
        model_old.load_state_dict(model.state_dict())

        # 학습결과 출력

        if writer.checkpoint(frames):
            writer.save_model(f"model-{writer.mean_score()}", model)

        if writer.esc_pressed():
            embed()

        if args.max_frames > 0 and frames > args.max_frames:
            writer.add_hparams(dict(algo="ppg"), dict(final_score=writer.mean_score()))
            break
