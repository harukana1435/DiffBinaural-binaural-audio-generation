# DiffBinaural: バイノーラル音声生成

モノラル音声と映像情報からバイノーラル音声を生成する拡散モデルの実装です。

## 概要

DiffBinauralは、バイノーラル音声を生成するための2段階アプローチです：

1. **Stage 1 (DiffBinaural)**: モノラル音声と映像特徴からバイノーラルメルスペクトログラムを生成する拡散モデル
2. **Stage 2 (BigVGAN)**: メルスペクトログラムから高品質なバイノーラル音声波形を生成するニューラルボコーダー

## プロジェクト構造

```
DiffBinaural: binaural audio generation/
├── DiffBinaural/              # Stage 1: バイノーラルメル生成用の拡散モデル
│   ├── modules/               # ネットワークアーキテクチャ（UNet、視覚エンコーダー）
│   ├── dataset/               # FairPlayとRealBinauralのデータセットローダー
│   ├── utils/                 # ヘルパー関数と引数パーサー
│   ├── diffusion_utils/       # 拡散モデルの実装
│   ├── train_fairplay.py      # FairPlayデータセット用学習スクリプト
│   ├── train_realBinaural.py  # RealBinauralデータセット用学習スクリプト
│   ├── test_fairplay.py       # FairPlayデータセット用テストスクリプト
│   └── test_realBinaural.py   # RealBinauralデータセット用テストスクリプト
│
├── BigVGAN/                   # Stage 2: ニューラルボコーダー
│   ├── train_binaural_mel.py  # 事前計算されたメルでの学習（Stage 2）
│   ├── train_binaural_both.py # スケジュールドサンプリングでの学習
│   ├── bigvgan.py             # BigVGANジェネレーターアーキテクチャ
│   ├── discriminators.py      # マルチスケール識別器
│   └── configs/               # BigVGAN設定ファイル
│
└── README.md                  # メインドキュメント（英語）
```

## 必要な環境

### Pythonパッケージ

```bash
pip install -r requirements.txt
```

主な依存関係：
- PyTorch >= 1.13.0
- librosa >= 0.8.1
- numpy, scipy
- tensorboard
- soundfile
- pesq, auraloss

### ハードウェア要件

- 最低16GB VRAM搭載のGPU（学習には24GB以上推奨）
- マルチGPU対応

## データセットの準備

### RealBinauralデータセット

```
real_dataset/
├── processed/
│   ├── mono_audios_22050Hz/      # モノラル音声ファイル
│   ├── binaural_audios_22050Hz/  # バイノーラル音声ファイル
│   └── frames/                    # ビデオフレーム
├── action_detection_results/
│   └── detection_results.csv      # アクション検出メタデータ
└── splits/
    ├── train.csv                  # 学習用分割
    └── val.csv                    # 検証用分割
```

## 学習方法

### Stage 1: DiffBinauralの学習

```bash
cd DiffBinaural

python train_realBinaural.py \
    --id realBinaural_experiment \
    --list_train /path/to/real_dataset/splits/train.csv \
    --list_val /path/to/real_dataset/splits/val.csv \
    --data_root /path/to/real_dataset \
    --ckpt ./checkpoints \
    --num_gpus 2 \
    --batch_size_per_gpu 8 \
    --num_epoch 1000 \
    --lr_unet 0.0001 \
    --lr_frame 0.0001 \
    --eval_epoch 10
```

### Stage 2: BigVGANの学習

#### Stage 2a: 事前計算されたメルでの学習（推奨）

まず、学習済みDiffBinauralモデルを使用してメルスペクトログラムを生成：

```bash
cd DiffBinaural

python test_realBinaural.py \
    --id realBinaural_experiment \
    --mode test \
    --weights_unet ./checkpoints/unet_best.pth \
    --weights_frame ./checkpoints/frame_best.pth \
    --output_dir ./generated_mels
```

次に、生成されたメルでBigVGANを学習：

```bash
cd BigVGAN

python train_binaural_mel.py \
    --mel_left_train_dir /path/to/generated_mels/left/train \
    --mel_right_train_dir /path/to/generated_mels/right/train \
    --audio_dir /path/to/real_dataset/processed/binaural_audios_22050Hz \
    --checkpoint_path ./checkpoints/bigvgan_stage2 \
    --config ./configs/bigvgan_binaural_22khz_80band_256x.json \
    --training_epochs 2000
```

#### Stage 2b: スケジュールドサンプリングでの学習（高度）

ロバスト性向上のため、正解メルから予測メルへ徐々に移行するスケジュールドサンプリングを使用：

```bash
cd BigVGAN

python train_binaural_both.py \
    --mel_left_train_dir /path/to/generated_mels/left/train \
    --mel_right_train_dir /path/to/generated_mels/right/train \
    --mel_pred_left_dir /path/to/predicted_mels/left/train \
    --mel_pred_right_dir /path/to/predicted_mels/right/train \
    --audio_dir /path/to/real_dataset/processed/binaural_audios_22050Hz \
    --checkpoint_path ./checkpoints/bigvgan_scheduled \
    --config ./configs/bigvgan_binaural_22khz_80band_256x.json \
    --training_epochs 2000 \
    --use_scheduled_sampling
```

**スケジュールドサンプリング戦略：**
- 100%正解メルから開始
- 徐々に予測メルを使用する確率を増加
- DiffBinauralからの不完全な入力に対してモデルをロバストにする
- 学習・テストのミスマッチ問題を軽減

## 主な特徴

### DiffBinaural (Stage 1)

- **拡散ベースの生成**: 効率的な推論のためDDIMサンプリングを使用
- **視覚条件付け**: ビデオフレームからの視覚特徴を組み込み
- **位置エンコーディング**: 空間音響のための2D位置エンコーディングをサポート
- **学習安定化**: 勾配クリッピングと損失監視を含む
- **マルチデータセット対応**: FairPlayとRealBinauralの両データセットで動作

### BigVGAN (Stage 2)

- **高品質ボコーダー**: 22kHzバイノーラル音声を生成
- **スケジュールドサンプリング**: ロバスト性のための新しい学習戦略
- **静寂認識損失**: 静寂領域でのアーティファクトを軽減
- **マルチスケール識別器**: 高品質な音声生成を保証
- **エイリアスフリー活性化**: エイリアシングアーティファクトを軽減

## 学習のコツ

1. **Stage 1 (DiffBinaural)**:
   - 安定性のため小さめの学習率（1e-4）から開始
   - 勾配爆発を防ぐため勾配クリッピングを使用
   - 検証時にメルスペクトログラムのL2距離を監視
   - 良好な収束のため最低500エポック学習

2. **Stage 2 (BigVGAN)**:
   - まず正解メルで事前学習（train_binaural_mel.py）
   - 次にスケジュールドサンプリングでファインチューニング（train_binaural_both.py）
   - 静かな領域のノイズを減らすため静寂認識損失を使用
   - メルスペクトログラムエラーと識別器損失の両方を監視

3. **マルチGPU学習**:
   - `--num_gpus`でGPU数を指定
   - GPUメモリに応じて`--batch_size_per_gpu`を調整
   - 総バッチサイズ = num_gpus × batch_size_per_gpu

## ドキュメント

- **README.md**: 包括的なプロジェクトドキュメント（英語）
- **QUICKSTART.md**: クイックスタートガイド（英語）
- **DiffBinaural/README.md**: Stage 1の詳細ドキュメント（英語）
- **BigVGAN/README.md**: Stage 2の詳細ドキュメント（英語）
- **PROJECT_STRUCTURE.md**: プロジェクト構造の詳細（英語）

## ライセンス

このプロジェクトには以下のコードが含まれています：
- BigVGAN: [NVIDIA BigVGAN](https://github.com/NVIDIA/BigVGAN) (MITライセンス)

詳細は各LICENSEファイルを参照してください。

## 謝辞

- NVIDIAのBigVGAN実装
- Denoising Diffusion Probabilistic Models (DDPM)にインスパイアされた拡散モデル実装
- バイノーラル音響研究のためのFairPlayデータセット

## お問い合わせ

質問や問題がある場合は、GitHubでissueを開いてください。


