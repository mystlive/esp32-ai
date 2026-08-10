# マイクロコントローラーで2,890万パラメータのLLMを動かす

[English](README.md)

<p align="center">
  <a href="https://x.com/slvDev">𝕏 slvDev</a> &nbsp;·&nbsp;
  <a href="https://www.linkedin.com/in/slvdev/">LinkedIn</a>
</p>

![ESP32-S3で動作する2,890万パラメータのLLM](media/esp32-ple-demo.gif)

これはESP32-S3マイクロコントローラー上で文章を生成する、2,890万パラメータの言語モデルです。
サーバーへ何も送信せず、チップ単体で動作します。
チップに接続した小型画面へ、毎秒9.88トークンの速度で生成文章を表示します。
GoogleのGemma 3nで採用されたPer-Layer Embeddingsを使い、モデルの大部分をRAMではなくフラッシュへ配置することで動作します。

このforkは元のESP32-S3プロジェクトを保持し、Raspberry Pi 3 Model Bで行ったhost実行と高速化実験を追加しています。
Raspberry Pi関連の成果はfork側の追加であり、元プロジェクトの成果の帰属は変更しません。

## Raspberry Pi 3 host実験

この実験では、ESP32-S3向けTinyStoriesモデルをPortable Cランタイムによって64 bit ARM Linux host上で実行し、行列ベクトル演算に対する複数の変更を測定しました。
検証基準はupstream commit [`97018d5`](https://github.com/slvDev/esp32-ai/commit/97018d5)です。
検証済み条件を維持するため、upstreamの後続変更は意図的に取り込んでいません。

### 検証環境

| 項目 | 値 |
| ---- | -- |
| ボード | Raspberry Pi 3 Model B |
| SoC | Broadcom BCM2837 |
| CPU | ARM Cortex-A53、4コア、1.2 GHz |
| メモリ | 1 GB RAM |
| OS | Debian GNU/Linux 12（bookworm） |
| アーキテクチャ | aarch64 |
| コンパイラ | GCC 12.2.0 |

### 結果

| 構成 | Throughput | Forward |
| ---- | ---------- | ------- |
| Portable int4 | 31.22 tok/s | 31.89 ms/token |
| int8 staging | 171.81 tok/s | 5.68 ms/token |
| int8 + OpenMP 4-core | 146.32 tok/s | 6.54 ms/token |
| int8 + NEON | 177.98 tok/s | 5.47 ms/token |
| int8 + NEON 32B / 2 accumulators | 195.81 tok/s | 4.96 ms/token |

Portable int4版は31.22 tok/sで、host実行の基準値となりました。
推論前に4 bit重みをint8 staging表現へ展開する変更が最大の改善を生み、171.81 tok/sへ到達しました。
OpenMPで4スレッドを使用すると、throughputは146.32 tok/sへ低下しました。
同期コスト、キャッシュ動作、メモリ帯域が原因候補ですが、この実験では単一原因を特定していません。

ARM NEON SIMDの使用により、throughputは177.98 tok/sへ向上しました。
1ループあたり32 byteを処理し、独立したaccumulatorを2本使用することで、最終結果は195.81 tok/sへ向上しました。
これはhost上の初期基準値に対して約6.27倍です。
再起動後に再コンパイルした最終版は195.55 tok/s、4.97 ms/tokenとなり、ほぼ同じ性能を再現しました。

profile結果では、処理時間の約59.1%がoutput head、27.9%がattention、5.7%がPLE、4.8%がFFN、2.4%がinput processingでした。
output headとattentionを合わせると、測定時間の約87%を占めました。

これらの数値が比較しているのは、このモデルと各推論実装によるtoken生成速度です。
Raspberry Pi 3とESP32-S3のCPU性能倍率ではありません。
今回のTinyStories推論条件では、Raspberry Pi最終版は公開されているESP32-S3の結果より大幅に高いthroughputとなりました。

### 実験用ソース

| ファイル | 役割 |
| -------- | ---- |
| `runtime/host_generate/tinystories_generate.c` | int8 stagingとOpenMP 4スレッドoutput head版 |
| `runtime/host_generate/tinystories_generate_i8_single.c` | 単一コアint8 staging基準版 |
| `runtime/host_generate/tinystories_generate_neon.c` | 16 byte単位のARM NEON内積版 |
| `runtime/host_generate/tinystories_generate_neon2.c` | 32 byte単位と2 accumulatorを使用する最終ARM NEON版 |
| `runtime/host_generate/tinystories_generate_profile.c` | 16 byte NEON版に`LLM_PROFILE`計測を追加した版 |

このディレクトリにはPortable int4専用のソースはありません。
`tinystories_generate.c`はPortable int4基準版ではなく、OpenMP int8実験版です。

### 最終版のビルドと実行

モデルと生成済みvocabulary headerは、意図的にGit管理対象から除外しています。
リポジトリのルートで固定済みTinyStories artifactを取得し、コンパイル前にvocabulary headerを生成します。

```bash
scripts/fetch_model.sh tinystories

uv run python firmware/esp32_tinystories/tools/generate_vocab.py \
  --tokenizer artifacts/tinystories/tokenizer.json \
  --out firmware/esp32_tinystories/generated/vocab.h

gcc -std=c11 -O3 -Wall -Wextra -march=armv8-a+simd \
  -o tinystories_generate_neon2 \
  runtime/host_generate/tinystories_generate_neon2.c -lm

./tinystories_generate_neon2 artifacts/tinystories/model.bin
```

実験当初は実行ファイルを`/tmp`へ配置していました。
再起動によって一時実行ファイルは消えましたが、ソース、モデル、仮想環境は残っていました。
継続利用する場合は、通常の作業ディレクトリへコンパイルしてください。

### モデルの用途

TinyStoriesは英語の短い物語を生成します。
一般的な質問回答モデル、instruction model、ChatGPTの代替、プログラミング用モデル、一般知識の回答元ではありません。
この実験は、小規模LLMと軽量CランタイムをARM CPU向けにどこまで高速化できるかを評価しています。
一般的なモデルの知能は評価対象ではありません。

元プロジェクト：[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)

forkおよびRaspberry Pi 3実験：[mystlive/esp32-ai](https://github.com/mystlive/esp32-ai)

## 数値

| 項目 | 値 |
| ---- | -- |
| パラメータ | 2,890万格納、そのうち2,500万はフラッシュ上のlookup table |
| チップ | ESP32-S3、512 KB SRAM、8 MB PSRAM、16 MBフラッシュ |
| 速度 | end-to-endで9.88 tok/s、計算時間94.9 ms/token |
| 接続 | なし、すべてデバイス上で実行 |
| モデルサイズ | 4 bitで14.9 MB |

## 難しい理由と、それでも格納できる仕組み

マイクロコントローラーで使用できる高速メモリはわずかです。
ESP32-S3のSRAMは512 KBであり、各tokenで何度も参照されるactivationとnorm weightだけを配置できます。
各positionで1回走査されるdense coreとoutput headはPSRAMへ配置します。
残るembedding tableのサイズが、通常はモデルの大きさを決定します。

このモデルでは、パラメータの大部分が計算対象ではなく参照対象のembedding tableに格納されています。
2,500万パラメータのtableは低速なフラッシュに留まり、各tokenに必要な数行、約450 byteだけを読み出します。
モデルの大部分は実行時にもロードされず、フラッシュに置いたまま少しずつ参照されます。

この方式は、Googleの[Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n)で採用されたPer-Layer Embeddingsです。
ここではスマートフォンやGPU向けのメモリ配置ではなく、マイクロコントローラーのメモリ配置上で動作します。

各階層には、参照頻度に応じたデータを配置します。

```text
  SRAM  （高速、小容量） activationとnorm weight、各tokenで複数回参照
  PSRAM （中速）         coreとoutput head、各positionで1回参照
  FLASH （低速、大容量） 2,500万パラメータのtable、各tokenで約6行を参照（約450 B）
```

## できることと、できないこと

このモデルはTinyStoriesで学習されているため、短く単純な物語を生成し、ある程度の一貫性を保ちます。
質問への回答、指示への追従、コード作成、事実知識の回答はできません。
この制約は推論を担う部分の小ささに由来し、メモリ配置の工夫では変わりません。
ここでの対象は2,890万パラメータのモデルが何を話せるかではなく、大きなモデルを小さなチップへ格納するアーキテクチャです。

## モデル

- [Barista](https://huggingface.co/slvDev/esp32-ai-barista)：エスプレッソに関する質問回答
- [TinyStories](https://huggingface.co/slvDev/esp32-ai-tinystories)：物語生成

## 自分で実行する

ダウンロードとデプロイは別の処理です。
前者はネットワークへ接続し、後者はボードへ書き込みます。

```bash
scripts/fetch_model.sh barista   # ダウンロード、検証、artifacts/への配置
scripts/deploy.sh barista        # header生成、gate実行、コンパイル、書き込み
```

もう一つのモデルは`tinystories`であり、同じ2つのコマンドを使用します。
ボードへ配置できるモデルは一度に1つであり、デプロイすると置き換わるため、どちらのコマンドにもモデル名が必要です。

`fetch_model.sh`は推論用artifactを、スクリプトに固定されたSHA-256とbyte sizeに対して検証します。
同じ固定値を使い、release内の`metadata.json`も照合します。
すべての検証に成功するまで何も配置しないため、ダウンロードに失敗しても既存ファイルは維持されます。
`deploy.sh`はモデルをダウンロードせず、`artifacts/<model>/`に存在するファイルを使用します。
2つのheader生成ツールは`uv`経由で実行されるため、wheelが未cacheの環境では、初回のみ固定されたwheelを取得します。

firmwareの詳細と想定されるboot outputは、[`firmware/esp32_barista/README.md`](firmware/esp32_barista/README.md)および[`firmware/esp32_tinystories/README.md`](firmware/esp32_tinystories/README.md)にあります。
再利用可能なアーキテクチャは`src/`にあります。
公開数値を再現する学習、ablation、quantizationコードは`research/tinystories/`にあります。
手法全体、ablation、チップ上の測定結果は[`RESULTS.md`](RESULTS.md)に記載されています。

## クレジット

TinyStoriesは、小規模モデルでも一貫した文章を学習できる程度に単純化された短い合成物語のデータセットです。
著者はMicrosoft ResearchのRonen Eldan氏とYuanzhi Li氏です。
論文は[arXiv:2305.07759](https://arxiv.org/abs/2305.07759)です。
もう一つの要素は、GoogleがGemma 3n向けに設計したPer-Layer Embeddingsであり、大きなモデルを小さなチップへ格納する仕組みです。

Andrej Karpathy氏の[llama2.c](https://github.com/karpathy/llama2.c)は、小規模言語モデルの学習とPlain Cによる実行の参考実装です。

## 測定結果

詳細な測定結果とablationは`RESULTS.md`に記載されています。
