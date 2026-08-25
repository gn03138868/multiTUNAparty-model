<div align="center">

<img src="docs/icon_main.png" width="120" alt="MultiTUNAparty">

# MultiTUNAparty

**Segment and measure cells, roots and fungal structures — without writing any code.**

**細胞、根、真菌構造的影像辨識與量化 —— 完全不用會寫程式。**

[![Windows](https://img.shields.io/badge/Windows-one--click%20installer-0078D6)](#install-on-windows--一鍵安裝)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA%2012.x%20%7C%20CPU-EE4C2C)](https://pytorch.org/)
![Licence](https://img.shields.io/badge/Licence-MIT-green)

</div>

---

## What this is / 這是什麼

A small lab has images and a question: *how many, how big, what shape?* The usual
answer is either hours of manual tracing in ImageJ, or a deep-learning pipeline
that needs someone who can write Python.

MultiTUNAparty is the third option. It is one trained model that already knows
three kinds of biological target, wrapped in two windows you open by
double-clicking an icon:

| | |
|---|---|
| <img src="docs/icon_main.png" width="72"> | **MultiTUNAparty** — load an image, pick the target type, press a button. Get a mask, and a table of per-object area, diameter, perimeter and circularity you can save as CSV. |
| <img src="docs/icon_training.png" width="72"> | **MultiTUNAparty Training** — point it at a folder of your own images and masks, press Start. It continues from the shipped model, so a few dozen annotated images are enough to teach it a new target. |

There is no command line anywhere in the normal path. There is no environment
to activate, no config file to edit, no checkpoint to locate — the installer
sets up its own private Python, and the program loads the model by itself.

**Three target types it ships knowing:**

| Slot | Name | Trained on |
|---|---|---|
| 0 | Cell (Plant Cell) | plant parenchyma, e.g. potato tuber sections |
| 1 | Blood (Blood Cell) | round, well-separated objects such as blood cells |
| 2 | Other (Other System) | roots, fungal trapping rings, hyphae, and anything you train it on |

---

## Install on Windows / 一鍵安裝

1. Download **`MultiTUNAparty_v1.2-Windows-Installer.zip`** from the
   [Releases page](../../releases/latest).
2. Right-click the zip → **Extract All**. Extract the *whole* folder; do not
   drag one file out of it.
3. Double-click **`INSTALL.bat`**.

下載 zip → 整個資料夾解壓縮 → 點兩下 `INSTALL.bat`。不用打任何指令，
不需要系統管理員密碼。

That is the entire procedure. The installer then does this by itself:

| | |
|---|---|
| 1 | Checks free disk space |
| 2 | Downloads and installs Python 3.11 for your user only, if the computer has none |
| 3 | Builds a private environment that cannot disturb any other software |
| 4 | Detects your graphics card and picks the matching PyTorch build (NVIDIA → CUDA, otherwise CPU) |
| 5 | Installs the packages, one at a time, printing pip's own error text if one fails |
| 6 | Installs the application |
| 7 | Finds or downloads the trained model |
| 8 | Creates the Desktop icons, makes your data folder, and runs a self-test |

**15–30 minutes, about 8 GB of disk, and an internet connection.** Windows may
show a blue *"Windows protected your PC"* box — click **More info** → **Run
anyway**. It appears because the file is not code-signed, not because anything
is wrong with it.

### Already installed? / 已經裝好了？

Run **`UPDATE.bat`** instead. It refreshes the program files, the icons and the
shortcuts in a few seconds, and leaves Python, PyTorch and the model alone.

### Offline machines / 離線散布

Put `best_model.pth` beside `INSTALL.bat` before zipping the folder. Whoever
receives it double-clicks `INSTALL.bat` and needs no network for the model step.

### Building a `.exe` wizard (optional)

`MultiTUNAparty.iss` builds a conventional Next/Next/Finish installer with
[Inno Setup 6](https://jrsoftware.org/isdl.php) — run `build_installer_exe.bat`,
or open the `.iss` and press F9. It needs no administrator rights either.

---

## Using it / 怎麼用

Double-click **MultiTUNAparty**. The model loads by itself. Five tabs:

| Tab | What it is for |
|---|---|
| 📦 Model Management | Which checkpoint, CPU or GPU. Already set; you can usually ignore it. |
| 🔍 Single Prediction | One image at a time. Pick the target type, press Start Prediction. |
| 📊 Morphological Analysis | Per-object area, diameter, perimeter, circularity, centroid → CSV. |
| 📁 Batch Processing | A whole folder in one go. |
| 📖 Help | Built-in notes. |

**The one mistake everyone makes** is choosing the wrong target type. A plant
cell image run as *Blood* gives a poor or empty mask. 選錯標的類型會得到很差
或全黑的結果——這是最常見的問題。

### If the result looks wrong

| Symptom | Try |
|---|---|
| Everything black | Lower Threshold to ~0.3, and check the target type |
| Too noisy | Raise Threshold to 0.6–0.7 |
| Objects merged together | Raise Threshold; set a minimum area in Morphological Analysis |
| Very slow | Check that GPU is selected on Model Management |

Large images are handled by sliding-window inference: overlapping tiles blended
with a Gaussian weight (σ = tile ⁄ 4, offset so a tile's edge never reaches zero)
so no seams appear in the output. Overlap is adjustable; 0 is fastest and may
show tile edges.

---

## Train it on your own images / 訓練自己的標的

Open **MultiTUNAparty Training**. A console window appears, then your browser at
`http://localhost:7860`. **Leave the console window open** — closing it stops
the training. 那個黑色視窗不能關。

Your folder is made for you at install time, under Documents:

```
Documents\MultiTUNAparty\
  data\
    train\
      other\
        images\    your images   (.jpg .png .tif)
        masks\     your masks    (.png, same file name as the image)
    val\
      other\
        images\
        masks\
  outputs\         results are saved here
```

The **Data Path** box in the interface is already filled in with that folder, so
you can press *Check Data Structure* straight away. On the first run the folder
is empty and opens in Explorer for you. It is also in the Start menu as
**MultiTUNAparty Data Folder**.

- A mask is a black-and-white image: **white is the thing you want to find.**
- **20–60** images in `train` and **5–15** in `val` is enough to see a result.
- The folder name sets the task slot: `cell\`, `blood\`, `other\`. Use `other\`
  for a new target — it is already created.
- **Use Pre-trained Model is ticked by default, and its path is filled in.**
  This matters: training this architecture from scratch would need thousands of
  images, whereas continuing from the shipped weights needs a few dozen.

When it finishes, the new model is at
`Documents\MultiTUNAparty\outputs\models\best_model.pth`. To use it: open
MultiTUNAparty → **Model Management** → **Browse** → that file → **Load Model**,
and choose the **Other** task type when predicting.

<details>
<summary><b>"❌ Data path does not exist: data"</b></summary>

An older build defaulted that box to the relative path `data`, which resolved
inside the program's own folder in `%LOCALAPPDATA%`, where your images are not.
It is filled in automatically now — run `UPDATE.bat`. If you still see it, paste
this into the Data Path box with your own Windows user name:

```
C:\Users\<your account>\Documents\MultiTUNAparty\data
```
</details>

---

## The model

A multi-task, boundary-aware TransUNet. One set of weights serves all three
targets; a task embedding tells the decoder which kind of image it is looking at.

| | |
|---|---|
| **Encoder** | ViT-B/16 — 400×400 input, 16×16 patches, 768-d, 12 blocks, 12 heads |
| **Bridge** | ASPP, multi-scale dilated fusion → 256 channels |
| **Decoder** | 80 boundary-aware blocks over channel stages 256 → 128 → 64 → 32 (20 blocks each), CBAM channel + spatial attention in every block |
| **Task conditioning** | A learned 256-d embedding per task, projected and added inside every decoder block |
| **Heads** | Segmentation head, Sobel-gradient boundary head, and a boundary-aware refinement module that sharpens the mask using the predicted edges |
| **Supervision** | Deep supervision on intermediate decoder stages |
| **Size** | 128,365,995 parameters — 86.13 M (67.1%) encoder, 36.14 M (28.2%) decoder |
| **Compute** | ≈121 GMACs at 400×400, of which the decoder is 52.7% — the parameters sit in the encoder, the arithmetic in the decoder |

**Training defaults** (what the interface writes into `config_training_ui.yaml`):
AdamW at 1e-5 with weight decay 0.01, gradient clipping at 1.0,
`CosineAnnealingWarmRestarts(T_0=20, T_mult=2, eta_min=1e-7)`, 400 px patches at
stride 200, batch 4 with 2-step gradient accumulation, mixed precision on,
200 epochs. The loss is `OptimizedBoundaryAwareLoss`: bidirectional
class-balanced BCE on the mask plus a weighted boundary term, with per-task
foreground and boundary weights.

### Reading the numbers honestly

Worth knowing before you trust an IoU, on our data or yours: when a mask is
mostly foreground, overlap metrics stop discriminating. In the plant-cell set
used here the annotated foreground fraction is ≈0.91, so a predictor that
labelled *every pixel* as cell would already score IoU ≈ 0.91. Check the
foreground fraction of your own masks; where it is high, judge the model on
boundary-sensitive measures and on the object counts and morphology you actually
care about, not on IoU alone.

---

## Repository layout

```
MultiTUNAparty-Windows-Installer/   the shipped Windows package
  INSTALL.bat                       double-click this
  UPDATE.bat                        refresh an existing installation
  README_INSTALL.txt                the same instructions, bilingual, offline
  installer/
    setup.ps1                       the installer itself, 8 steps
    update_existing.ps1             what UPDATE.bat runs
    launcher.py                     opens the GUI with the model preloaded
    train_launcher.py               opens the training UI, checks its packages first
    check_setup_ps1.py              guards setup.ps1 against the encoding and
                                    quoting bugs that broke the first release
    make_icons.py                   generates both .ico files from vector code
    requirements-core.txt           needed to predict; a failure is fatal
    requirements-optional.txt       needed only to train; a failure is a warning
  app/                              the application source, bundled

model_multitask_boundaryversion.py  the architecture
dataset_multitask.py                dataset, task mapping, patch sampling
losses_multitask.py                 loss functions
train_multitask_optimized.py        the training loop
app_gui_en.py                       prediction and measurement interface (Tkinter)
app_train_en.py                     training interface (Gradio)
```

### Running from source / 進階：直接跑原始碼

For anyone who does want the command line:

```bash
conda create -n multitask python=3.11 && conda activate multitask
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r MultiTUNAparty-Windows-Installer/installer/requirements-core.txt
pip install -r MultiTUNAparty-Windows-Installer/installer/requirements-optional.txt

python app_gui_en.py                                    # predict and measure
python app_train_en.py                                  # training interface, port 7860
python train_multitask_optimized.py --config config_training_ui.yaml   # headless
```

`best_model.pth` is a Releases asset rather than a repository file — it is
larger than GitHub's 100 MiB per-file limit for repository contents.
So you can find one on Google Drive here:
https://drive.google.com/drive/folders/1DDyytvb28C9CVCamu7t48qhi8ZFkQXye?usp=sharing

---

## Citation

If this software is useful in your work, please cite it. The manuscript
describing the method is in review; until it appears, cite the software:

```bibtex
@software{wang_multitunaparty,
  author  = {Wang, Shitephen},
  title   = {{MultiTUNAparty}: an open, one-click workflow for segmenting and
             quantifying cells, roots and fungal structures without programming},
  url     = {https://github.com/gn03138868/multiTUNAparty-model},
  year    = {2025}
}
```

---

## Licence and contact

MIT. Use it, change it, ship it; keep the notice.
(The full text ships as `MultiTUNAparty-Windows-Installer/LICENSE.txt`.)

**Shitephen Wang**
School of Forestry and Resource Conservation, National Taiwan University
[shitephenwang@ntu.edu.tw](mailto:shitephenwang@ntu.edu.tw) ·
[gn03138868@gmail.com](mailto:gn03138868@gmail.com)

Bug reports and "it did not work on my computer" reports are both welcome in
[Issues](../../issues) — the second kind is more useful than it sounds, since
the point of this project is that it works on an ordinary lab computer.
