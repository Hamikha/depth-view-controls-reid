# Depth-Based View Synthesis for Few-Shot Person Re-Identification: A Controlled Evaluation with Monocular and RGB-D Depth

Code, per-split results and analysis scripts for the paper of the same title.

## Summary

Recent few-shot person re-identification methods add training views rendered through a 3D reconstruction and credit their value to its shape. This study tests that premise for depth-image-based rendering (DIBR) from monocular estimated depth. One network design is trained on the same real images and identity splits under every condition, and conditions differ only in the images added or in a warp applied to the real images. Three controls apply the same rotation to a plane or to another image's depth, or apply a random two-dimensional perspective warp. Networks learn from 1, 2, 5 or 10 labeled images per identity over ten paired identity splits on three datasets: two ground-level video datasets (PRID-2011, iLIDS-VID) and the overhead color-and-depth (RGB-D) dataset TVRID. The main backbone is an ImageNet-pretrained ResNet-50, and the main conditions on PRID-2011 and iLIDS-VID are repeated with DINOv2 ViT-S/14. On TVRID the study also compares views from sensor depth with views from estimated depth, trains networks on depth alone and fuses color and depth scores. The findings are reported in the paper.

## View sources and controls

Each view source gives the depth used for warping. Views are rendered at fixed yaw angles (default ±15° and ±30° on PRID-2011 and iLIDS-VID, and yaw and pitch of ±10° and ±20° on TVRID).

| View source (paper) | Code name (`--geometry` of `render_views.py`, view tag in results) | What it is |
|---|---|---|
| Estimated-depth views | `estimated`, tag `estimated` | DIBR with depth from Depth Anything V2 Small |
| Matched flat views (control) | `flat_matched`, tag `flatm` | Constant depth, so the image turns like a flat picture (one homography of the whole image, no shape). Person mask, rotation center and hole filling come from the image's own estimated depth |
| Shuffled-depth views (control) | `shuffled`, tag `shuffled` | Warped with the estimated depth of another training image, so the depth statistics are real but the shape is wrong |
| Center-pivot flat views | `flat`, tag `flat` | Constant depth, rotation about the image center, nearest-neighbor hole filling, no depth estimation needed |
| Sensor-depth views (TVRID only) | `measured`, tag `measured` | DIBR with the 16-bit depth map recorded with each TVRID frame |

The third control, the random perspective warp, is not a view source. It moves each image corner by up to 0.15 of the half-width and half-height and is applied in place to real images with probability 0.5 (conditions Light+P and Strong+P).

Training conditions of the paper (Table 1) and their names in the code (`--arms` of `run_grid.py`):

| Paper | Code arm |
|---|---|
| Light | `real` |
| Strong | `real_aug` |
| Light+V | `real_synth_weak` |
| Strong+V | `real_synth` |
| Views-only (Light) | `synth_weak` |
| Light+P | `real_persp` |
| Strong+P | `real_aug_persp` |

The arm `synth` (views only with Strong augmentation) also exists in the code and in the result files.

## Repository layout

```
reid3d/                  the library
  data.py                dataset loaders, identity splits, few-shot sampling, image and depth decoding
  monocular.py           Depth Anything V2 Small wrapper
  geometry.py            DIBR rendering, flat and shuffled controls
  augment.py             Light and Strong augmentation, random perspective warp
  model.py               ResNet-50 and DINOv2 ViT-S/14 re-identification networks and losses
  train.py               one condition: training, feature extraction, scoring, cost
  metrics.py             CMC, mAP, score fusion
  stats.py               paired intervals, sign-flip test, Holm adjustment, decision rule
run_paper.py             the whole study, stage by stage
estimate_depth.py        monocular depth for the frames that are used
render_views.py          views from estimated, flat, shuffled or sensor depth
run_grid.py              trains and scores a grid of conditions and appends one JSON line per model
analyze.py               summary tables and paired comparisons from result files
speed_test.py            short speed test of a GPU
make_example_views.py    example-view panels (Figure 1 inputs)
make_figure1.py          Figure 1 from the example-view panels
make_result_figures.py   result figures from result files
make_extra_figures.py    view-angle and backbone figures from result files
mesh_views.py            bridge to externally generated mesh views (not used in the paper)
tests/                   automated tests (pytest)
results/                 per-split result files and reports behind the paper tables
results/timing/          the separate timing run behind Table S6
analysis/                scripts and outputs for Tables 3, S11a, S11b, S13a, S13b and S14
```

## Datasets

The datasets are not included in this repository. Obtain each one from its provider and use it under that provider's terms.

| Dataset | Where to obtain it | Layout expected by `reid3d/data.py` |
|---|---|---|
| PRID-2011 | https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/prid11 | `<root>/multi_shot/cam_a/person_0001/*.png` (or `<root>/prid_2011/multi_shot/...`) |
| iLIDS-VID | https://xiatian-zhu.github.io/downloads_qmul_iLIDS-VID_ReID_dataset.html | `<root>/i-LIDS-VID/sequences/cam1/person001/*.png` (or `<root>/sequences/...`) |
| TVRID | https://doi.org/10.5281/zenodo.17909410 | `<root>/train/<person>/<camera>/<passage>/*_RGB.png` with matching `*_depth.png` |

For PRID-2011 the loader keeps the 178 identities whose tracklets in both cameras have more than 27 frames. For iLIDS-VID it uses all 300 identities. For TVRID it uses the official training partition of 62 identities.

## Installation

```
python -m venv .venv
.venv\Scripts\activate          (Linux and macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m pytest -q tests
```

`requirements.txt` gives minimum versions. The exact versions of Python, PyTorch, torchvision and Hugging Face Transformers used for the paper are [to be added]. Pretrained weights are downloaded on first use: torchvision `IMAGENET1K_V2` (ResNet-50), `facebook/dinov2-small` and `depth-anything/Depth-Anything-V2-Small-hf`. The test suite uses small synthetic data and a stand-in depth model, so it needs no dataset.

## Identity splits

All conditions share ten random identity splits with seeds 0 to 9, so comparisons are paired by split. The splits are not stored as files. `reid3d.data.make_split` draws them from the seed (`random.Random(f"split-{seed}")` over the sorted identity list), so the same seed always gives the same split for the same dataset. Half of the identities are used for training and half for testing (89/89 on PRID-2011, 150/150 on iLIDS-VID, 31/31 on TVRID). `reid3d.data.sample_images` draws the *k* training images per identity with its own seed per split and identity, and the set for a smaller *k* is a subset of the set for every larger *k*.

## Reproducing the study

`run_paper.py` runs the study in stages. Every stage can be restarted, because finished work is skipped. Start each stage with `--dry-run`, which prints the commands and the number of planned models without running them.

Important: the default `--results` folder is `results/` in this repository, which already holds the released result files. Runs whose key is already in a result file are skipped, so pass a new folder such as `--results my_results` to train the models again.

```
python run_paper.py --stage prepare --prid DATA/prid --ilids DATA/ilids --tvrid DATA/tvrid --work work --results my_results --dry-run
python run_paper.py --stage prepare --prid DATA/prid --ilids DATA/ilids --tvrid DATA/tvrid --work work --results my_results
python run_paper.py --stage q1      --prid DATA/prid --ilids DATA/ilids --work work --results my_results
python run_paper.py --stage q2      --tvrid DATA/tvrid --work work --results my_results
python run_paper.py --stage report  --prid DATA/prid --ilids DATA/ilids --tvrid DATA/tvrid --work work --results my_results
python run_paper.py --stage extra   --prid DATA/prid --ilids DATA/ilids --work work --results my_results
python run_paper.py --stage fix     --prid DATA/prid --ilids DATA/ilids --tvrid DATA/tvrid --work work --results my_results
```

| Stage | What it does | Result files |
|---|---|---|
| `prepare` | Estimates depth (`estimate_depth.py`) and renders estimated-depth and center-pivot flat views (`render_views.py`), and sensor-depth views on TVRID, at full resolution and at 64 pixels for TVRID | derived data in `--work` |
| `q1` | PRID-2011 and iLIDS-VID with ResNet-50: Light, Strong, views with Light and Strong augmentation, Views-only, center-pivot flat views, and the untrained network | `q1_<dataset>.split<s>.jsonl` |
| `q2` | TVRID at full resolution and 64 pixels: color, sensor depth and estimated depth as inputs, views from each depth source, score fusion with several weights, and the untrained network | `q2_tvrid*.split<s>.jsonl` |
| `report` | Reports for the questions above | `report_q1.txt`, `report_q2.txt`, `report_q3.txt` |
| `extra` | The single-angle sets (±15°, ±30°, ±45°) and the main conditions with DINOv2 ViT-S/14, with their reports | `ab_*`, `bb_*`, `report_angles.txt`, `report_backbone.txt` |
| `fix` | The control run: matched flat views, shuffled-depth views and the random perspective warp (Light+P, Strong+P), matched flat views at single angles, DINOv2 with matched flat views and Light+P, and matched flat views on TVRID, with one report over all result files | `fx_*`, `report_fix.txt` |

The stage `all` runs `prepare`, `q1`, `q2` and `report`. Run `extra` and then `fix` on their own. `--workers 3` trains three identity splits at once, as in the paper. A report never replaces an earlier one, and a new report gets a numbered name (for example `report_q1_2.txt`). In `results/` the report with the highest number is the latest one.

The steps can also be run by hand, for example:

```
python estimate_depth.py --dataset prid2011 --data-root DATA/prid --out work/prid2011/estimated_depth
python render_views.py --dataset prid2011 --data-root DATA/prid --geometry flat_matched --est-root work/prid2011/estimated_depth --out work/prid2011/views_flatm
python run_grid.py --dataset prid2011 --data-root DATA/prid --arms real real_synth_weak --views flatm=work/prid2011/views_flatm --k 1 2 --splits 0 --out my_results/test.jsonl
```

Each line of a result file holds the full configuration of one model and its Rank-1, Rank-5, Rank-10, Rank-20, mAP and timing. Paths in the configuration fields are recorded as given on the command line (relative paths in the released files).

## Analysis and tables

The analysis reads only the result files, so it runs without the datasets or a GPU. Run these commands from the repository root:

```
python analysis/focal_comparisons.py results > focal.txt
python analysis/dataset_contrast.py results > contrast.txt
python analyze.py "results/*.jsonl" --question fix
python analyze.py "results/q2_*.jsonl" --baseline rgb/real_synth/estimated --treatment rgb/real_synth/measured
python analyze.py "results/q2_*.jsonl" --baseline depth/real_aug --treatment est_depth/real_aug
```

The first two commands reproduce `analysis/focal_comparisons_output_v5.txt` and `analysis/dataset_contrast_output.txt`. The permutation test in `dataset_contrast.py` enumerates all 184,756 divisions of 20 values, so it takes a few minutes.

| Paper table | Content | Produced by | Result files used |
|---|---|---|---|
| Table 3 | Family A, views against 2D, both backbones, with Holm and Nadeau-Bengio *p* | `analysis/focal_comparisons.py` (Family A) and `analysis/focal_comparisons_output_v5.txt` | `q1_*`, `bb_*` |
| Table 4 | Rank-1 gain by view source, C1 and C2 | `results/report_fix.txt` (sections on every view source against 2D, estimated-depth against matched flat views, and own against borrowed depth), and the NB *p* of C1 from `analysis/focal_comparisons_output_v5.txt` | `q1_*`, `fx_*` |
| Table 5 | TVRID: sensor against estimated depth, views, depth input and fusion | `results/report_fix.txt` (condition means and differences against matched flat views), `results/report_q2_3.txt` (depth input and fusion weights), and the two `analyze.py --baseline ... --treatment ...` commands above (intervals for sensor minus estimated views and for estimated minus sensor depth input) | `q2_tvrid*`, `fx_tvrid*` |
| Table S6 | Costs from the separate timing run | `results/timing/report_q1.txt`, `report_q2.txt`, `report_q3.txt` | `results/timing/*.jsonl` |
| Tables S11a and S11b | Families A and B under four inferential readings, and the Nadeau-Bengio test | `analysis/focal_comparisons_output_v5.txt` | `q1_*`, `bb_*` |
| Tables S13a and S13b | Declared control families C1, C2 and C3 | `analysis/focal_comparisons_output_v5.txt` | `q1_*`, `bb_*`, `fx_*` |
| Table S14 | Post hoc contrasts of the shape effect, Rank-1 and mAP | `analysis/dataset_contrast.py` and `analysis/dataset_contrast_output.txt` | `q1_*`, `fx_*` |

The other supplementary tables come from the reports in `results/`: `report_angles.txt` (view angles), `report_backbone.txt` (DINOv2), `report_q1_3.txt`, `report_q2_3.txt` and `report_q3_3.txt` (main runs) and `report_fix.txt` (control run).

The families of the control comparisons C1 to C3 are defined in `analysis/focal_comparisons.py` (`FAMILIES`). The decision rule is in `reid3d/stats.py`.

## Figures

`make_example_views.py` and `make_figure1.py` build Figure 1 from rendered views, so they need the datasets and the `--work` folder. `make_result_figures.py` and `make_extra_figures.py` build the result figures from the result files and also need matplotlib, which is not listed in `requirements.txt`. `make_figure1.py`, `make_result_figures.py` and `make_extra_figures.py` stop if the Times New Roman font is not installed.

## Hardware and run time

All models of the paper were trained on one NVIDIA RTX 5090 graphics processor (32 GB), three at a time. Run times on other hardware are not guaranteed. `speed_test.py` measures the training speed of a new GPU without any dataset, and `run_paper.py` prints the number of planned models, which together give a rough time estimate.

## Citation

If you use this code, please cite:

> Depth-Based View Synthesis for Few-Shot Person Re-Identification: A Controlled Evaluation with Monocular and RGB-D Depth. [citation to be added after publication]

## License

MIT License, see `LICENSE`. The datasets and pretrained weights are covered by their own licenses and terms.
