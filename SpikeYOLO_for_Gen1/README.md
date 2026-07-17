The purpose of this document is to illustrate the training method of SpikeYOLO on Gen1 data(GEN1's experiments were performed based on spikingjelly).
## 1. Data download:
https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/

## 2. preprocessing: python pre_gen1.py
Refer to environmental requirements “https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/”

## 3. install spikingjelly
cd spikingjelly-0.0.0.0.12
python setup.py install

## 4.train
python train.py

## 5.test / get_firing_rate
python test.py

## 6. Jellyfish video inference demo

`video_inference_delta.py` runs the jellyfish SNN detector (RGB adaptation of SpikeYOLO, see
`customdataset.py`/`spike_trainer.py`) over a video clip: each frame is edge-filtered (Canny) and
spike-encoded via delta modulation (one spike train per frame, comparing it against the previous
frame) before going through the model. The clips below show the Canny edge map the model actually
sees, annotated with the predicted species and confidence.

<table>
<tr>
<td align="center"><b>R- pulmo</b></td>
<td align="center"><b>P- noctiluca</b></td>
<td align="center"><b>C- tuberculata</b></td>
</tr>
<tr>
<td><video src="video/pulmo_video_pred_delta_batch.mp4" controls width="260"></video></td>
<td><video src="video/pelagia_video_pred_delta_batch.mp4" controls width="260"></video></td>
<td><video src="video/huevo_frito_video_pred_delta_batch.mp4" controls width="260"></video></td>
</tr>
</table>