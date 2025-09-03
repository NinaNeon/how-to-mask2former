# how-to-mask2former

(base) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$ conda activate mask2former

(mask2former) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$

python script.py --mode train

demo cpu

python demo/demo.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --input demo_inputs/input.jpg \
  --output demo_inputs/output_panoptic_cpu.png \
  --opts MODEL.DEVICE cpu

cd ~/Mask2Former

# 先試載入擴充（不一定必要，但能提早發現動態連結問題）
python - <<'PY'
import importlib
import MultiScaleDeformableAttention as m
print("MultiScaleDeformableAttention loaded ✅")
PY

# 跑 GPU demo
python demo/demo.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --input demo_inputs/input.jpg \
  --output demo_inputs/output_panoptic_gpu.png \
  --opts MODEL.DEVICE cuda
cp ~/Mask2Former/demo_inputs/output_panoptic_gpu.png /mnt/c/Users/USER/Desktop/output_panoptic_gpu.png
