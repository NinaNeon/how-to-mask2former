# how-to-mask2former

(base) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$ conda activate mask2former

(mask2former) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$

python script.py --mode train

demo

python demo/demo.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --input demo_inputs/input.jpg \
  --output demo_inputs/output_panoptic_cpu.png \
  --opts MODEL.DEVICE cpu

