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

# 在 Mask2Former 專案根目錄
python demo/demo.py \
  --config-file configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
  --input demo_inputs/input.jpg \
  --output demo_inputs/output_panoptic_gpu.png \
  --opts \
    MODEL.DEVICE cuda \
    MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl\
cp ~/Mask2Former/demo_inputs/output_panoptic_gpu.png /mnt/c/Users/USER/Desktop/output_panoptic_gpu.png
<img width="942" height="537" alt="螢幕擷取畫面 2025-09-03 152404" src="https://github.com/user-attachments/assets/abebd819-c5d2-4a24-89be-97bc60fed4e2" />



# 基本训练（Wide ResNet-50）
python script.py --mode train

# 使用 UNet 增强
python script.py --mode train --use-unet

# 评估模型
python script.py --mode eval --model-path ./segmentation_output_*/model_final.pth

# 可视化结果
python script.py --mode viz --model-path ./segmentation_output_*/model_final.pth




# 3) 先做 20 iter「煙霧測試」

確保資料讀得到、能跑完整訓練 loop，之後再回到正式參數。

```bash
# 把 iter/batch 暫時調小（如果你的腳本裡這兩行存在）
sed -i -E 's/(cfg\.SOLVER\.MAX_ITER\s*=\s*)[0-9]+/\1 20/' /mnt/c/Users/USER/Desktop/script.py
sed -i -E 's/(cfg\.SOLVER\.IMS_PER_BATCH\s*=\s*)[0-9]+/\1 2/'  /mnt/c/Users/USER/Desktop/script.py

cd /mnt/c/Users/USER/Desktop
# 用你剛剛能建起來的路徑先跑（--use-unet 成功率最高）
python script.py --mode train --use-unet
```

完成後目錄會出現 `segmentation_output_.../`，裡面應該有權重檔（名稱依你的程式而定）。接著：

```bash
LATEST=$(ls -dt segmentation_output_* | head -n 1)
echo "LATEST = $LATEST"
ls -lh "$LATEST"

# 如果有 model_final.pth 就帶它；若叫別的檔名，改成那個實名
python script.py --mode eval --model-path "$LATEST/model_final.pth"
python script.py --mode viz  --model-path "$LATEST/model_final.pth"
```

> **eval/viz 找不到模型** 的主因就是訓練沒有產生模型檔（或路徑沒展開），按上面取 `LATEST` 的做法就能避開萬用字元無法展開的問題。

---

# 4) 正式訓練前，把參數改回去

```bash
sed -i -E 's/(cfg\.SOLVER\.MAX_ITER\s*=\s*)20/\1 8000/' /mnt/c/Users/USER/Desktop/script.py
sed -i -E 's/(cfg\.SOLVER\.IMS_PER_BATCH\s*=\s*)2/\1 4/'  /mnt/c/Users/USER/Desktop/script.py
```



