(base) jovyan@unzip-workspace-0-10:~$ mv work/vv.py /mnt/nfs/nina/nina/
(base) jovyan@unzip-workspace-0-10:~$ cd /mnt/nfs/nina/nina/
(base) jovyan@unzip-workspace-0-10:/mnt/nfs/nina/nina$ python vv.py
Traceback (most recent call last):
  File "/mnt/nfs/nina/nina/vv.py", line 6, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
(base) jovyan@unzip-workspace-0-10:/mnt/nfs/nina/nina$ rm /mnt/nfs/nina/nina/vv.py
(base) jovyan@unzip-workspace-0-10:/mnt/nfs/nina/nina$ python vv.py
Traceback (most recent call last):
  File "/mnt/nfs/nina/nina/vv.py", line 6, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
(base) jovyan@unzip-workspace-0-10:/mnt/nfs/nina/nina$ nvcc --version
bash: nvcc: command not found
(base) jovyan@unzip-workspace-0-10:/mnt/nfs/nina/nina$ nvidia-smi
Sat Sep  6 13:30:07 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L40                     On  |   00000000:1C:00.0 Off |                    0 |
| N/A   27C    P8             35W /  300W |       5MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
(base) jovyan@unzip-workspace-0-10:/mnt/nfs/nina/nina$ 





# 建議別在 base，先建乾淨環境（可選）
# python -m venv ~/venvs/torch121 && source ~/venvs/torch121/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121















# how-to-mask2former

(base) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$ conda activate mask2former

(mask2former) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$

python script.py --mode train

demo cpu
cd ~/Mask2Former

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

python demo/demo.py --config-file configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml --input demo_inputs/input.jpg --output demo_inputs/output_panoptic_gpu.png --opts MODEL.DEVICE cuda MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl


# 基本训练（Wide ResNet-50）

conda activate mask2former
cd ~/Mask2Former
python /mnt/c/Users/USER/Desktop/mask2former_training.py --mode train --use-unet


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



你現在的錯誤又回到 `KeyError: 'res5'`——意思是**骨幹輸出是 FPN（p2–p5），但 head/decoder 仍在吃 res2–res5**。最小風險的修法：只在 `cfg.merge_from_file(...)` 後面加幾行覆寫，把特徵名強制改成 `["p2","p3","p4","p5"]`（同時順手允許未知鍵，避免之前的 `STEM_TYPE` 類似報錯）。下面的補丁是「精準插入 + 自動偵測縮排」，不會破壞函式結構。

**照貼跑：**

```bash
cd /mnt/c/Users/USER/Desktop

# 安全補丁：在 add_maskformer2_config 後插入 set_new_allowed(True)；
# 在 merge_from_file 後插入 IN_FEATURES 覆寫為 p2~p5
python - <<'PY'
import time
p="script.py"
s=open(p,"r",encoding="utf-8").read()
open(f"script.py.bak.{int(time.time())}","w",encoding="utf-8").write(s)  # 備份

lines=s.splitlines()
def insert_after(substr, new_lines):
    for i,l in enumerate(lines):
        if substr in l:
            indent = l[:len(l)-len(l.lstrip())]
            for k,x in enumerate(new_lines):
                lines.insert(i+1+k, indent + x)
            return True
    return False

changed=False

# A) 允許未知鍵（防 STEM_TYPE 之類）
if "cfg.set_new_allowed(True)" not in s:
    if insert_after("add_maskformer2_config(cfg)", ["cfg.set_new_allowed(True)  # allow non-registered keys"]):
        changed=True

# B) 強制用 p2~p5（解 'res5'）
if 'SEM_SEG_HEAD.IN_FEATURES' not in s:
    if insert_after("cfg.merge_from_file", [
        '# force FPN features to p2..p5 to avoid KeyError: res5',
        'cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2","p3","p4","p5"]',
        'try:',
        '    cfg.MODEL.PIXEL_DECODER.IN_FEATURES = ["p2","p3","p4","p5"]',
        'except Exception:',
        '    pass',
    ]):
        changed=True

if changed:
    open(p,"w",encoding="utf-8").write("\n".join(lines)+"\n")
    print("✅ Patch applied.")
else:
    print("ℹ️ Nothing to patch (maybe already applied).")
PY
```

**確認有插入到：**

```bash
grep -n "set_new_allowed" script.py
grep -n "IN_FEATURES" script.py
```

**先做 20 iter 煙霧測試（避免一次長訓）：**

```bash
sed -i -E 's/(cfg\.SOLVER\.MAX_ITER\s*=\s*)[0-9]+/\1 20/' script.py
sed -i -E 's/(cfg\.SOLVER\.IMS_PER_BATCH\s*=\s*)[0-9]+/\1 2/'  script.py

python script.py --mode train --use-unet
```

> 這一步應該不會再看到 `'res5'`。若還有，我再給你「把 backbone 也硬指定為 FPN 並覆寫 IN\_FEATURES」的第二層保險。

**一切 OK 後**再把參數改回來跑正式訓練：

```bash
sed -i -E 's/(cfg\.SOLVER\.MAX_ITER\s*=\s*)20/\1 8000/' script.py
sed -i -E 's/(cfg\.SOLVER\.IMS_PER_BATCH\s*=\s*)2/\1 4/'  script.py
```

---

如果你想直接改「不用 FPN、回到 res2–res5」也可以，在 `cfg.merge_from_file(...)` 後面額外塞這幾行（擇一方案，不要同時用）：

```python
# 用 ResNet 主幹 + res2~res5（而非 FPN）
cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2","res3","res4","res5"]
try:
    cfg.MODEL.PIXEL_DECODER.IN_FEATURES = ["res2","res3","res4","res5"]
except Exception:
    pass
```

但以你現在的 `WideResNetFPNWithUNet` 來看，**改成 p2–p5**是最自然、侵入最小的解法。

抱歉，剛剛一堆補丁把檔案弄亂，讓你超煩。我把重點收斂成**一個精準修正**：你的骨幹是 **FPN**（輸出 `p2~p5`），但腳本把 head 設成吃 `res2~res5`，所以一直 `KeyError: 'res5'`。把所有 `SEM_SEG_HEAD.IN_FEATURES`（以及 `PIXEL_DECODER.IN_FEATURES`）強制改為 `["p2","p3","p4","p5"]` 就好。

直接貼這些指令跑：

```bash
cd /mnt/c/Users/USER/Desktop

# 1) 把 SEM_SEG_HEAD.IN_FEATURES 從 res2..res5 改成 p2..p5
sed -i -E 's/(cfg\.MODEL\.SEM_SEG_HEAD\.IN_FEATURES\s*=\s*)\["res2", "res3", "res4", "res5"\]/\1["p2", "p3", "p4", "p5"]/' script.py

# 2) 若附近沒有幫 PIXEL_DECODER 設定，補上（自動插在 SEM_SEG_HEAD 那行後面）
python - <<'PY'
import re
p="script.py"
txt=open(p,"r",encoding="utf-8").read().splitlines()
for i,l in enumerate(txt):
    if re.search(r'cfg\.MODEL\.SEM_SEG_HEAD\.IN_FEATURES\s*=', l):
        indent = l[:len(l)-len(l.lstrip())]
        have=False
        for j in range(i+1, min(i+6,len(txt))):
            if 'PIXEL_DECODER.IN_FEATURES' in txt[j]:
                have=True; break
        if not have:
            txt[i+1:i+1] = [
                indent+'try:',
                indent+'    cfg.MODEL.PIXEL_DECODER.IN_FEATURES = ["p2", "p3", "p4", "p5"]',
                indent+'except Exception:',
                indent+'    pass',
            ]
        break
open(p,"w",encoding="utf-8").write("\n".join(txt)+"\n")
print("✅ set PIXEL_DECODER.IN_FEATURES to p2..p5 (if missing)")
PY

# 3) 確認現在只剩 p2..p5（不該再看到 res2..res5）
grep -n "SEM_SEG_HEAD.IN_FEATURES" script.py
grep -n "PIXEL_DECODER.IN_FEATURES" script.py

# 4) 先短跑 20 iter（避免一次長訓）
sed -i -E 's/(cfg\.SOLVER\.MAX_ITER\s*=\s*)[0-9]+/\1 20/' script.py
sed -i -E 's/(cfg\.SOLVER\.IMS_PER_BATCH\s*=\s*)[0-9]+/\1 2/'  script.py

python script.py --mode train --use-unet
```

**預期結果**：不再出現 `'res5'`。如果還看到 `res5`，代表腳本裡**別的地方又把它覆寫回 res**\*，再跑這個查所有殘留並貼給我（我幫你一次剷除）：

```bash
grep -nE 'IN_FEATURES.*res[2345]' -n script.py || echo "OK: no res* left"
```

> 註：`cfg.MODEL.FPN.IN_FEATURES = ["res2","res3","res4","res5"]` 那行**可以保留**（那是告訴 FPN 從 ResNet 哪些 stage 取特徵來產生 `p2~p5`），真正要餵給 Mask2Former 的 head/decoder 才是 `p2~p5`。

等短跑 ok 之後，再把 iter / batch 改回你要的正式值。需要的話我也可以幫你加一個 CLI 參數 `--features p|res`，之後切換骨幹和特徵名稱不用再改檔。

