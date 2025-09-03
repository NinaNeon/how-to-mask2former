Welcome to Ubuntu 24.04.3 LTS (GNU/Linux 6.6.87.2-microsoft-standard-WSL2 x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/pro

 System information as of Wed Sep  3 12:54:39 CST 2025

  System load:  0.0                 Processes:             62
  Usage of /:   3.6% of 1006.85GB   Users logged in:       0
  Memory usage: 1%                  IPv4 address for eth0: 172.29.39.124
  Swap usage:   0%


This message is shown once a day. To disable it please create the
/home/nina/.hushlogin file.
(base) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$ conda activate mask2former

python -c "import torch; print(torch.cuda.is_available())"
True
(mask2former) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$ python -c "import detectron2; print('Detectron2 OK')"
Detectron2 OK
(mask2former) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$ python -c "import mask2former; print('Mask2Former OK')"
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/msdeformattn.py:314: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
Mask2Former OK
(mask2former) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$ cd ~/Mask2Former
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ ls demo
README.md  demo.py  predictor.py
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ cp /mnt/c/Users/USER/Pictures/test.jpg ~/Mask2Former/demo_inputs/input.jpg
cp: cannot create regular file '/home/nina/Mask2Former/demo_inputs/input.jpg': No such file or directory
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ mkdir ~/Mask2Former/demo_inputs
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ cp "/mnt/c/Users/USER/Pictures/test.jpg" ~/Mask2Former/demo_inputs/input.jpg
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ ls ~/Mask2Former/demo_inputs
input.jpg
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ python demo/demo.py \
  --config-file configs/coco/semantic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --input demo_inputs/input.jpg \
  --output demo_inputs/output.png \
  --opts MODEL.DEVICE cpu
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/home/nina/Mask2Former/demo/../mask2former/modeling/pixel_decoder/msdeformattn.py:314: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
[09/03 13:00:41 detectron2]: Arguments: Namespace(config_file='configs/coco/semantic-segmentation/maskformer2_R50_bs16_50ep.yaml', webcam=False, video_input=None, input=['demo_inputs/input.jpg'], output='demo_inputs/output.png', confidence_threshold=0.5, opts=['MODEL.DEVICE', 'cpu'])
Traceback (most recent call last):
  File "/home/nina/Mask2Former/demo/demo.py", line 106, in <module>
    cfg = setup_cfg(args)
  File "/home/nina/Mask2Former/demo/demo.py", line 39, in setup_cfg
    cfg.merge_from_file(args.config_file)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/detectron2/config/config.py", line 45, in merge_from_file
    assert PathManager.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
AssertionError: Config file 'configs/coco/semantic-segmentation/maskformer2_R50_bs16_50ep.yaml' does not exist!
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ ls
ADVANCED_USAGE.md   GETTING_STARTED.md  MODEL_ZOO.md  configs   demo_inputs  mask2former_video  tools
CODE_OF_CONDUCT.md  INSTALL.md          README.md     datasets  demo_video   predict.py         train_net.py
CONTRIBUTING.md     LICENSE             cog.yaml      demo      mask2former  requirements.txt   train_net_video.py
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ ls configs
ade20k  cityscapes  coco  mapillary-vistas  youtubevis_2019  youtubevis_2021
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ ls configs/coco
ls configs/cityscapes
ls configs/ade20k
instance-segmentation  panoptic-segmentation
instance-segmentation  panoptic-segmentation  semantic-segmentation
instance-segmentation  panoptic-segmentation  semantic-segmentation
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ ls configs/coco/panoptic-segmentation
Base-COCO-PanopticSegmentation.yaml  maskformer2_R101_bs16_50ep.yaml  maskformer2_R50_bs16_50ep.yaml  swin
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ python demo/demo.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --input demo_inputs/input.jpg \
  --output demo_inputs/output_panoptic_cpu.png \
  --opts MODEL.DEVICE cpu
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/home/nina/Mask2Former/demo/../mask2former/modeling/pixel_decoder/msdeformattn.py:314: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
[09/03 13:03:01 detectron2]: Arguments: Namespace(config_file='configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml', webcam=False, video_input=None, input=['demo_inputs/input.jpg'], output='demo_inputs/output_panoptic_cpu.png', confidence_threshold=0.5, opts=['MODEL.DEVICE', 'cpu'])
[09/03 13:03:02 fvcore.common.checkpoint]: [Checkpointer] Loading from detectron2://ImageNetPretrained/torchvision/R-50.pkl ...
R-50.pkl: 102MB [00:01, 78.7MB/s]
[09/03 13:03:03 fvcore.common.checkpoint]: Reading a file from 'torchvision'
[09/03 13:03:03 d2.checkpoint.c2_model_loading]: Following weights matched with submodule backbone:
| Names in Model    | Names in Checkpoint                                                               | Shapes                                          |
|:------------------|:----------------------------------------------------------------------------------|:------------------------------------------------|
| res2.0.conv1.*    | res2.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,1,1)             |
| res2.0.conv2.*    | res2.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,3,3)             |
| res2.0.conv3.*    | res2.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| res2.0.shortcut.* | res2.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| res2.1.conv1.*    | res2.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,256,1,1)            |
| res2.1.conv2.*    | res2.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,3,3)             |
| res2.1.conv3.*    | res2.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| res2.2.conv1.*    | res2.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,256,1,1)            |
| res2.2.conv2.*    | res2.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,3,3)             |
| res2.2.conv3.*    | res2.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| res3.0.conv1.*    | res3.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,256,1,1)       |
| res3.0.conv2.*    | res3.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| res3.0.conv3.*    | res3.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| res3.0.shortcut.* | res3.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (512,) (512,) (512,) (512,) (512,256,1,1)       |
| res3.1.conv1.*    | res3.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,512,1,1)       |
| res3.1.conv2.*    | res3.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| res3.1.conv3.*    | res3.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| res3.2.conv1.*    | res3.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,512,1,1)       |
| res3.2.conv2.*    | res3.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| res3.2.conv3.*    | res3.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| res3.3.conv1.*    | res3.3.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,512,1,1)       |
| res3.3.conv2.*    | res3.3.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| res3.3.conv3.*    | res3.3.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| res4.0.conv1.*    | res4.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,512,1,1)       |
| res4.0.conv2.*    | res4.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.0.conv3.*    | res4.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res4.0.shortcut.* | res4.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (1024,) (1024,) (1024,) (1024,) (1024,512,1,1)  |
| res4.1.conv1.*    | res4.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| res4.1.conv2.*    | res4.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.1.conv3.*    | res4.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res4.2.conv1.*    | res4.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| res4.2.conv2.*    | res4.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.2.conv3.*    | res4.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res4.3.conv1.*    | res4.3.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| res4.3.conv2.*    | res4.3.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.3.conv3.*    | res4.3.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res4.4.conv1.*    | res4.4.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| res4.4.conv2.*    | res4.4.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.4.conv3.*    | res4.4.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res4.5.conv1.*    | res4.5.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| res4.5.conv2.*    | res4.5.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.5.conv3.*    | res4.5.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res5.0.conv1.*    | res5.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,1024,1,1)      |
| res5.0.conv2.*    | res5.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,512,3,3)       |
| res5.0.conv3.*    | res5.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (2048,) (2048,) (2048,) (2048,) (2048,512,1,1)  |
| res5.0.shortcut.* | res5.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (2048,) (2048,) (2048,) (2048,) (2048,1024,1,1) |
| res5.1.conv1.*    | res5.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,2048,1,1)      |
| res5.1.conv2.*    | res5.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,512,3,3)       |
| res5.1.conv3.*    | res5.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (2048,) (2048,) (2048,) (2048,) (2048,512,1,1)  |
| res5.2.conv1.*    | res5.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,2048,1,1)      |
| res5.2.conv2.*    | res5.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,512,3,3)       |
| res5.2.conv3.*    | res5.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (2048,) (2048,) (2048,) (2048,) (2048,512,1,1)  |
| stem.conv1.*      | stem.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}      | (64,) (64,) (64,) (64,) (64,3,7,7)              |
WARNING [09/03 13:03:03 fvcore.common.checkpoint]: Some model parameters or buffers are not found in the checkpoint:
criterion.empty_weight
sem_seg_head.pixel_decoder.adapter_1.norm.{bias, weight}
sem_seg_head.pixel_decoder.adapter_1.weight
sem_seg_head.pixel_decoder.input_proj.0.0.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.0.1.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.1.0.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.1.1.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.2.0.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.2.1.{bias, weight}
sem_seg_head.pixel_decoder.layer_1.norm.{bias, weight}
sem_seg_head.pixel_decoder.layer_1.weight
sem_seg_head.pixel_decoder.mask_features.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.level_embed
sem_seg_head.predictor.class_embed.{bias, weight}
sem_seg_head.predictor.decoder_norm.{bias, weight}
sem_seg_head.predictor.level_embed.weight
sem_seg_head.predictor.mask_embed.layers.0.{bias, weight}
sem_seg_head.predictor.mask_embed.layers.1.{bias, weight}
sem_seg_head.predictor.mask_embed.layers.2.{bias, weight}
sem_seg_head.predictor.query_embed.weight
sem_seg_head.predictor.query_feat.weight
sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.0.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.1.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.2.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.3.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.4.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.5.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.6.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.7.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.8.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.0.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.0.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.0.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.1.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.1.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.1.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.2.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.2.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.2.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.3.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.3.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.3.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.4.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.4.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.4.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.5.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.5.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.5.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.6.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.6.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.6.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.7.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.7.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.7.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.8.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.8.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.8.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.0.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.1.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.2.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.3.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.4.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.5.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.6.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.7.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.8.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn.{in_proj_bias, in_proj_weight}
WARNING [09/03 13:03:03 fvcore.common.checkpoint]: The checkpoint state_dict contains keys that are not used by the model:
  stem.fc.{bias, weight}
  0%|                                                                                             | 0/1 [00:00<?, ?it/s]/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/functional.py:534: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
[09/03 13:03:06 detectron2]: demo_inputs/input.jpg: detected 100 instances in 2.77s
100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.84s/it]
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ ls -lh demo_inputs/output_panoptic_cpu.png
-rw-r--r-- 1 nina nina 403K Sep  3 13:03 demo_inputs/output_panoptic_cpu.png
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ python demo/demo.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --input demo_inputs/input.jpg \
  --output demo_inputs/output_panoptic_gpu.png \
  --opts MODEL.DEVICE cuda
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/home/nina/Mask2Former/demo/../mask2former/modeling/pixel_decoder/msdeformattn.py:314: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
[09/03 13:04:07 detectron2]: Arguments: Namespace(config_file='configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml', webcam=False, video_input=None, input=['demo_inputs/input.jpg'], output='demo_inputs/output_panoptic_gpu.png', confidence_threshold=0.5, opts=['MODEL.DEVICE', 'cuda'])
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/cuda/__init__.py:235: UserWarning:
NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5060 Ti GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(
[09/03 13:04:08 fvcore.common.checkpoint]: [Checkpointer] Loading from detectron2://ImageNetPretrained/torchvision/R-50.pkl ...
[09/03 13:04:08 fvcore.common.checkpoint]: Reading a file from 'torchvision'
[09/03 13:04:08 d2.checkpoint.c2_model_loading]: Following weights matched with submodule backbone:
| Names in Model    | Names in Checkpoint                                                               | Shapes                                          |
|:------------------|:----------------------------------------------------------------------------------|:------------------------------------------------|
| res2.0.conv1.*    | res2.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,1,1)             |
| res2.0.conv2.*    | res2.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,3,3)             |
| res2.0.conv3.*    | res2.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| res2.0.shortcut.* | res2.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| res2.1.conv1.*    | res2.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,256,1,1)            |
| res2.1.conv2.*    | res2.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,3,3)             |
| res2.1.conv3.*    | res2.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| res2.2.conv1.*    | res2.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,256,1,1)            |
| res2.2.conv2.*    | res2.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (64,) (64,) (64,) (64,) (64,64,3,3)             |
| res2.2.conv3.*    | res2.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,64,1,1)        |
| res3.0.conv1.*    | res3.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,256,1,1)       |
| res3.0.conv2.*    | res3.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| res3.0.conv3.*    | res3.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| res3.0.shortcut.* | res3.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (512,) (512,) (512,) (512,) (512,256,1,1)       |
| res3.1.conv1.*    | res3.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,512,1,1)       |
| res3.1.conv2.*    | res3.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| res3.1.conv3.*    | res3.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| res3.2.conv1.*    | res3.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,512,1,1)       |
| res3.2.conv2.*    | res3.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| res3.2.conv3.*    | res3.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| res3.3.conv1.*    | res3.3.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,512,1,1)       |
| res3.3.conv2.*    | res3.3.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (128,) (128,) (128,) (128,) (128,128,3,3)       |
| res3.3.conv3.*    | res3.3.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,128,1,1)       |
| res4.0.conv1.*    | res4.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,512,1,1)       |
| res4.0.conv2.*    | res4.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.0.conv3.*    | res4.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res4.0.shortcut.* | res4.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (1024,) (1024,) (1024,) (1024,) (1024,512,1,1)  |
| res4.1.conv1.*    | res4.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| res4.1.conv2.*    | res4.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.1.conv3.*    | res4.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res4.2.conv1.*    | res4.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| res4.2.conv2.*    | res4.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.2.conv3.*    | res4.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res4.3.conv1.*    | res4.3.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| res4.3.conv2.*    | res4.3.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.3.conv3.*    | res4.3.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res4.4.conv1.*    | res4.4.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| res4.4.conv2.*    | res4.4.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.4.conv3.*    | res4.4.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res4.5.conv1.*    | res4.5.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,1024,1,1)      |
| res4.5.conv2.*    | res4.5.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (256,) (256,) (256,) (256,) (256,256,3,3)       |
| res4.5.conv3.*    | res4.5.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (1024,) (1024,) (1024,) (1024,) (1024,256,1,1)  |
| res5.0.conv1.*    | res5.0.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,1024,1,1)      |
| res5.0.conv2.*    | res5.0.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,512,3,3)       |
| res5.0.conv3.*    | res5.0.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (2048,) (2048,) (2048,) (2048,) (2048,512,1,1)  |
| res5.0.shortcut.* | res5.0.shortcut.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight} | (2048,) (2048,) (2048,) (2048,) (2048,1024,1,1) |
| res5.1.conv1.*    | res5.1.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,2048,1,1)      |
| res5.1.conv2.*    | res5.1.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,512,3,3)       |
| res5.1.conv3.*    | res5.1.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (2048,) (2048,) (2048,) (2048,) (2048,512,1,1)  |
| res5.2.conv1.*    | res5.2.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,2048,1,1)      |
| res5.2.conv2.*    | res5.2.conv2.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (512,) (512,) (512,) (512,) (512,512,3,3)       |
| res5.2.conv3.*    | res5.2.conv3.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}    | (2048,) (2048,) (2048,) (2048,) (2048,512,1,1)  |
| stem.conv1.*      | stem.conv1.{norm.bias,norm.running_mean,norm.running_var,norm.weight,weight}      | (64,) (64,) (64,) (64,) (64,3,7,7)              |
WARNING [09/03 13:04:08 fvcore.common.checkpoint]: Some model parameters or buffers are not found in the checkpoint:
criterion.empty_weight
sem_seg_head.pixel_decoder.adapter_1.norm.{bias, weight}
sem_seg_head.pixel_decoder.adapter_1.weight
sem_seg_head.pixel_decoder.input_proj.0.0.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.0.1.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.1.0.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.1.1.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.2.0.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.2.1.{bias, weight}
sem_seg_head.pixel_decoder.layer_1.norm.{bias, weight}
sem_seg_head.pixel_decoder.layer_1.weight
sem_seg_head.pixel_decoder.mask_features.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.level_embed
sem_seg_head.predictor.class_embed.{bias, weight}
sem_seg_head.predictor.decoder_norm.{bias, weight}
sem_seg_head.predictor.level_embed.weight
sem_seg_head.predictor.mask_embed.layers.0.{bias, weight}
sem_seg_head.predictor.mask_embed.layers.1.{bias, weight}
sem_seg_head.predictor.mask_embed.layers.2.{bias, weight}
sem_seg_head.predictor.query_embed.weight
sem_seg_head.predictor.query_feat.weight
sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.0.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.1.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.2.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.3.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.4.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.5.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.6.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.7.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.8.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.0.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.0.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.0.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.1.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.1.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.1.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.2.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.2.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.2.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.3.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.3.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.3.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.4.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.4.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.4.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.5.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.5.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.5.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.6.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.6.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.6.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.7.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.7.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.7.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.8.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.8.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.8.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.0.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.1.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.2.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.3.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.4.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.5.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.6.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.7.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.8.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn.{in_proj_bias, in_proj_weight}
WARNING [09/03 13:04:08 fvcore.common.checkpoint]: The checkpoint state_dict contains keys that are not used by the model:
  stem.fc.{bias, weight}
  0%|                                                                                             | 0/1 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/nina/Mask2Former/demo/demo.py", line 118, in <module>
    predictions, visualized_output = demo.run_on_image(img)
  File "/home/nina/Mask2Former/demo/predictor.py", line 49, in run_on_image
    predictions = self.predictor(image)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/detectron2/engine/defaults.py", line 317, in __call__
    predictions = self.model([inputs])[0]
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/nina/Mask2Former/demo/../mask2former/maskformer_model.py", line 194, in forward
    images = [(x - self.pixel_mean) / self.pixel_std for x in images]
  File "/home/nina/Mask2Former/demo/../mask2former/maskformer_model.py", line 194, in <listcomp>
    images = [(x - self.pixel_mean) / self.pixel_std for x in images]
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('CC:', torch.cuda.get_device_capability(0))"
nvidia-smi
torch 2.5.1+cu124 cuda 12.4
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/cuda/__init__.py:235: UserWarning:
NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5060 Ti GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(
GPU: NVIDIA GeForce RTX 5060 Ti
CC: (12, 0)
Wed Sep  3 14:37:51 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.76.04              Driver Version: 580.97         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti     On  |   00000000:01:00.0  On |                  N/A |
|  0%   34C    P8              4W /  180W |    1079MiB /  16311MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ pip uninstall -y torch torchvision torchaudio
pip cache purge
Found existing installation: torch 2.5.1+cu124
Uninstalling torch-2.5.1+cu124:
  Successfully uninstalled torch-2.5.1+cu124
Found existing installation: torchvision 0.20.1+cu124
Uninstalling torchvision-0.20.1+cu124:
  Successfully uninstalled torchvision-0.20.1+cu124
WARNING: Skipping torchaudio as it is not installed.
Files removed: 304 (2922.5 MB)
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
Looking in indexes: https://download.pytorch.org/whl/nightly/cu128
Collecting torch
  Downloading https://download.pytorch.org/whl/nightly/cu128/torch-2.9.0.dev20250902%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (31 kB)
Collecting torchvision
  Downloading https://download.pytorch.org/whl/nightly/cu128/torchvision-0.24.0.dev20250902%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (5.9 kB)
Collecting torchaudio
  Downloading https://download.pytorch.org/whl/nightly/cu128/torchaudio-2.8.0.dev20250902%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (7.3 kB)
Requirement already satisfied: filelock in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from torch) (3.17.0)
Requirement already satisfied: typing-extensions>=4.10.0 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from torch) (4.12.2)
Collecting sympy>=1.13.3 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Requirement already satisfied: networkx>=2.5.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from torch) (3.4.2)
Requirement already satisfied: jinja2 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from torch) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from torch) (2025.7.0)
Collecting nvidia-cuda-nvrtc-cu12==12.8.93 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-runtime-cu12==12.8.90 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-cupti-cu12==12.8.90 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cudnn-cu12==9.10.2.21 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cublas-cu12==12.8.4.1 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cufft-cu12==11.3.3.83 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-curand-cu12==10.3.9.90 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cusolver-cu12==11.7.3.90 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusparse-cu12==12.5.8.93 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from torch) (0.7.1)
Collecting nvidia-nccl-cu12==2.27.5 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.0 kB)
Requirement already satisfied: nvidia-nvshmem-cu12==3.3.20 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from torch) (3.3.20)
Collecting nvidia-nvtx-cu12==12.8.90 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nvjitlink-cu12==12.8.93 (from torch)
  Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from torch) (1.13.1.3)
Requirement already satisfied: pytorch-triton==3.4.0+gitf7888497 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from torch) (3.4.0+gitf7888497)
Requirement already satisfied: setuptools>=40.8.0 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from pytorch-triton==3.4.0+gitf7888497->torch) (78.1.1)
Requirement already satisfied: numpy in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from torchvision) (2.0.1)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from torchvision) (9.5.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from sympy>=1.13.3->torch) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from jinja2->torch) (3.0.2)
Downloading https://download.pytorch.org/whl/nightly/cu128/torch-2.9.0.dev20250902%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl (900.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 900.4/900.4 MB 58.7 MB/s  0:00:10
Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl (594.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 594.3/594.3 MB 70.8 MB/s  0:00:06
Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (10.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.2/10.2 MB 85.9 MB/s  0:00:00
Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (88.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88.0/88.0 MB 90.0 MB/s  0:00:00
Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (954 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 954.8/954.8 kB 48.8 MB/s  0:00:00
Downloading https://download.pytorch.org/whl/nightly/nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl (706.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 706.8/706.8 MB 64.6 MB/s  0:00:08
Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (193.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 193.1/193.1 MB 91.2 MB/s  0:00:02
Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl (63.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.6/63.6 MB 68.4 MB/s  0:00:00
Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl (267.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 267.5/267.5 MB 86.1 MB/s  0:00:03
Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (288.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 288.2/288.2 MB 85.5 MB/s  0:00:03
Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (322.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 322.3/322.3 MB 83.9 MB/s  0:00:03
Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (39.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 39.3/39.3 MB 88.7 MB/s  0:00:00
Downloading https://download.pytorch.org/whl/nightly/cu128/nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89 kB)
Downloading https://download.pytorch.org/whl/nightly/cu128/torchvision-0.24.0.dev20250902%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl (8.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.2/8.2 MB 86.9 MB/s  0:00:00
Downloading https://download.pytorch.org/whl/nightly/cu128/torchaudio-2.8.0.dev20250902%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl (2.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 67.8 MB/s  0:00:00
Downloading https://download.pytorch.org/whl/nightly/sympy-1.14.0-py3-none-any.whl (6.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 89.8 MB/s  0:00:00
Installing collected packages: sympy, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, torch, torchvision, torchaudio
  Attempting uninstall: sympy
    Found existing installation: sympy 1.13.1
    Uninstalling sympy-1.13.1:
      Successfully uninstalled sympy-1.13.1
  Attempting uninstall: nvidia-nvtx-cu12
    Found existing installation: nvidia-nvtx-cu12 12.4.127
    Uninstalling nvidia-nvtx-cu12-12.4.127:
      Successfully uninstalled nvidia-nvtx-cu12-12.4.127
  Attempting uninstall: nvidia-nvjitlink-cu12
    Found existing installation: nvidia-nvjitlink-cu12 12.4.127
    Uninstalling nvidia-nvjitlink-cu12-12.4.127:
      Successfully uninstalled nvidia-nvjitlink-cu12-12.4.127
  Attempting uninstall: nvidia-nccl-cu12
    Found existing installation: nvidia-nccl-cu12 2.21.5
    Uninstalling nvidia-nccl-cu12-2.21.5:
      Successfully uninstalled nvidia-nccl-cu12-2.21.5
  Attempting uninstall: nvidia-curand-cu12
    Found existing installation: nvidia-curand-cu12 10.3.5.147
    Uninstalling nvidia-curand-cu12-10.3.5.147:
      Successfully uninstalled nvidia-curand-cu12-10.3.5.147
  Attempting uninstall: nvidia-cuda-runtime-cu12
    Found existing installation: nvidia-cuda-runtime-cu12 12.4.127
    Uninstalling nvidia-cuda-runtime-cu12-12.4.127:
      Successfully uninstalled nvidia-cuda-runtime-cu12-12.4.127
  Attempting uninstall: nvidia-cuda-nvrtc-cu12
    Found existing installation: nvidia-cuda-nvrtc-cu12 12.4.127
    Uninstalling nvidia-cuda-nvrtc-cu12-12.4.127:
      Successfully uninstalled nvidia-cuda-nvrtc-cu12-12.4.127
  Attempting uninstall: nvidia-cuda-cupti-cu12
    Found existing installation: nvidia-cuda-cupti-cu12 12.4.127
    Uninstalling nvidia-cuda-cupti-cu12-12.4.127:
      Successfully uninstalled nvidia-cuda-cupti-cu12-12.4.127
  Attempting uninstall: nvidia-cublas-cu12
    Found existing installation: nvidia-cublas-cu12 12.4.5.8
    Uninstalling nvidia-cublas-cu12-12.4.5.8:
      Successfully uninstalled nvidia-cublas-cu12-12.4.5.8
  Attempting uninstall: nvidia-cusparse-cu12
    Found existing installation: nvidia-cusparse-cu12 12.3.1.170
    Uninstalling nvidia-cusparse-cu12-12.3.1.170:
      Successfully uninstalled nvidia-cusparse-cu12-12.3.1.170
  Attempting uninstall: nvidia-cufft-cu12
    Found existing installation: nvidia-cufft-cu12 11.2.1.3
    Uninstalling nvidia-cufft-cu12-11.2.1.3:
      Successfully uninstalled nvidia-cufft-cu12-11.2.1.3
  Attempting uninstall: nvidia-cudnn-cu12
    Found existing installation: nvidia-cudnn-cu12 9.1.0.70
    Uninstalling nvidia-cudnn-cu12-9.1.0.70:
      Successfully uninstalled nvidia-cudnn-cu12-9.1.0.70
  Attempting uninstall: nvidia-cusolver-cu12
    Found existing installation: nvidia-cusolver-cu12 11.6.1.9
    Uninstalling nvidia-cusolver-cu12-11.6.1.9:
      Successfully uninstalled nvidia-cusolver-cu12-11.6.1.9
Successfully installed nvidia-cublas-cu12-12.8.4.1 nvidia-cuda-cupti-cu12-12.8.90 nvidia-cuda-nvrtc-cu12-12.8.93 nvidia-cuda-runtime-cu12-12.8.90 nvidia-cudnn-cu12-9.10.2.21 nvidia-cufft-cu12-11.3.3.83 nvidia-curand-cu12-10.3.9.90 nvidia-cusolver-cu12-11.7.3.90 nvidia-cusparse-cu12-12.5.8.93 nvidia-nccl-cu12-2.27.5 nvidia-nvjitlink-cu12-12.8.93 nvidia-nvtx-cu12-12.8.90 sympy-1.14.0 torch-2.9.0.dev20250902+cu128 torchaudio-2.8.0.dev20250902+cu128 torchvision-0.24.0.dev20250902+cu128
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('CC:', torch.cuda.get_device_capability(0))
x = torch.randn(1, device='cuda'); print('CUDA OK, tensor:', x.device)
PY
2.9.0.dev20250902+cu128 12.8
GPU: NVIDIA GeForce RTX 5060 Ti
CC: (12, 0)
CUDA OK, tensor: cuda:0
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ pip uninstall -y detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
Found existing installation: detectron2 0.6
Uninstalling detectron2-0.6:
  Successfully uninstalled detectron2-0.6
Collecting git+https://github.com/facebookresearch/detectron2.git
  Cloning https://github.com/facebookresearch/detectron2.git to /tmp/pip-req-build-01en43bi
  Running command git clone --filter=blob:none --quiet https://github.com/facebookresearch/detectron2.git /tmp/pip-req-build-01en43bi
  Resolved https://github.com/facebookresearch/detectron2.git to commit a1ce2f956a1d2212ad672e3c47d53405c2fe4312
  Preparing metadata (setup.py) ... done
Requirement already satisfied: Pillow>=7.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (9.5.0)
Requirement already satisfied: matplotlib in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (3.10.6)
Requirement already satisfied: pycocotools>=2.0.2 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (2.0.10)
Requirement already satisfied: termcolor>=1.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (3.1.0)
Requirement already satisfied: yacs>=0.1.8 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (0.1.8)
Requirement already satisfied: tabulate in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (0.9.0)
Requirement already satisfied: cloudpickle in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (3.1.1)
Requirement already satisfied: tqdm>4.29.0 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (4.67.1)
Requirement already satisfied: tensorboard in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (2.20.0)
Requirement already satisfied: fvcore<0.1.6,>=0.1.5 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (0.1.5.post20221221)
Requirement already satisfied: iopath<0.1.10,>=0.1.7 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (0.1.9)
Requirement already satisfied: omegaconf<2.4,>=2.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (2.3.0)
Requirement already satisfied: hydra-core>=1.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (1.3.2)
Requirement already satisfied: black in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (21.4b2)
Requirement already satisfied: packaging in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from detectron2==0.6) (25.0)
Requirement already satisfied: numpy in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from fvcore<0.1.6,>=0.1.5->detectron2==0.6) (2.0.1)
Requirement already satisfied: pyyaml>=5.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from fvcore<0.1.6,>=0.1.5->detectron2==0.6) (6.0.2)
Requirement already satisfied: portalocker in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from iopath<0.1.10,>=0.1.7->detectron2==0.6) (3.2.0)
Requirement already satisfied: antlr4-python3-runtime==4.9.* in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from omegaconf<2.4,>=2.1->detectron2==0.6) (4.9.3)
Requirement already satisfied: click>=7.1.2 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from black->detectron2==0.6) (8.2.1)
Requirement already satisfied: appdirs in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from black->detectron2==0.6) (1.4.4)
Requirement already satisfied: toml>=0.10.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from black->detectron2==0.6) (0.10.2)
Requirement already satisfied: regex>=2020.1.8 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from black->detectron2==0.6) (2025.8.29)
Requirement already satisfied: pathspec<1,>=0.8.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from black->detectron2==0.6) (0.12.1)
Requirement already satisfied: mypy-extensions>=0.4.3 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from black->detectron2==0.6) (1.1.0)
Requirement already satisfied: contourpy>=1.0.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from matplotlib->detectron2==0.6) (1.3.2)
Requirement already satisfied: cycler>=0.10 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from matplotlib->detectron2==0.6) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from matplotlib->detectron2==0.6) (4.59.2)
Requirement already satisfied: kiwisolver>=1.3.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from matplotlib->detectron2==0.6) (1.4.9)
Requirement already satisfied: pyparsing>=2.3.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from matplotlib->detectron2==0.6) (3.2.3)
Requirement already satisfied: python-dateutil>=2.7 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from matplotlib->detectron2==0.6) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib->detectron2==0.6) (1.17.0)
Requirement already satisfied: absl-py>=0.4 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from tensorboard->detectron2==0.6) (2.3.1)
Requirement already satisfied: grpcio>=1.48.2 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from tensorboard->detectron2==0.6) (1.74.0)
Requirement already satisfied: markdown>=2.6.8 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from tensorboard->detectron2==0.6) (3.8.2)
Requirement already satisfied: protobuf!=4.24.0,>=3.19.6 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from tensorboard->detectron2==0.6) (6.32.0)
Requirement already satisfied: setuptools>=41.0.0 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from tensorboard->detectron2==0.6) (78.1.1)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from tensorboard->detectron2==0.6) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from tensorboard->detectron2==0.6) (3.1.3)
Requirement already satisfied: MarkupSafe>=2.1.1 in /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard->detectron2==0.6) (3.0.2)
Building wheels for collected packages: detectron2
  DEPRECATION: Building 'detectron2' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of 'detectron2'. Discussion can be found at https://github.com/pypa/pip/issues/6334
  Building wheel for detectron2 (setup.py) ... done
  Created wheel for detectron2: filename=detectron2-0.6-cp310-cp310-linux_x86_64.whl size=1267811 sha256=81979c1349b8d3f0e7e6d87823e739050a8a91216133ff3b34562aad044ac532
  Stored in directory: /tmp/pip-ephem-wheel-cache-ebcaducu/wheels/47/e5/15/94c80df2ba85500c5d76599cc307c0a7079d0e221bb6fc4375
Successfully built detectron2
Installing collected packages: detectron2
Successfully installed detectron2-0.6
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ cd ~/Mask2Former/mask2former/modeling/pixel_decoder/ops
rm -rf build *.so
TORCH_CUDA_ARCH_LIST="12.0" python setup.py build install
cd ~/Mask2Former
running build
running build_py
creating build/lib.linux-x86_64-cpython-310/functions
copying functions/__init__.py -> build/lib.linux-x86_64-cpython-310/functions
copying functions/ms_deform_attn_func.py -> build/lib.linux-x86_64-cpython-310/functions
creating build/lib.linux-x86_64-cpython-310/modules
copying modules/__init__.py -> build/lib.linux-x86_64-cpython-310/modules
copying modules/ms_deform_attn.py -> build/lib.linux-x86_64-cpython-310/modules
running build_ext
W0903 14:47:58.077000 1537 site-packages/torch/utils/cpp_extension.py:533] There are no g++ version bounds defined for CUDA version 12.8
building 'MultiScaleDeformableAttention' extension
creating /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cpu
creating /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda
[1/3] c++ -MMD -MF /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cpu/ms_deform_attn_cpu.o.d -pthread -B /home/nina/miniconda3/envs/mask2former/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/nina/miniconda3/envs/mask2former/include -fPIC -O2 -isystem /home/nina/miniconda3/envs/mask2former/include -fPIC -DWITH_CUDA -I/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/nina/miniconda3/envs/mask2former/include/python3.10 -c -c /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cpu/ms_deform_attn_cpu.cpp -o /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cpu/ms_deform_attn_cpu.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=MultiScaleDeformableAttention -std=c++17
[2/3] /usr/local/cuda-12.8/bin/nvcc --generate-dependencies-with-compile --dependency-output /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.o.d -DWITH_CUDA -I/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/nina/miniconda3/envs/mask2former/include/python3.10 -c -c /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu -o /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=MultiScaleDeformableAttention -gencode=arch=compute_120,code=sm_120 -std=c++17
FAILED: /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.o
/usr/local/cuda-12.8/bin/nvcc --generate-dependencies-with-compile --dependency-output /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.o.d -DWITH_CUDA -I/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/nina/miniconda3/envs/mask2former/include/python3.10 -c -c /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu -o /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=MultiScaleDeformableAttention -gencode=arch=compute_120,code=sm_120 -std=c++17
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu(69): error: no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType" exists
          [&] { const auto& the_type = value.type(); constexpr const char* at_dispatch_name = "ms_deform_attn_forward_cuda"; at::ScalarType _st = ::detail::scalar_type(the_type); ; switch (_st) { case at::ScalarType::Double: { do { if constexpr (!at::should_include_kernel_dtype( at_dispatch_name, at::ScalarType::Double)) { if (!(false)) { ::c10::detail::torchCheckFail( __func__, "/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu", static_cast<uint32_t>(69), (::c10::detail::torchCheckMsgImpl( "Expected " "false" " to be true, but got false.  " "(Could this error message be improved?  If so, " "please report an enhancement request to PyTorch.)", "dtype '", toString(at::ScalarType::Double), "' not selected for kernel tag ", at_dispatch_name))); }; } } while (0); using scalar_t [[maybe_unused]] = c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Double>; return ([&] { ms_deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(), value.data<scalar_t>() + n * im2col_step_ * per_value_size, spatial_shapes.data<int64_t>(), level_start_index.data<int64_t>(), sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size, attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size, batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point, columns.data<scalar_t>()); })(); } case at::ScalarType::Float: { do { if constexpr (!at::should_include_kernel_dtype( at_dispatch_name, at::ScalarType::Float)) { if (!(false)) { ::c10::detail::torchCheckFail( __func__, "/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu", static_cast<uint32_t>(69), (::c10::detail::torchCheckMsgImpl( "Expected " "false" " to be true, but got false.  " "(Could this error message be improved?  If so, " "please report an enhancement request to PyTorch.)", "dtype '", toString(at::ScalarType::Float), "' not selected for kernel tag ", at_dispatch_name))); }; } } while (0); using scalar_t [[maybe_unused]] = c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Float>; return ([&] { ms_deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(), value.data<scalar_t>() + n * im2col_step_ * per_value_size, spatial_shapes.data<int64_t>(), level_start_index.data<int64_t>(), sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size, attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size, batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point, columns.data<scalar_t>()); })(); } default: if (!(false)) { throw ::c10::NotImplementedError( {__func__, "/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu", static_cast<uint32_t>(69)}, (::c10::detail::torchCheckMsgImpl( "Expected " "false" " to be true, but got false.  " "(Could this error message be improved?  If so, " "please report an enhancement request to PyTorch.)", '"', at_dispatch_name, "\" not implemented for '", toString(_st), "'"))); }; } }()
                                                                                                                                                                        ^

/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu(139): error: no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType" exists
          [&] { const auto& the_type = value.type(); constexpr const char* at_dispatch_name = "ms_deform_attn_backward_cuda"; at::ScalarType _st = ::detail::scalar_type(the_type); ; switch (_st) { case at::ScalarType::Double: { do { if constexpr (!at::should_include_kernel_dtype( at_dispatch_name, at::ScalarType::Double)) { if (!(false)) { ::c10::detail::torchCheckFail( __func__, "/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu", static_cast<uint32_t>(139), (::c10::detail::torchCheckMsgImpl( "Expected " "false" " to be true, but got false.  " "(Could this error message be improved?  If so, " "please report an enhancement request to PyTorch.)", "dtype '", toString(at::ScalarType::Double), "' not selected for kernel tag ", at_dispatch_name))); }; } } while (0); using scalar_t [[maybe_unused]] = c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Double>; return ([&] { ms_deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(), grad_output_g.data<scalar_t>(), value.data<scalar_t>() + n * im2col_step_ * per_value_size, spatial_shapes.data<int64_t>(), level_start_index.data<int64_t>(), sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size, attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size, batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point, grad_value.data<scalar_t>() + n * im2col_step_ * per_value_size, grad_sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size, grad_attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size); })(); } case at::ScalarType::Float: { do { if constexpr (!at::should_include_kernel_dtype( at_dispatch_name, at::ScalarType::Float)) { if (!(false)) { ::c10::detail::torchCheckFail( __func__, "/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu", static_cast<uint32_t>(139), (::c10::detail::torchCheckMsgImpl( "Expected " "false" " to be true, but got false.  " "(Could this error message be improved?  If so, " "please report an enhancement request to PyTorch.)", "dtype '", toString(at::ScalarType::Float), "' not selected for kernel tag ", at_dispatch_name))); }; } } while (0); using scalar_t [[maybe_unused]] = c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Float>; return ([&] { ms_deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(), grad_output_g.data<scalar_t>(), value.data<scalar_t>() + n * im2col_step_ * per_value_size, spatial_shapes.data<int64_t>(), level_start_index.data<int64_t>(), sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size, attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size, batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point, grad_value.data<scalar_t>() + n * im2col_step_ * per_value_size, grad_sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size, grad_attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size); })(); } default: if (!(false)) { throw ::c10::NotImplementedError( {__func__, "/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu", static_cast<uint32_t>(139)}, (::c10::detail::torchCheckMsgImpl( "Expected " "false" " to be true, but got false.  " "(Could this error message be improved?  If so, " "please report an enhancement request to PyTorch.)", '"', at_dispatch_name, "\" not implemented for '", toString(_st), "'"))); }; } }()
                                                                                                                                                                         ^

2 errors detected in the compilation of "/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu".
[3/3] c++ -MMD -MF /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/vision.o.d -pthread -B /home/nina/miniconda3/envs/mask2former/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/nina/miniconda3/envs/mask2former/include -fPIC -O2 -isystem /home/nina/miniconda3/envs/mask2former/include -fPIC -DWITH_CUDA -I/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/nina/miniconda3/envs/mask2former/include/python3.10 -c -c /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/vision.cpp -o /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/vision.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=MultiScaleDeformableAttention -std=c++17
In file included from /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/vision.cpp:16:
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/ms_deform_attn.h: In function ‘at::Tensor ms_deform_attn_forward(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int)’:
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/ms_deform_attn.h:34:19: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   34 |     if (value.type().is_cuda())
      |         ~~~~~~~~~~^~
In file included from /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/Tensor.h:3,
                 from /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/Tensor.h:3,
                 from /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/function_hook.h:3,
                 from /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/cpp_hook.h:2,
                 from /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/variable.h:6,
                 from /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/autograd.h:3,
                 from /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/api/include/torch/autograd.h:3,
                 from /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/api/include/torch/all.h:7,
                 from /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/extension.h:5,
                 from /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cpu/ms_deform_attn_cpu.h:17,
                 from /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/ms_deform_attn.h:18:
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:30: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      |                              ^~~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/ms_deform_attn.h: In function ‘std::vector<at::Tensor> ms_deform_attn_backward(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int)’:
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/ms_deform_attn.h:56:19: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   56 |     if (value.type().is_cuda())
      |         ~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:30: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      |                              ^~~~
ninja: build stopped: subcommand failed.
Traceback (most recent call last):
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 2591, in _run_ninja_build
    subprocess.run(
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/setup.py", line 69, in <module>
    setup(
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/__init__.py", line 117, in setup
    return distutils.core.setup(**attrs)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 186, in setup
    return run_commands(dist)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 202, in run_commands
    dist.run_commands()
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1002, in run_commands
    self.run_command(cmd)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/dist.py", line 1104, in run_command
    super().run_command(command)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
    cmd_obj.run()
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/command/build.py", line 135, in run
    self.run_command(cmd_name)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
    self.distribution.run_command(command)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/dist.py", line 1104, in run_command
    super().run_command(command)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
    cmd_obj.run()
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/command/build_ext.py", line 99, in run
    _build_ext.run(self)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 368, in run
    self.build_extensions()
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1084, in build_extensions
    build_ext.build_extensions(self)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 484, in build_extensions
    self._build_extensions_serial()
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 510, in _build_extensions_serial
    self.build_extension(ext)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/command/build_ext.py", line 264, in build_extension
    _build_ext.build_extension(self, ext)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/Cython/Distutils/build_ext.py", line 136, in build_extension
    super().build_extension(ext)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 565, in build_extension
    objects = self.compiler.compile(
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 868, in unix_wrap_ninja_compile
    _write_ninja_file_and_compile_objects(
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 2222, in _write_ninja_file_and_compile_objects
    _run_ninja_build(
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 2608, in _run_ninja_build
    raise RuntimeError(message) from e
RuntimeError: Error compiling objects for extension
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ cd ~/Mask2Former/mask2former/modeling/pixel_decoder/ops

# 把 .cu 檔內的 dispatch 從 value.type() 改成 value.scalar_type()
sed -i 's/AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.type()/AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type()/g' src/cuda/ms_deform_attn_cuda.cu
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' src/cuda/ms_deform_attn_cuda.cu

# 把 .h 檔內的 CUDA 判斷從 value.type().is_cuda() 改為 value.is_cuda()
sed -i 's/value.type().is_cuda()/value.is_cuda()/g' src/ms_deform_attn.h
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former/mask2former/modeling/pixel_decoder/ops$ rm -rf build *.so
TORCH_CUDA_ARCH_LIST="12.0" python setup.py build install
running build
running build_py
creating build/lib.linux-x86_64-cpython-310/functions
copying functions/__init__.py -> build/lib.linux-x86_64-cpython-310/functions
copying functions/ms_deform_attn_func.py -> build/lib.linux-x86_64-cpython-310/functions
creating build/lib.linux-x86_64-cpython-310/modules
copying modules/__init__.py -> build/lib.linux-x86_64-cpython-310/modules
copying modules/ms_deform_attn.py -> build/lib.linux-x86_64-cpython-310/modules
running build_ext
W0903 14:50:29.251000 1571 site-packages/torch/utils/cpp_extension.py:533] There are no g++ version bounds defined for CUDA version 12.8
building 'MultiScaleDeformableAttention' extension
creating /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cpu
creating /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda
[1/3] c++ -MMD -MF /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cpu/ms_deform_attn_cpu.o.d -pthread -B /home/nina/miniconda3/envs/mask2former/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/nina/miniconda3/envs/mask2former/include -fPIC -O2 -isystem /home/nina/miniconda3/envs/mask2former/include -fPIC -DWITH_CUDA -I/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/nina/miniconda3/envs/mask2former/include/python3.10 -c -c /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cpu/ms_deform_attn_cpu.cpp -o /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cpu/ms_deform_attn_cpu.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=MultiScaleDeformableAttention -std=c++17
[2/3] c++ -MMD -MF /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/vision.o.d -pthread -B /home/nina/miniconda3/envs/mask2former/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/nina/miniconda3/envs/mask2former/include -fPIC -O2 -isystem /home/nina/miniconda3/envs/mask2former/include -fPIC -DWITH_CUDA -I/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/nina/miniconda3/envs/mask2former/include/python3.10 -c -c /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/vision.cpp -o /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/vision.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=MultiScaleDeformableAttention -std=c++17
[3/3] /usr/local/cuda-12.8/bin/nvcc --generate-dependencies-with-compile --dependency-output /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.o.d -DWITH_CUDA -I/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include -I/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/nina/miniconda3/envs/mask2former/include/python3.10 -c -c /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu -o /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=MultiScaleDeformableAttention -gencode=arch=compute_120,code=sm_120 -std=c++17
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_im2col_cuda.cuh(266): warning #177-D: variable "q_col" was declared but never referenced
      const int q_col = _temp % num_query;
                ^
          detected during instantiation of "void ms_deformable_im2col_cuda(cudaStream_t, const scalar_t *, const int64_t *, const int64_t *, const scalar_t *, const scalar_t *, int, int, int, int, int, int, int, scalar_t *) [with scalar_t=double]" at line 69 of /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_im2col_cuda.cuh(767): warning #177-D: variable "q_col" was declared but never referenced
      const int q_col = _temp % num_query;
                ^
          detected during instantiation of "void ms_deformable_col2im_cuda(cudaStream_t, const scalar_t *, const scalar_t *, const int64_t *, const int64_t *, const scalar_t *, const scalar_t *, int, int, int, int, int, int, int, scalar_t *, scalar_t *, scalar_t *) [with scalar_t=double]" at line 139 of /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu

/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_im2col_cuda.cuh(877): warning #177-D: variable "q_col" was declared but never referenced
      const int q_col = _temp % num_query;
                ^
          detected during instantiation of "void ms_deformable_col2im_cuda(cudaStream_t, const scalar_t *, const scalar_t *, const int64_t *, const int64_t *, const scalar_t *, const scalar_t *, int, int, int, int, int, int, int, scalar_t *, scalar_t *, scalar_t *) [with scalar_t=double]" at line 139 of /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu

/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_im2col_cuda.cuh(336): warning #177-D: variable "q_col" was declared but never referenced
      const int q_col = _temp % num_query;
                ^
          detected during instantiation of "void ms_deformable_col2im_cuda(cudaStream_t, const scalar_t *, const scalar_t *, const int64_t *, const int64_t *, const scalar_t *, const scalar_t *, int, int, int, int, int, int, int, scalar_t *, scalar_t *, scalar_t *) [with scalar_t=double]" at line 139 of /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu

/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_im2col_cuda.cuh(441): warning #177-D: variable "q_col" was declared but never referenced
      const int q_col = _temp % num_query;
                ^
          detected during instantiation of "void ms_deformable_col2im_cuda(cudaStream_t, const scalar_t *, const scalar_t *, const int64_t *, const int64_t *, const scalar_t *, const scalar_t *, int, int, int, int, int, int, int, scalar_t *, scalar_t *, scalar_t *) [with scalar_t=double]" at line 139 of /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu

/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_im2col_cuda.cuh(549): warning #177-D: variable "q_col" was declared but never referenced
      const int q_col = _temp % num_query;
                ^
          detected during instantiation of "void ms_deformable_col2im_cuda(cudaStream_t, const scalar_t *, const scalar_t *, const int64_t *, const int64_t *, const scalar_t *, const scalar_t *, int, int, int, int, int, int, int, scalar_t *, scalar_t *, scalar_t *) [with scalar_t=double]" at line 139 of /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu

/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_im2col_cuda.cuh(654): warning #177-D: variable "q_col" was declared but never referenced
      const int q_col = _temp % num_query;
                ^
          detected during instantiation of "void ms_deformable_col2im_cuda(cudaStream_t, const scalar_t *, const scalar_t *, const int64_t *, const int64_t *, const scalar_t *, const scalar_t *, int, int, int, int, int, int, int, scalar_t *, scalar_t *, scalar_t *) [with scalar_t=double]" at line 139 of /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu

/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu: In function ‘at::Tensor ms_deform_attn_cuda_forward(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int)’:
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:39:61: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   39 |     AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
      |                                                   ~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:1: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:40:70: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   40 |     AT_ASSERTM(spatial_shapes.type().is_cuda(), "spatial_shapes must be a CUDA tensor");
      |                                                   ~~~~~~~~~~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:1: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:41:73: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   41 |     AT_ASSERTM(level_start_index.type().is_cuda(), "level_start_index must be a CUDA tensor");
      |                                                   ~~~~~~~~~~~~~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:1: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:42:68: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   42 |     AT_ASSERTM(sampling_loc.type().is_cuda(), "sampling_loc must be a CUDA tensor");
      |                                                   ~~~~~~~~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:1: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:43:67: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   43 |     AT_ASSERTM(attn_weight.type().is_cuda(), "attn_weight must be a CUDA tensor");
      |                                                   ~~~~~~~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:1: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu: In lambda function:
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:998: warning: ‘T* at::Tensor::data() const [with T = double]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:1084: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:1127: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:1160: warning: ‘T* at::Tensor::data() const [with T = double]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:1243: warning: ‘T* at::Tensor::data() const [with T = double]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:1401: warning: ‘T* at::Tensor::data() const [with T = double]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu: In lambda function:
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:2209: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:2295: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:2338: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:2370: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:2452: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:69:2609: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
   69 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu: In function ‘std::vector<at::Tensor> ms_deform_attn_cuda_backward(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int)’:
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:105:61: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
  105 |     AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
      |                                                   ~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:1: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:106:70: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
  106 |     AT_ASSERTM(spatial_shapes.type().is_cuda(), "spatial_shapes must be a CUDA tensor");
      |                                                   ~~~~~~~~~~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:1: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:107:73: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
  107 |     AT_ASSERTM(level_start_index.type().is_cuda(), "level_start_index must be a CUDA tensor");
      |                                                   ~~~~~~~~~~~~~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:1: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:108:68: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
  108 |     AT_ASSERTM(sampling_loc.type().is_cuda(), "sampling_loc must be a CUDA tensor");
      |                                                   ~~~~~~~~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:1: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:109:67: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
  109 |     AT_ASSERTM(attn_weight.type().is_cuda(), "attn_weight must be a CUDA tensor");
      |                                                   ~~~~~~~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:1: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:110:67: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
  110 |     AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");
      |                                                   ~~~~~~~~~~~~~~~~^~
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:225:1: note: declared here
  225 |   DeprecatedTypeProperties & type() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu: In lambda function:
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:1008: warning: ‘T* at::Tensor::data() const [with T = double]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:1034: warning: ‘T* at::Tensor::data() const [with T = double]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:1120: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:1163: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:1196: warning: ‘T* at::Tensor::data() const [with T = double]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:1279: warning: ‘T* at::Tensor::data() const [with T = double]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:1440: warning: ‘T* at::Tensor::data() const [with T = double]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:1524: warning: ‘T* at::Tensor::data() const [with T = double]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:1612: warning: ‘T* at::Tensor::data() const [with T = double]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu: In lambda function:
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:2481: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:2506: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:2592: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:2635: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:2667: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:2749: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:2909: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:2992: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu:139:3079: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
  139 |         AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       ^
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:247:1: note: declared here
  247 |   T * data() const {
      | ^ ~~
g++ -pthread -B /home/nina/miniconda3/envs/mask2former/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/nina/miniconda3/envs/mask2former/include -fPIC -O2 -isystem /home/nina/miniconda3/envs/mask2former/include -pthread -B /home/nina/miniconda3/envs/mask2former/compiler_compat -shared /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cpu/ms_deform_attn_cpu.o /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.o /home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/build/temp.linux-x86_64-cpython-310/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/vision.o -L/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/lib -L/usr/local/cuda-12.8/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda -o build/lib.linux-x86_64-cpython-310/MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so
running install
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/cmd.py:90: SetuptoolsDeprecationWarning: setup.py install is deprecated.
!!

        ********************************************************************************
        Please avoid running ``setup.py`` directly.
        Instead, use pypa/build, pypa/installer or other
        standards-based tools.

        See https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html for details.
        ********************************************************************************

!!
  self.initialize_options()
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/setuptools/_distutils/cmd.py:90: EasyInstallDeprecationWarning: easy_install command is deprecated.
!!

        ********************************************************************************
        Please avoid running ``setup.py`` and ``easy_install``.
        Instead, use pypa/build, pypa/installer or other
        standards-based tools.

        See https://github.com/pypa/setuptools/issues/917 for details.
        ********************************************************************************

!!
  self.initialize_options()
running bdist_egg
running egg_info
creating MultiScaleDeformableAttention.egg-info
writing MultiScaleDeformableAttention.egg-info/PKG-INFO
writing dependency_links to MultiScaleDeformableAttention.egg-info/dependency_links.txt
writing top-level names to MultiScaleDeformableAttention.egg-info/top_level.txt
writing manifest file 'MultiScaleDeformableAttention.egg-info/SOURCES.txt'
reading manifest file 'MultiScaleDeformableAttention.egg-info/SOURCES.txt'
writing manifest file 'MultiScaleDeformableAttention.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
creating build/bdist.linux-x86_64/egg
creating build/bdist.linux-x86_64/egg/functions
copying build/lib.linux-x86_64-cpython-310/functions/__init__.py -> build/bdist.linux-x86_64/egg/functions
copying build/lib.linux-x86_64-cpython-310/functions/ms_deform_attn_func.py -> build/bdist.linux-x86_64/egg/functions
creating build/bdist.linux-x86_64/egg/modules
copying build/lib.linux-x86_64-cpython-310/modules/__init__.py -> build/bdist.linux-x86_64/egg/modules
copying build/lib.linux-x86_64-cpython-310/modules/ms_deform_attn.py -> build/bdist.linux-x86_64/egg/modules
copying build/lib.linux-x86_64-cpython-310/MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg
byte-compiling build/bdist.linux-x86_64/egg/functions/__init__.py to __init__.cpython-310.pyc
byte-compiling build/bdist.linux-x86_64/egg/functions/ms_deform_attn_func.py to ms_deform_attn_func.cpython-310.pyc
byte-compiling build/bdist.linux-x86_64/egg/modules/__init__.py to __init__.cpython-310.pyc
byte-compiling build/bdist.linux-x86_64/egg/modules/ms_deform_attn.py to ms_deform_attn.cpython-310.pyc
creating stub loader for MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so
byte-compiling build/bdist.linux-x86_64/egg/MultiScaleDeformableAttention.py to MultiScaleDeformableAttention.cpython-310.pyc
creating build/bdist.linux-x86_64/egg/EGG-INFO
copying MultiScaleDeformableAttention.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
copying MultiScaleDeformableAttention.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying MultiScaleDeformableAttention.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying MultiScaleDeformableAttention.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
writing build/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
zip_safe flag not set; analyzing archive contents...
__pycache__.MultiScaleDeformableAttention.cpython-310: module references __file__
creating dist
creating 'dist/MultiScaleDeformableAttention-1.0-py3.10-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing MultiScaleDeformableAttention-1.0-py3.10-linux-x86_64.egg
creating /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/MultiScaleDeformableAttention-1.0-py3.10-linux-x86_64.egg
Extracting MultiScaleDeformableAttention-1.0-py3.10-linux-x86_64.egg to /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages
Adding MultiScaleDeformableAttention 1.0 to easy-install.pth file

Installed /home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/MultiScaleDeformableAttention-1.0-py3.10-linux-x86_64.egg
Processing dependencies for MultiScaleDeformableAttention==1.0
Finished processing dependencies for MultiScaleDeformableAttention==1.0
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former/mask2former/modeling/pixel_decoder/ops$ cd ~/Mask2Former

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
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ImportError: libc10.so: cannot open shared object file: No such file or directory
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/home/nina/Mask2Former/demo/../mask2former/modeling/pixel_decoder/msdeformattn.py:314: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
[09/03 14:50:55 detectron2]: Arguments: Namespace(config_file='configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml', webcam=False, video_input=None, input=['demo_inputs/input.jpg'], output='demo_inputs/output_panoptic_gpu.png', confidence_threshold=0.5, opts=['MODEL.DEVICE', 'cuda'])
[09/03 14:50:56 d2.checkpoint.detection_checkpoint]: [DetectionCheckpointer] Loading from detectron2://ImageNetPretrained/torchvision/R-50.pkl ...
[09/03 14:50:56 fvcore.common.checkpoint]: [Checkpointer] Loading from /home/nina/.torch/iopath_cache/detectron2/ImageNetPretrained/torchvision/R-50.pkl ...
[09/03 14:50:56 fvcore.common.checkpoint]: Reading a file from 'torchvision'
[09/03 14:50:56 d2.checkpoint.c2_model_loading]: Following weights matched with submodule backbone - Total num: 53
WARNING [09/03 14:50:56 fvcore.common.checkpoint]: Some model parameters or buffers are not found in the checkpoint:
criterion.empty_weight
sem_seg_head.pixel_decoder.adapter_1.norm.{bias, weight}
sem_seg_head.pixel_decoder.adapter_1.weight
sem_seg_head.pixel_decoder.input_proj.0.0.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.0.1.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.1.0.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.1.1.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.2.0.{bias, weight}
sem_seg_head.pixel_decoder.input_proj.2.1.{bias, weight}
sem_seg_head.pixel_decoder.layer_1.norm.{bias, weight}
sem_seg_head.pixel_decoder.layer_1.weight
sem_seg_head.pixel_decoder.mask_features.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm1.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm2.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.attention_weights.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.output_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.sampling_offsets.{bias, weight}
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.value_proj.{bias, weight}
sem_seg_head.pixel_decoder.transformer.level_embed
sem_seg_head.predictor.class_embed.{bias, weight}
sem_seg_head.predictor.decoder_norm.{bias, weight}
sem_seg_head.predictor.level_embed.weight
sem_seg_head.predictor.mask_embed.layers.0.{bias, weight}
sem_seg_head.predictor.mask_embed.layers.1.{bias, weight}
sem_seg_head.predictor.mask_embed.layers.2.{bias, weight}
sem_seg_head.predictor.query_embed.weight
sem_seg_head.predictor.query_feat.weight
sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.0.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.1.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.2.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.3.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.4.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.5.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.6.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.7.norm.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_cross_attention_layers.8.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.0.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.0.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.0.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.1.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.1.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.1.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.2.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.2.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.2.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.3.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.3.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.3.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.4.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.4.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.4.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.5.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.5.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.5.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.6.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.6.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.6.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.7.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.7.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.7.norm.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.8.linear1.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.8.linear2.{bias, weight}
sem_seg_head.predictor.transformer_ffn_layers.8.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.0.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.1.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.2.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.3.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.4.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.5.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.6.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.7.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn.{in_proj_bias, in_proj_weight}
sem_seg_head.predictor.transformer_self_attention_layers.8.norm.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn.out_proj.{bias, weight}
sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn.{in_proj_bias, in_proj_weight}
WARNING [09/03 14:50:56 fvcore.common.checkpoint]: The checkpoint state_dict contains keys that are not used by the model:
  stem.fc.{bias, weight}
  0%|                                                                                             | 0/1 [00:00<?, ?it/s]/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/functional.py:505: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:4317.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
[09/03 14:50:58 detectron2]: demo_inputs/input.jpg: detected 100 instances in 1.81s
100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.88s/it]
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ cp ~/Mask2Former/demo_inputs/output_panoptic_gpu.png /mnt/c/Users/USER/Desktop/output_panoptic_gpu.png
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ # 在 Mask2Former 專案根目錄
python demo/demo.py \
  --config-file configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
  --input demo_inputs/input.jpg \
  --output demo_inputs/output_panoptic_gpu.png \
  --opts \
    MODEL.DEVICE cuda \
    MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/home/nina/Mask2Former/demo/../mask2former/modeling/pixel_decoder/msdeformattn.py:314: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
[09/03 15:22:39 detectron2]: Arguments: Namespace(config_file='configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml', webcam=False, video_input=None, input=['demo_inputs/input.jpg'], output='demo_inputs/output_panoptic_gpu.png', confidence_threshold=0.5, opts=['MODEL.DEVICE', 'cuda', 'MODEL.WEIGHTS', 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'])
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/torch/functional.py:505: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:4317.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
[09/03 15:22:40 d2.checkpoint.detection_checkpoint]: [DetectionCheckpointer] Loading from https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl ...
model_final_f07440.pkl: 866MB [00:08, 101MB/s]
[09/03 15:22:49 fvcore.common.checkpoint]: [Checkpointer] Loading from /home/nina/.torch/iopath_cache/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl ...
[09/03 15:22:50 fvcore.common.checkpoint]: Reading a file from 'MaskFormer Model Zoo'
Weight format of MultiScaleMaskedTransformerDecoder have changed! Please upgrade your models. Applying automatic conversion now ...
  0%|                                                                                             | 0/1 [00:00<?, ?it/s][09/03 15:22:52 detectron2]: demo_inputs/input.jpg: detected 28 instances in 1.68s
100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.77s/it]
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ cp ~/Mask2Former/demo_inputs/output_panoptic_gpu.png /mnt/c/Users/USER/Desktop/output_panoptic_gpu.png
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ python script.py --mode train
python: can't open file '/home/nina/Mask2Former/script.py': [Errno 2] No such file or directory
(mask2former) nina@DESKTOP-QF7UBMR:~/Mask2Former$ cd /mnt/c/Users/USER/Desktop
(mask2former) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$ python script.py --mode train
/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/home/nina/Mask2Former/mask2former/modeling/pixel_decoder/msdeformattn.py:314: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
🚀 Starting Mask2Former training
   Backbone: Wide ResNet-50
   Parameters: improved
   Task: Semantic Segmentation
📊 Dataset split: 960 train, 240 val
Loading config /home/nina/local_cfgs/Base-ADE20K-SemanticSegmentation.yaml with yaml.unsafe_load. Your machine may be at risk if the file contains malicious content.
Traceback (most recent call last):
  File "/mnt/c/Users/USER/Desktop/script.py", line 624, in <module>
    main()
  File "/mnt/c/Users/USER/Desktop/script.py", line 609, in main
    cfg = train(use_custom_backbone=args.use_unet,
  File "/mnt/c/Users/USER/Desktop/script.py", line 445, in train
    cfg = setup_cfg(use_custom_backbone, improved_params)
  File "/mnt/c/Users/USER/Desktop/script.py", line 343, in setup_cfg
    cfg.merge_from_file("/home/nina/local_cfgs/maskformer2_R50_bs16_160k.yaml")
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/detectron2/config/config.py", line 69, in merge_from_file
    self.merge_from_other_cfg(loaded_cfg)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/fvcore/common/config.py", line 132, in merge_from_other_cfg
    return super().merge_from_other_cfg(cfg_other)
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/yacs/config.py", line 217, in merge_from_other_cfg
    _merge_a_into_b(cfg_other, self, self, [])
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/yacs/config.py", line 478, in _merge_a_into_b
    _merge_a_into_b(v, b[k], root, key_list + [k])
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/yacs/config.py", line 478, in _merge_a_into_b
    _merge_a_into_b(v, b[k], root, key_list + [k])
  File "/home/nina/miniconda3/envs/mask2former/lib/python3.10/site-packages/yacs/config.py", line 491, in _merge_a_into_b
    raise KeyError("Non-existent config key: {}".format(full_key))
KeyError: 'Non-existent config key: MODEL.RESNETS.RES5_MULTI_GRID'
(mask2former) nina@DESKTOP-QF7UBMR:/mnt/c/Users/USER/Desktop$
