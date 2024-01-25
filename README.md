# pdf-token-type-publaynet


# execute VGT with PubLayNet

cd /home/gabo/projects/AdvancedLiterateMachinery/DocumentUnderstanding/VGT/object_detection
python train_VGT.py --config-file /home/gabo/projects/AdvancedLiterateMachinery/DocumentUnderstanding/VGT/object_detection/Configs/cascade/publaynet_VGT_cascade_PTM.yaml --eval-only --num-gpus 1 MODEL.WEIGHTS /home/gabo/projects/pdf-token-type-publaynet/model/publaynet_VGT_model.pth OUTPUT_DIR /home/gabo/projects/pdf-token-type-publaynet/vgt_result
