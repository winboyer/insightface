#CUDA_VISIBLE_DEVICES=0
#python extract_feat.py --network r50 --weight /root/jinyfeng/models/insightface_models/ms1mv3_r50/model.pt --img-folder /root/jinyfeng/datas/sensoro/20230823_faces --output-path /root/jinyfeng/datas/sensoro/facefeat_v0.bin > face_feat.log
CUDA_VISIBLE_DEVICES=1 python extract_alignface_feat.py --network r50 --weight /root/jinyfeng/models/insightface/insightfaceModels/ms1mv3_r50/model.pt --img-folder /data2/ossdata/mz/dy_wb_xhs_alignface --output-path /data2/ossdata/mz/dy_wb_xhs_alignfacefeat_ms1mv3_r50.bin > alignfacefeat_ms1mv3_r50.log

#CUDA_VISIBLE_DEVICES=2 python extract_alignface_feat.py --network r50 --weight /root/jinyfeng/models/insightface/insightfaceModels/glint360k_r50/model.pt --img-folder /data2/ossdata/mz/dy_wb_xhs_alignface --output-path /data2/ossdata/mz/dy_wb_xhs_alignfacefeat_glint360k_r50.bin > alignfacefeat_glint360k_r50.log

