from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
# config.output = '/root/jinyfeng/models/insightface_models/ms1mv2_r50'
config.output = '/root/jinyfeng/models/insightface_models/ms1mv3_r50'
# config.embedding_size = 512
config.embedding_size = 256
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
# config.lr = 0.02
config.verbose = 2000
config.dali = False

config.lr = 0.02
config.rec = "/root/jinyfeng/datas/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510

# config.lr = 0.1
# config.rec = "/root/jinyfeng/datas/faces_emore"
# config.num_classes = 85742
# config.num_image = 5822653

config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
