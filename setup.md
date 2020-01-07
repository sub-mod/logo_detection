# Setup Tensorflow Objection Detection API

**Prerequisite**: 
Ensure protocol buffer is installed.Instructions for installing on Centos7/RHEL7/RHEL8 is given [here](https://blog.jeffli.me/blog/2016/12/08/install-protocol-buffer-from-source-in-centos-7/)
```
#test if protoc is installed.
▶ protoc --version
libprotoc 3.11.1
```
If you are using TensorFlow with GPU then CUDA10.0 must be installed and the following envs should be set.  
```
▶ export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
▶ export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

#### Clone tensorflow/models repo
```
▶ mkdir obj_detection
▶ cd obj_detection
▶ git clone https://github.com/tensorflow/models.git
▶ mkdir workspace
▶ ll
total 0
drwxrwxr-x 7 suku suku 249 Jan  6 21:06 models
drwxrwxr-x 7 suku suku  97 Jan  6 22:18 workspace
```

#### Create virtualenv
```
▶ virtualenv env
▶ source env/bin/activate
▶ ll
total 0
drwxrwxr-x 7 suku suku  80 Jan  6 22:16 env
drwxrwxr-x 7 suku suku 249 Jan  6 21:06 models
drwxrwxr-x 7 suku suku  97 Jan  6 22:18 workspace
▶ python -V
Python 3.6.8
```

#### Build Object Detection Python module & compile the protobufs
```
▶ cd models/research
▶ protoc object_detection/protos/*.proto --python_out=.
▶ export PYTHONPATH=$(pwd):$(pwd)/slim/:$PYTHONPATH
▶ python3 setup.py install
...
Using /obj_detection/env/lib/python3.6/site-packages
Finished processing dependencies for object-detection==0.1
▶ python3 setup.py build
```

##### Install dependencies
```
▶ cd ../../
▶ ls
drwxrwxr-x 7 suku suku  80 Jan  6 23:01 env
drwxrwxr-x 7 suku suku 249 Jan  6 22:59 models
drwxrwxr-x 7 suku suku  97 Jan  6 23:07 workspace
▶ pip3 install absl-py tensorflow-gpu==1.15.0 matplotlib image Cython jupyterlab pandas
▶ pip3 install pycocotools
```
#### Test if object_detection api is working.
You should get an OK at the end.
```
▶ python3 -c "from pycocotools import mask as mask ;print(mask.__author__)"
tsungyi
▶ python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.test.is_gpu_available())"
1.15.0
True
▶ python3 models/research/object_detection/builders/model_builder_test.py
...
[       OK ] ModelBuilderTest.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTest.test_unknown_meta_architecture
[       OK ] ModelBuilderTest.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTest.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTest.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 17 tests in 0.126s

OK (skipped=1)
```

## Training Custom Object Detector
Create workspace folders
```
▶ cd workspace
▶ mkdir -p images annotations pre-trained-model scripts my_project
▶ ll
total 0
drwxrwxr-x 2 suku suku 115 Jan  6 23:05 annotations
drwxrwxr-x 4 suku suku  31 Jan  6 23:04 images
drwxrwxr-x 2 suku suku   6 Jan  6 23:02 my_project
drwxrwxr-x 3 suku suku  46 Jan  6 23:03 pre-trained-model
drwxrwxr-x 2 suku suku  55 Jan  6 23:04 scripts
```
#### Download SSD Model
```
▶ cd pre-trained-model
▶ wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
▶ tar -xvf ssd_inception_v2_coco_2017_11_17.tar.gz
▶ ll ./
total 266M
drwxr-x--- 3 suku suku  161 Nov 17  2017 ssd_inception_v2_coco_2017_11_17
-rw-rw-r-- 1 suku suku 266M Nov 17  2017 ssd_inception_v2_coco_2017_11_17.tar.gz
▶ rm -fr ssd_inception_v2_coco_2017_11_17.tar.gz
▶ ll ./ssd_inception_v2_coco_2017_11_17
total 197M
-rw-r----- 1 suku suku   77 Nov 17  2017 checkpoint
-rw-r----- 1 suku suku  98M Nov 17  2017 frozen_inference_graph.pb
-rw-r----- 1 suku suku  96M Nov 17  2017 model.ckpt.data-00000-of-00001
-rw-r----- 1 suku suku  18K Nov 17  2017 model.ckpt.index
-rw-r----- 1 suku suku 3.6M Nov 17  2017 model.ckpt.meta
drwxr-x--- 3 suku suku   45 Nov 17  2017 saved_model
```

#### Prepare the Datasets using labelImg
Separate the data into 2 sets train and test.
```
▶ cd ../
▶ ls
annotations  images  my_project  pre-trained-model  scripts
▶ ll images
total 16K
drwxr-xr-x 2 suku suku 4.0K Jan  6 21:11 test
drwxr-xr-x 2 suku suku 8.0K Jan  6 21:11 train
```

#### Convert the xml files to csv files
```
▶ ll scripts
total 12K
-rw------- 1 suku suku 4.1K Jan  6 21:20 generate_tfrecord.py
-rw------- 1 suku suku 2.4K Jan  6 21:20 xml_to_csv.py
▶ python3 scripts/xml_to_csv.py -i images/test -o annotations/test_labels.csv
Successfully converted xml to csv.

▶ python3 scripts/xml_to_csv.py -i images/train -o annotations/train_labels.csv
Successfully converted xml to csv.
```
#### Generate tf records from datasets
create a Label map file which maps indices to category names.
```
▶ cat annotations/label_map.pbtxt
item{
       id:1
       name:'dynatrace_logo'
      }
item{
       id:2
       name: 'dynatrace_text'
      }

▶ ll annotations
total 20K
-rw------- 1 suku suku  111 Jan  6 23:05 label_map.pbtxt
-rw-rw-r-- 1 suku suku 2.8K Jan  6 23:04 test_labels.csv
-rw-rw-r-- 1 suku suku  12K Jan  6 23:04 train_labels.csv
▶ python3 scripts/generate_tfrecord.py --label0=dynatrace_logo --label1=dynatrace_text --csv_input=annotations/test_labels.csv --img_path=images/test --output_path=annotations/test.record
Successfully created the TFRecords: /home/suku/development/tf_odapi/obj_detection/workspace/annotations/test.record

▶ python3 scripts/generate_tfrecord.py --label0=dynatrace_logo --label1=dynatrace_text --csv_input=annotations/train_labels.csv --img_path=images/train --output_path=annotations/train.record
Successfully created the TFRecords: /home/suku/development/tf_odapi/obj_detection/workspace/annotations/train.record

▶ ll annotations
total 22M
-rw------- 1 suku suku  111 Jan  6 22:18 label_map.pbtxt
-rw-rw-r-- 1 suku suku 2.8K Jan  6 22:24 test_labels.csv
-rw-rw-r-- 1 suku suku 2.8M Jan  6 22:26 test.record
-rw-rw-r-- 1 suku suku  12K Jan  6 21:10 train_labels.csv
-rw-rw-r-- 1 suku suku  20M Jan  6 22:27 train.record
```
#### Setup the training projects with relevant files
open ssd_inception_v2_coco.config and make change where "CHANGE HERE" comment is present.
```
▶ cp ../models/research/object_detection/legacy/train.py my_project/
▶ cp ../models/research/object_detection/export_inference_graph.py my_project/
▶ cd my_project
▶ wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_inception_v2_coco.config
▶ ll ./
total 16K
-rw------- 1 suku suku 4.5K Jan  6 21:25 ssd_inception_v2_coco.config
-rw-rw-r-- 1 suku suku 6.8K Jan  6 21:27 train.py
-rw-rw-r-- 1 suku suku 7.1K Jan  6 21:38 export_inference_graph.py
```
#### Train the model
```
▶ python3 train.py --pipeline_config_path=./ssd_inception_v2_coco.config --train_dir=./training  --alsologtostderr
...
I0106 22:33:28.245677 140687540409920 learning.py:507] global step 59: loss = 16.1677 (0.050 sec/step)
INFO:tensorflow:global step 60: loss = 12.5695 (0.052 sec/step)
I0106 22:33:28.298278 140687540409920 learning.py:507] global step 60: loss = 12.5695 (0.052 sec/step)
INFO:tensorflow:global step 61: loss = 11.3254 (0.056 sec/step)
I0106 22:33:28.355229 140687540409920 learning.py:507] global step 61: loss = 11.3254 (0.056 sec/step)
^CTraceback (most recent call last):
[CtrL+C] to terminate the training

▶ ls
training  ssd_inception_v2_coco.config  train.py export_inference_graph.py

▶ ll training
total 248M
-rw-rw-r-- 1 suku suku   81 Jan  6 21:36 checkpoint
-rw-rw-r-- 1 suku suku  19M Jan  6 21:36 events.out.tfevents.1578364596.localhost.localdomain
-rw-rw-r-- 1 suku suku  17M Jan  6 21:36 graph.pbtxt
-rw-rw-r-- 1 suku suku 204M Jan  6 21:36 model.ckpt-50.data-00000-of-00001
-rw-rw-r-- 1 suku suku  53K Jan  6 21:36 model.ckpt-50.index
-rw-rw-r-- 1 suku suku 9.7M Jan  6 21:36 model.ckpt-50.meta
-rw------- 1 suku suku 4.8K Jan  6 21:36 pipeline.config
```

#### Create saved_model for exporting the model.
```
▶ python3 export_inference_graph.py --input_type=image_tensor --pipeline_config_path=./ssd_inception_v2_coco.config --trained_checkpoint_prefix=./training/model.ckpt-50 --output_directory=./inference
...
INFO:tensorflow:Writing pipeline config file to ./inference/pipeline.config
I0106 22:35:18.542130 140537440007744 config_util.py:190] Writing pipeline config file to ./inference/pipeline.config


▶ ll
total 696K
-rw-rw-r-- 1 suku suku 7.1K Jan  6 23:09 export_inference_graph.py
drwxr-xr-x 3 suku suku  184 Jan  6 23:11 inference
-rw-rw-r-- 1 suku suku 672K Jan  6 23:13 inference.ipynb
-rw------- 1 suku suku 4.8K Jan  6 23:09 ssd_inception_v2_coco.config
drwxrwxr-x 2 suku suku   40 Jan  6 23:21 test_images
drwxr-xr-x 2 suku suku  217 Jan  6 23:10 training
-rw-rw-r-- 1 suku suku 6.8K Jan  6 23:09 train.py

▶ ll inference
total 105M
-rw-rw-r-- 1 suku suku   77 Jan  6 21:41 checkpoint
-rw-rw-r-- 1 suku suku  52M Jan  6 21:41 frozen_inference_graph.pb
-rw-rw-r-- 1 suku suku  51M Jan  6 21:41 model.ckpt.data-00000-of-00001
-rw-rw-r-- 1 suku suku  18K Jan  6 21:41 model.ckpt.index
-rw-rw-r-- 1 suku suku 1.7M Jan  6 21:41 model.ckpt.meta
-rw-rw-r-- 1 suku suku 3.9K Jan  6 21:41 pipeline.config
drwxr-xr-x 3 suku suku   45 Jan  6 21:41 saved_model


▶ ll inference/saved_model
total 52M
-rw-rw-r-- 1 suku suku 52M Jan  6 21:41 saved_model.pb
drwxr-xr-x 2 suku suku   6 Jan  6 21:41 variables
```

#### Test the model
To test Inference code create a new environment
```
▶ cd ../../../
▶ ll
total 4.0K
drwxrwxr-x 7 suku suku  80 Jan  6 23:01 env
drwxrwxr-x 7 suku suku 249 Jan  6 22:59 models
drwxrwxr-x 7 suku suku  97 Jan  6 23:07 workspace

▶ virtualenv env2
▶ source env2/bin/activate
▶ ll
total 4.0K
drwxrwxr-x 7 suku suku  80 Jan  6 23:01 env
drwxrwxr-x 5 suku suku  56 Jan  6 23:15 env2
drwxrwxr-x 7 suku suku 249 Jan  6 22:59 models
-rw-rw-r-- 1 suku suku  15 Jan  6 22:58 README.md
drwxrwxr-x 7 suku suku  97 Jan  6 23:07 workspace

▶ pip3 install absl-py tensorflow-gpu=="2.*"  matplotlib image Cython jupyterlab pandas
Ensure that PYTHONPATH has not changed.Objection detection apis will be used in inference.
▶ jupyter lab
Open inference.ipynb
```
