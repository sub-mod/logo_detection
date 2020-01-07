Download the SSD model in this Folder.  

Instructions to setup SSD model: 
```
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
