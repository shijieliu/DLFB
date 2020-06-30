###
 # @Author: liushijie
 # @Date: 2020-06-28 16:26:17
 # @LastEditTime: 2020-06-28 16:27:49
 # @LastEditors: liushijie
 # @Description: 
 # @FilePath: /LightLR/data/mnist.sh
### 
#!/usr/bin/env bash

ROOT_DIR='./'
mkdir -p ${ROOT_DIR}

curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip t*-ubyte.gz