PY_BIN='/usr/bin/python3'
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_pool_cnn.py --fold 0 --dp 0.25 --sdp 0.4
CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_pool_cnn_skip.py --fold 0 --dp 0.35 --sdp 0.4
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_pool_cnn.py --fold 0 --dp 0.4 --sdp 0.4
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_pool_cnn.py --fold 0 --dp 0.45 --sdp 0.4
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_pool_cnn.py --fold 0 --dp 0.5 --sdp 0.4
# conv
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_maxpool_deep_cnn.py --fold 0 --dp 0.35 --sdp 0.4
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_maxpool_deep_cnn.py --fold 0 --dp 0.4 --sdp 0.4
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_maxpool_deep_cnn.py --fold 0 --dp 0.45 --sdp 0.4
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_maxpool_deep_cnn.py --fold 0 --dp 0.5 --sdp 0.4
# maxpool avgpool cnn
# CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_maxpool_avgpool_cnn.py --fold 0 --dp 0.3 --sdp 0.4
# gru conv1d
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_gru_conv1d.py --fold 0 --sdp 0.25
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_gru_conv1d.py --fold 0 --sdp 0.3
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_gru_conv1d.py --fold 0 --sdp 0.35
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_gru_conv1d.py --fold 0 --sdp 0.4
# maxpool deep cnn
#CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_maxpool_deep_cnn.py --fold 0 --dp 0.55 --sdp 0.4
# pool cnn tta
# CUDA_VISIBLE_DEVICES=0 ${PY_BIN} ./model_pool_cnn_tta.py --fold 0 --dp 0.3 --sdp 0.4
