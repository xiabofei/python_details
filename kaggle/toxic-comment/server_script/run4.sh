PY_BIN='/usr/bin/python3'
# deep gru manul feat 
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_deep_gru_manulFeat.py --fold 4 --dp 0.375 --sdp 0.4
# deep gru
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_deep_gru.py --fold 4 --dp 0.375 --sdp 0.3
CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_deep_gru.py --fold 4 --dp 0.375 --sdp 0.4
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_deep_gru.py --fold 4 --dp 0.35 --sdp 0.4
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_deep_gru.py --fold 4 --dp 0.4 --sdp 0.4
# maxpool gru
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_maxpool_gru.py --fold 4 --sdp 0.25
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_maxpool_gru.py --fold 4 --sdp 0.3
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_maxpool_gru.py --fold 4 --dp 0.3 --sdp 0.35
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_maxpool_gru.py --fold 4 --sdp 0.4
# skip gru
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_skip_gru.py --fold 4 --dp 0.375 --sdp 0.25
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_skip_gru.py --fold 4 --dp 0.375 --sdp 0.3
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_skip_gru.py --fold 4 --dp 0.375 --sdp 0.35
#CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_skip_gru.py --fold 4 --dp 0.375 --sdp 0.4
# deep gru tta
# CUDA_VISIBLE_DEVICES=4 ${PY_BIN} ./model_deep_gru_tta.py --fold 4 --dp 0.375 --sdp 0.4
