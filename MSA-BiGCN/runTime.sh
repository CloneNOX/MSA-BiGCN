python train.py --dataPath ../dataset/semeval2017-task8/ --w2vDim 50 --s2vDim 64 --gcnHiddenDim 256 \
--rumorFeatureDim 256 --w2vAttHeads 5 --s2vAttHeads 8 --epoch 500 --lr 3e-5 \
--dropout 0.2 --weightDecay 0.001000 \
--logName ./log/runtime/needStanceMultitask.txt \
--savePath ./model/runtime/needNoStance/50w2v-64s2v-256hidden-0.2dropout-0.001000weight.pt \
--device cpu

python train.py --dataPath ../dataset/semeval2017-task8/ --w2vDim 50 --s2vDim 64 --gcnHiddenDim 256 \
--rumorFeatureDim 256 --w2vAttHeads 5 --s2vAttHeads 8 --epoch 500 --lr 3e-5 \
--dropout 0.2 --weightDecay 0.001000 --needStance False \
--logName ./log/runtime/needStanceMultitask.txt \
--savePath ./model/runtime/needNoStance/50w2v-64s2v-256hidden-0.2dropout-0.001000weight.pt \
--device cpu

python trainOnlyRumor.py --dataPath ../dataset/semeval2017-task8/ --w2vDim 50 --s2vDim 64 \
--gcnHiddenDim 256 --rumorFeatureDim 256 --w2vAttHeads 5 --s2vAttHeads 8 --epoch 500 --lr 3e-5 \
--dropout 0.2 --weightDecay 0.001000 \
--logName ./log/runtime/onlyRumor.txt \
--savePath ./model/runtime/onlyRumor.pt \
--device cpu
