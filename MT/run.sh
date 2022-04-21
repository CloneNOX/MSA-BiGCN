python train.py --dataPath ../dataset/semeval2017-task8/ --epoch 1000 --modelType mtus --logName ./log/mtus-semeval.txt --savePath ./model/mtus-semeval.pt --lr 3e-5 --device cpu
python trainOnlyRumor.py --dataPath ../dataset/semeval2017-task8/ --epoch 500 --modelType mtus --logName ./log/mtus-semeval-onlyRumor.txt --savePath ./model/mtus-semeval-onlyRumor.pt --lr 3e-5 --device cpu
#python trainOnlyRumor.py --dataPath ../dataset/semeval2017-task8/ --epoch 500 --modelType mtus --logName ./log/mtus-semeval-onlyStance.txt --savePath ./model/mtus-semeval-onlyStance.pt --lr 3e-5
#python trainOnlyRumor.py --dataPath ../dataset/PHEME/ --epoch 1000 --modelType mtus --logName ./log/mtus-PHEME.txt --savePath ./model/mtus-PHEME.pt --lr 3e-5

python train.py --dataPath ../dataset/semeval2017-task8/ --epoch 1000 --modelType mtes --logName ./log/mtes-semeval.txt --savePath ./model/mtes-semeval.pt --lr 3e-5 --device cpu
python trainOnlyRumor.py --dataPath ../dataset/semeval2017-task8/ --epoch 500 --modelType mtes --logName ./log/mtes-semeval-onlyRumor.txt --savePath ./model/mtes-semeval-onlyRumor.pt --lr 3e-5 --device cpu
#python trainOnlyRumor.py --dataPath ../dataset/semeval2017-task8/ --epoch 500 --modelType mtes --logName ./log/mtes-semeval-onlyStance.txt --savePath ./model/mtes-semeval-onlyStance.pt --lr 3e-5
#python trainOnlyRumor.py --dataPath ../dataset/PHEME/ --epoch 1000 --modelType mtes --logName ./log/mtes-PHEME.txt --savePath ./model/mtes-PHEME.pt --lr 3e-5