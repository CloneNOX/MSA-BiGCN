w2vDim = [25, 50, 100, 200]
s2vDim = [64, 128, 256]
gcnHiddenDim = [128, 256, 512]
needStance = ['True', 'False']
lossRatio = [1.0, 0.5, 2.0]

for need in needStance:
    with open('./runNeedStance' + need + '.sh', 'w') as f:
        for ratio in lossRatio:
            for w2vD in w2vDim:
                for s2vD in s2vDim:
                    for gcnH in gcnHiddenDim:
                        order = 'python train.py '
                        order += '--w2vDim {:d} '.format(w2vD)
                        order += '--s2vDim {:d} '.format(s2vD)
                        order += '--gcnHiddenDim {:d} --rumorFeatureDim {:d} '.format(gcnH, gcnH)
                        order += '--w2vAttHeads 5 --s2vAttHeads 8 --dropout 0.5 --epoch 5000 '  
                        order += '--needStance ' + need + ' --lossRatio {:.2f} '.format(ratio)
                        order += '--logName ./log/needStance-{:s}/lossRatio-{:.2f}/{:d}w2vdim-{:d}s2vdim-{:d}hidden.txt '.format(
                            need, ratio, w2vD, s2vD, gcnH
                        )
                        order += '--savePath ./model/needStance-{:s}/lossRatio-{:.2f}/{:d}w2vdim-{:d}s2vdim-{:d}hidden.pt '.format(
                            need, ratio, w2vD, s2vD, gcnH
                        )
                        if need == 'True':
                            order += '--device cuda:0 '
                        else:
                            order += '--device cuda:1 '
                        order += "\n"
                        f.write(order)
                        
