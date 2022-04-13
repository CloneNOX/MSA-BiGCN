w2vDim = [25, 50, 100, 200]
s2vDim = [64, 128, 256]
gcnHiddenDim = [64, 128, 256]
dropout = [0.2]
weightDecay = [1e-3]
needStance = ['True', 'False']

for need in needStance:
    with open('./compare{:s}.sh'.format('NeedStance' if need == 'True' else 'NeedNoStance'), 'w') as f:
        for ratio in dropout:
            for weight in weightDecay:
                for w2vD in w2vDim:
                    for s2vD in [64]:
                        for gcnH in [256]:
                            order = 'python train.py '
                            order += '--dataPath ../dataset/semeval2017-task8/ '
                            order += '--w2vDim {:d} '.format(w2vD)
                            order += '--s2vDim {:d} '.format(s2vD)
                            order += '--gcnHiddenDim {:d} --rumorFeatureDim {:d} '.format(gcnH, gcnH)
                            order += '--w2vAttHeads 5 --s2vAttHeads 8 --epoch 2000 --lr 3e-5 '  
                            order += '--dropout {:.1f} '.format(ratio)
                            order += '--weightDecay {:f} '.format(weight)
                            order += '--needStance ' + need + ' '
                            order += '--logName ./log/compare/{:s}/{:d}w2v-{:d}s2v-{:d}hidden-{:.1f}dropout-{:f}weight.txt '.format(
                                ('needStance' if need == 'True' else 'needNoStance'), w2vD, s2vD, gcnH, ratio, weight
                            )
                            order += '--savePath ./model/compare/{:s}/{:d}w2v-{:d}s2v-{:d}hidden-{:.1f}dropout-{:f}weight.pt '.format(
                                ('needStance' if need == 'True' else 'needNoStance'), w2vD, s2vD, gcnH, ratio, weight
                            )
                            if need == 'True':
                                order += '--device cuda:0 '
                            else:
                                order += '--device cuda:1'
                            order += "\n"
                            f.write(order)
                for w2vD in [50]:
                    for s2vD in s2vDim:
                        for gcnH in [256]:
                            order = 'python train.py '
                            order += '--dataPath ../dataset/semeval2017-task8/ '
                            order += '--w2vDim {:d} '.format(w2vD)
                            order += '--s2vDim {:d} '.format(s2vD)
                            order += '--gcnHiddenDim {:d} --rumorFeatureDim {:d} '.format(gcnH, gcnH)
                            order += '--w2vAttHeads 5 --s2vAttHeads 8 --epoch 2000 --lr 3e-5 '  
                            order += '--dropout {:.1f} '.format(ratio)
                            order += '--weightDecay {:f} '.format(weight)
                            order += '--needStance ' + need + ' '
                            order += '--logName ./log/compare/{:s}/{:d}w2v-{:d}s2v-{:d}hidden-{:.1f}dropout-{:f}weight.txt '.format(
                                ('needStance' if need == 'True' else 'needNoStance'), w2vD, s2vD, gcnH, ratio, weight
                            )
                            order += '--savePath ./model/compare/{:s}/{:d}w2v-{:d}s2v-{:d}hidden-{:.1f}dropout-{:f}weight.pt '.format(
                                ('needStance' if need == 'True' else 'needNoStance'), w2vD, s2vD, gcnH, ratio, weight
                            )
                            if need == 'True':
                                order += '--device cuda:0 '
                            else:
                                order += '--device cuda:1'
                            order += "\n"
                            f.write(order)
                for w2vD in [50]:
                    for s2vD in [64]:
                        for gcnH in gcnHiddenDim:
                            order = 'python train.py '
                            order += '--dataPath ../dataset/semeval2017-task8/ '
                            order += '--w2vDim {:d} '.format(w2vD)
                            order += '--s2vDim {:d} '.format(s2vD)
                            order += '--gcnHiddenDim {:d} --rumorFeatureDim {:d} '.format(gcnH, gcnH)
                            order += '--w2vAttHeads 5 --s2vAttHeads 8 --epoch 2000 --lr 3e-5 '  
                            order += '--dropout {:.1f} '.format(ratio)
                            order += '--weightDecay {:f} '.format(weight)
                            order += '--needStance ' + need + ' '
                            order += '--logName ./log/compare/{:s}/{:d}w2v-{:d}s2v-{:d}hidden-{:.1f}dropout-{:f}weight.txt '.format(
                                ('needStance' if need == 'True' else 'needNoStance'), w2vD, s2vD, gcnH, ratio, weight
                            )
                            order += '--savePath ./model/compare/{:s}/{:d}w2v-{:d}s2v-{:d}hidden-{:.1f}dropout-{:f}weight.pt '.format(
                                ('needStance' if need == 'True' else 'needNoStance'), w2vD, s2vD, gcnH, ratio, weight
                            )
                            if need == 'True':
                                order += '--device cuda:0 '
                            else:
                                order += '--device cuda:1'
                            order += "\n"
                            f.write(order)
                        
