w2vDim = [25, 50, 100, 200]
s2vDim = [64, 128, 256]
gcnHiddenDim = [64, 128, 256]
dropout = [0.2, 0.5]
#needStance = ['True', 'False']

#for need in needStance:
    #with open('./runNeedStance' + need + '.sh', 'w') as f:
with open('./runOnlyRumorSemEval.sh', 'w') as f:
    for ratio in dropout:
        for w2vD in w2vDim:
            for s2vD in s2vDim:
                for gcnH in gcnHiddenDim:
                    order = 'python trainOnlyRumor.py '
                    order += '--dataPath ../dataset/semeval2017-task8/ '
                    order += '--w2vDim {:d} '.format(w2vD)
                    order += '--s2vDim {:d} '.format(s2vD)
                    order += '--gcnHiddenDim {:d} --rumorFeatureDim {:d} '.format(gcnH, gcnH)
                    order += '--w2vAttHeads 5 --s2vAttHeads 8 --epoch 5000 '  
                    order +=  '--dropout {:.1f} '.format(ratio)
                    # order += '--needStance ' + need + ' --lossRatio {:.2f} '.format(ratio)
                    order += '--logName ./log/final/onlyRumor/semeval/dropout={:.1f}/{:d}w2v-{:d}s2v-{:d}hidden.txt '.format(
                        ratio, w2vD, s2vD, gcnH
                    )
                    order += '--savePath ./model/final/onlyRumor/semeval/dropout={:.1f}/{:d}w2v-{:d}s2v-{:d}hidden.pt '.format(
                        ratio, w2vD, s2vD, gcnH
                    )
                    # if need == 'True':
                    #     order += '--device cuda:0 '
                    # else:
                    order += '--device cuda:1'
                    order += "\n"
                    f.write(order)
                    
