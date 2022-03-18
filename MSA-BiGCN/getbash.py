w2vDim = [25, 50, 100, 200]
s2vDim = [64, 128, 256]
w2vAttHeads = [5]
s2vAttHeads = [8]
gcnHiddenDim = [128, 256, 512]

with open('./paramRun.sh', 'w') as f:
    for w2vD in w2vDim:
        for s2vD in s2vDim:
            for w2vH in w2vAttHeads:
                for s2vH in s2vAttHeads:
                    for gcnH in gcnHiddenDim:
                        if(w2vDim == 25 and w2vAttHeads == 10):
                            continue
                        order = 'python train.py --w2vDim {:d} --s2vDim {:d} --w2vAttHeads {:d} --s2vAttHeads {:d} --gcnHiddenDim {:d} --rumorFeatureDim {:d} --dropout 0.5 --lr 3e-4 --weightDecay 5e-4 --epoch 2000 '.format(
                            w2vD,
                            s2vD,
                            w2vH,
                            s2vH,
                            gcnH,
                            gcnH,
                        )
                        order += '--logName ./log/{:d}w2vdim-{:d}s2vdim-{:d}w2vhead-{:d}s2vhead-{:d}hidden.txt '.format(
                            w2vD, s2vD, w2vH, s2vH, gcnH, gcnH,
                        )
                        order += '--savePath ./model/{:d}w2vdim-{:d}s2vdim-{:d}w2vhead-{:d}s2vhead-{:d}hidden.pt '.format(
                            w2vD, s2vD, w2vH, s2vH, gcnH, gcnH,
                        )
                        order += '--optimizer SGD'
                        f.write(order)
                        
