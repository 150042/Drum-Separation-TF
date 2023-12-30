
epochs=10
debugs=(500 1000 2000 3000 9000) 
debugs=(10) 
for debug in ${debugs[@]}
do
    # MERT-FT
    python train.py -debug=$debug -warm_start=0   \
        -nntype=MERT -device=2,3 -a=0.1 -b=0.1 \
        -batch_size=8 -epochs=$epochs -lr=0.00002 -weight_decay=0.00

    # MERT
    python train.py -debug=$debug -warm_start=0   \
        -nntype=MERT -device=2,3 -a=0 -b=0 \
        -batch_size=8 -epochs=$epochs -lr=0.00002 -weight_decay=0.00
done

