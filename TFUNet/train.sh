epochs=10
debugs=(500 1000 2000 3000 9000) 
debugs=(10) 
for debug in ${debugs[@]}
do
    # Unet
    python train.py -debug=$debug -warm_start=0    \
        -nntype=Unet -device=1,2 -a=0 -b=0 \
        -batch_size=4 -epochs=$epochs -lr=0.0001 -weight_decay=0.01

    # TF-Unet
    python train.py -debug=$debug -warm_start=0   \
        -nntype=Unet -device=1,2 -a=0.1 -b=0.1 \
        -batch_size=4 -epochs=$epochs -lr=0.0001 -weight_decay=0.01
done


