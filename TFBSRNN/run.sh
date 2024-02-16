# eval
python evaluate.py -debug=0 -dataset=IDMT -device=1\
    -checkpoint=./weights/debug0_BSRNN_datasetIDMT_Ga0.0_Gb0.0_epoch53.pth \
    -nntype=BSRNN -a=0 -b=0 \
    -batch_size=10 -epochs=80 -lr=0.001 -weight_decay=0.00
