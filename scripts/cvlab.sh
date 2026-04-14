python examples/musc_main.py --device 0 \
--data_path ../SR/ --dataset_name cvlab --class_name china \
--backbone_name ViT-B-16 --pretrained openai --feature_layers 2 5 8 11 \
--img_resize 1024 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True