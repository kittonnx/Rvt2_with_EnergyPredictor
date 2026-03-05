#!/bin/bash

xvfb-run --auto-servernum python tools/dataset_generator.py \
	--save_path=/misc/dl001/dataset/tani/task_15 \
	--tasks=stack_blocks \
	--episodes_per_task=500 \
	--image_size=128,128 \
	--renderer=opengl \
	# --processes=1

# close_jar
# insert_onto_square_peg
# light_bulb_in
# meat_off_grill
# open_drawer
# place_cups
# place_shape_in_shape_sorter
# place_wine_at_rack_location
# push_buttons
# put_groceries_in_cupboard
# put_item_in_drawer
# put_money_in_safe
# reach_and_drag
# slide_block_to_color_target
# stack_blocks
# stack_cups
# sweep_to_dustpan_of_size
# turn_tap