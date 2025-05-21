

annotation_file = '/expanse/lustre/projects/ddp390/akulkarni/datasets/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt'

with open(annotation_file, 'r') as f:
    lines = f.read().splitlines()

lines = [line.split(' ') for line in lines]

lines = lines[1:]
for idx in range(len(lines)):
    if '' in lines[idx]:
        lines[idx].remove('')

# cls_of_interest = ['Smiling', 'Eyeglasses', 'Male', 'Bald', 'Blond_Hair', 'Brown_Hair', 'Black_Hair', 'Gray_Hair']
# # since first value is image name, so we take index+1 for the class labels
# cls_idx = [(lines[0].index(curr_cls) + 1) for curr_cls in cls_of_interest]
# # print(cls_idx)

# label_dict = {}
# # one line is extra (has class names)
# for img_idx in range(len(lines) - 1):
#     curr_line = lines[img_idx + 1]
#     label_dict[f'{curr_line[0]}'] = [curr_line[lbl_idx] for lbl_idx in cls_idx]

# print(label_dict['0.jpg'])

# print(label_dict['4.jpg'])

num_pos_samples = [0] * len(lines[0])
for idx in range(len(lines) - 1):
    curr_line = lines[idx + 1]
    for cls_idx in range(len(num_pos_samples)):
        if curr_line[cls_idx + 1] == '1':
            num_pos_samples[cls_idx] += 1


for cls_idx in range(len(num_pos_samples)):
    print(lines[0][cls_idx], num_pos_samples[cls_idx])