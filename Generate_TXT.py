import os

train_txt_path = r"/home/ubuntu/Yaozhichao/Litong_JDCV_DCL/datastes/Train.txt"
test_txt_path = r"/home/ubuntu/Yaozhichao/Litong_JDCV_DCL/datastes/Test.txt"
val_txt_path = r'/home/ubuntu/Yaozhichao/Litong_JDCV_DCL/datastes/Val.txt'

try:
    train_file = open(train_txt_path, "w")
    test_file = open(test_txt_path, "w")
    val_file = open(val_txt_path, "w")

    img_dir = r"/home/ubuntu/Yaozhichao/Saturation_Classification/0"
    imgs = os.listdir(img_dir)
    test_count = 0
    val_count = 0
    for i, img in enumerate(imgs):
        img_path = os.path.join(img_dir, img)

        if i % 79 == 0 and test_count < 50:
            test_file.write(f"{img_path} 0\n")
            test_count += 1

        elif i % 73 == 0 and val_count < 50:
            val_file.write(f'{img_path} 0\n')
            val_count += 1
        else:
            train_file.write(f"{img_path} 0\n")
    print("0标签写入完成")

    img_dir = r"/home/ubuntu/Yaozhichao/Saturation_Classification/1"
    imgs = os.listdir(img_dir)
    test_count = 0
    val_count = 0
    for i, img in enumerate(imgs):
        img_path = os.path.join(img_dir, img)

        if i % 79 == 0 and test_count < 50:
            test_file.write(f"{img_path} 1\n")
            test_count += 1

        elif i % 73 == 0 and val_count < 50:
            val_file.write(f'{img_path} 1\n')
            val_count += 1
        else:
            train_file.write(f"{img_path} 1\n")
    print("1该标签写入完成")

    img_dir = r"/home/ubuntu/Yaozhichao/Saturation_Classification/2"
    imgs = os.listdir(img_dir)
    test_count = 0
    val_count = 0
    for i, img in enumerate(imgs):
        img_path = os.path.join(img_dir, img)

        if i % 79 == 0 and test_count < 50:
            test_file.write(f"{img_path} 2\n")
            test_count += 1

        elif i % 73 == 0 and val_count < 50:
            val_file.write(f'{img_path} 2\n')
            val_count += 1
        else:
            train_file.write(f"{img_path} 2\n")
    print("2标签写入完成")

    img_dir = r"/home/ubuntu/Yaozhichao/Saturation_Classification/3"
    imgs = os.listdir(img_dir)
    test_count = 0
    val_count = 0
    for i, img in enumerate(imgs):
        img_path = os.path.join(img_dir, img)

        if i % 79 == 0 and test_count < 50:
            test_file.write(f"{img_path} 3\n")
            test_count += 1

        elif i % 73 == 0 and val_count < 50:
            val_file.write(f'{img_path} 3\n')
            val_count += 1
        else:
            train_file.write(f"{img_path} 3\n")
    print("3标签写入完成")
finally:
    train_file.close()
    test_file.close()
    val_file.close()