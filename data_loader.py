from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, validation_dir, test_dir):
    # 数据增强和预处理
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # 新增垂直翻转
        brightness_range=[0.5, 1.5],  # 新增亮度变化
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # 加载训练数据
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    # 加载验证数据
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    # 加载测试数据
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    print("The data load is successful .")

    return train_generator, validation_generator, test_generator