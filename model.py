import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from data_loader import load_data  # 导入 load_data 函数


def build_model():
    """
    构建卷积神经网络模型。
    """
    # 加载预训练的 VGG16 模型
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    # 冻结基模型的层
    for layer in base_model.layers:
        layer.trainable = False

    # 添加自定义层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # 构建最终模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=1e-4),  # 使用 Adam 优化器，学习率为 1e-4
                  loss='binary_crossentropy',  # 损失函数为二元交叉熵
                  metrics=['accuracy'])  # 评估指标为准确率

    return model


if __name__ == "__main__":
    train_dir = 'dataset/train'
    validation_dir = 'dataset/validation'
    test_dir = 'dataset/test'
    # 加载数据
    train_generator, validation_generator, test_generator = load_data(train_dir, validation_dir, test_dir)

    # 构建模型
    model = build_model()

    # 训练模型
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator) // train_generator.batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=len(validation_generator) // validation_generator.batch_size
    )

    # 评估模型
    test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))  # 在测试集上评估模型
    print('Test accuracy:', test_acc)  # 打印测试集上的准确率

    # 保存模型
    model.save('plastic_bag_recognition_model.keras')  # 保存训练好的模型

    # 预测并打印含有塑料袋的图片
    predictions = model.predict(test_generator, steps=len(test_generator))  # 在测试集上进行预测
    class_indices = test_generator.class_indices  # 获取类别索引
    labels = dict((v, k) for k, v in class_indices.items())  # 将类别索引转换为字典

    # 重置测试生成器以确保从头开始读取数据
    test_generator.reset()

    # 获取所有测试数据
    images, labels_list, filenames = [], [], []
    for _ in range(len(test_generator)):
        batch_images, batch_labels = next(test_generator)  # 获取一个批次的数据
        images.extend(batch_images)  # 将当前批次的图像添加到列表中
        labels_list.extend(batch_labels)  # 将当前批次的标签添加到列表中
        filenames.extend(test_generator.filenames)  # 获取当前批次的文件名

    for i in range(len(predictions)):
        predicted_class = 1 if predictions[i][0] > 0.5 else 0  # 根据预测值判断类别
        true_class = labels_list[i]  # 获取真实类别

        if predicted_class == 1:  # 如果预测结果为塑料袋
            plt.imshow(images[i])  # 显示图像
            plt.title(f'Predicted: {labels[predicted_class]}, True: {labels[true_class]}')  # 设置标题
            plt.show()  # 显示图像
            print(f'File name: {filenames[i]}')  # 打印文件名
