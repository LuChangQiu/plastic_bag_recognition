import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau


def plot_history(history):
    # 获取训练和验证的准确率及损失值
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history['loss']
    val_loss = history.history.get('val_loss', [])

    # 打印训练和验证的准确率及损失值
    print("Training accuracy:", acc)
    print("Validation accuracy:", val_acc)
    print("Training loss:", loss)
    print("Validation loss:", val_loss)

    # 生成epoch的序列
    epochs = range(1, len(acc) + 1)

    # 检查验证准确率长度是否与训练准确率长度匹配
    if len(val_acc) != len(acc):
        print("Warning: Validation accuracy length does not match training accuracy length.")
        val_acc = [None] * len(acc)

    # 检查验证损失长度是否与训练损失长度匹配
    if len(val_loss) != len(loss):
        print("Warning: Validation loss length does not match training loss length.")
        val_loss = [None] * len(loss)

    # 创建图形
    plt.figure(figsize=(12, 4))

    # 绘制训练和验证的准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')  # 蓝色圆点表示训练准确率
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')  # 蓝色线条表示验证准确率
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制训练和验证的损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')  # 蓝色圆点表示训练损失
    plt.plot(epochs, val_loss, 'b', label='Validation loss')  # 蓝色线条表示验证损失
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 显示图形
    plt.show()


if __name__ == "__main__":
    from model import build_model
    from data_loader import load_data

    # 定义数据集路径
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
        steps_per_epoch=len(train_generator) // 32,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=len(validation_generator) // 32,
        callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)]
    )

    # 绘制训练历史
    plot_history(history)
