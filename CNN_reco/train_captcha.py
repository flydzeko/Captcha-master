from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET
from sample import sample_conf
import numpy as np
import tensorflow as tf
import os
import random
from PIL import Image
import time
import index


class TrainError(Exception):
    pass

class TrainModel(object):

    def __init__(self, train_img_path, char_set, verify=False):


        # 打乱文件顺序+校验图片格式
        self.train_img_path = train_img_path
        self.train_images_list = os.listdir(train_img_path)
        # 校验格式
        if verify:
            self.confirm_image_suffix()
        # 打乱文件顺序
        random.seed(time.time())
        random.shuffle(self.train_images_list)
        '''
        # 验证集文件
        self.verify_img_path = verify_img_path
        self.verify_images_list = os.listdir(verify_img_path)
        '''
        # 获得图片宽高和字符长度基本信息
        text, image = self.gen_captcha_text_image(train_img_path, self.train_images_list[0])
        image = np.array(image)
        print("验证码图像channel:", image.shape)  # (60, 160, 3)

        # 初始化变量
        # 图片尺寸
        self.IMAGE_HEIGHT = 60
        self.IMAGE_WIDTH = 160
        # 验证码长度（位数）
        self.MAX_CAPTCHA = len(text)
        print("验证码文本最长字符数", self.MAX_CAPTCHA)  # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐
        # 验证码字符类别
        self.char_set = char_set
        self.CHAR_SET_LEN = len(char_set)

        # tf初始化占位符
        self.X = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        self.Y = tf.placeholder(tf.float32, [None, self.MAX_CAPTCHA * self.CHAR_SET_LEN])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout
        self.w_alpha = 0.01
        self.b_alpha = 0.1

    # 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
    def convert2gray(self,img):
        if len(img.shape) > 2:
            gray = np.mean(img, -1)
            # 上面的转法较快，正规转法如下
            # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
            # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    def gen_captcha_text_image(self,img_path, img_name):

        # 标签
        label = img_name.split(".")[0]
        # 文件
        img_file = os.path.join(img_path, img_name)  # 路径字符串拼接
        captcha_image = Image.open(img_file)
        captcha_array = np.array(captcha_image)  # 向量化
        return label, captcha_array

    # 图像二值化
    def two_value(self,image):
        # 灰度化
        lim = image.convert('L')
        # 灰度阈值设为155，低于这个值的点全部填白色
        threshold = 155
        table = []

        for j in range(256):
            if j < threshold:
                table.append(0)
            else:
                table.append(1)

        bim = lim.point(table, '1')
        return bim


    def noise_remove(self,img, k):

        def calculate_noise_count(img_obj, w, h):
            count = 0
            width, height = img_obj.size
            for _w_ in [w - 1, w, w + 1]:
                for _h_ in [h - 1, h, h + 1]:
                    if _w_ > width - 1:
                        continue
                    if _h_ > height - 1:
                        continue
                    if _w_ == w and _h_ == h:
                        continue
                    if img_obj.getpixel((_w_, _h_)) < 230:
                        count += 1
            return count


        w, h = img.size
        for _w in range(w):
            for _h in range(h):
                if _w == 0 or _h == 0:
                    img.putpixel((_w, _h), 255)
                    continue
                # 计算邻域非白色的个数
                pixel = img.getpixel((_w, _h))
                if pixel == 255:
                    continue

                if calculate_noise_count(img, _w, _h) < k:
                    img.putpixel((_w, _h), 255)
        return img

    # 文本转向量
    def text2vec(self, text):
        """
        转标签为oneHot编码
        :param text: str
        :return: numpy.array
        """
        text_len = len(text)
        if text_len > self.MAX_CAPTCHA:
            raise ValueError('验证码最长{}个字符'.format(self.MAX_CAPTCHA))

        vector = np.zeros(self.MAX_CAPTCHA * self.CHAR_SET_LEN)

        for i, ch in enumerate(text):
            idx = i * self.CHAR_SET_LEN + self.char_set.index(ch)
            vector[idx] = 1
        return vector

    # 向量转回文本
    def vec2text(self,vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_at_pos = i  # c/63
            char_idx = c % self.CHAR_SET_LEN
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)

    # 生成一个训练batch
    def get_next_batch(self,n,size=128):
        batch_x = np.zeros([size, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])  # 初始化
        batch_y = np.zeros([size, self.MAX_CAPTCHA * self.CHAR_SET_LEN])  # 初始化

        max_batch = int(len(self.train_images_list) / size)
        # print(max_batch)
        if max_batch - 1 < 0:
            raise TrainError("训练集图片数量需要大于每批次训练的图片数量")
        if n > max_batch - 1:
            n = n % max_batch
        s = n * size
        e = (n + 1) * size
        this_batch = self.train_images_list[s:e]

        for i, img_name in enumerate(this_batch):
            label, image_array = self.gen_captcha_text_image(self.train_img_path, img_name)
            image_array = self.convert2gray(image_array)  # 灰度化图片
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.text2vec(label)  # 生成 oneHot
        return batch_x, batch_y

    def confirm_image_suffix(self):
        # 在训练前校验所有文件格式
        print("开始校验所有图片后缀")
        for index, img_name in enumerate(self.train_images_list):
            print("{} image pass".format(index), end='\r')
            if not img_name.endswith(sample_conf['image_suffix']):
                raise TrainError('confirm images suffix：you request [.{}] file but get file [{}]'
                                 .format(sample_conf['image_suffix'], img_name))
        print("所有图片格式校验通过")


####################################################################

    # 定义CNN
    def crack_captcha_cnn(self,w_alpha=0.01, b_alpha=0.1):
        x = tf.reshape(self.X, shape=[-1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1])

        # 3 conv layer
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))#3*3卷积核/卷积patch的大小，1个输入，32个输出
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))#每个输出对应一个偏置值
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))#strides步长   padding边距
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1,self.keep_prob)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        # print("con3====",conv3.get_shape())

        # Fully connected layer
        w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))#输入64张8*20的图片，加入一个有1024个神经元的全连接层
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self.keep_prob)

        # Fully connected layer2
        # w_d2 = tf.Variable(w_alpha * tf.random_normal([1024, 1024]))
        # b_d2 = tf.Variable(b_alpha * tf.random_normal([1024]))
        # dense2 = tf.nn.relu(tf.add(tf.matmul(dense, w_d2), b_d2))
        # dense2 = tf.nn.dropout(dense2, keep_prob)

        w_out = tf.Variable(w_alpha * tf.random_normal([1024,self.MAX_CAPTCHA * self.CHAR_SET_LEN]))
        b_out = tf.Variable(b_alpha * tf.random_normal([self.MAX_CAPTCHA * self.CHAR_SET_LEN]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        # out = tf.nn.softmax(out)
        return out
    def model(self):
        x = tf.reshape(self.X, shape=[-1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1])
        print(">>> input x: {}".format(x))

        # 卷积层1
        wc1 = tf.get_variable(name='wc1', shape=[3, 3, 1, 32], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc1 = tf.Variable(self.b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)

        # 卷积层2
        wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc2 = tf.Variable(self.b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        # 卷积层3
        wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc3 = tf.Variable(self.b_alpha * tf.random_normal([128]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        print(">>> convolution 3: ", conv3.shape)
        next_shape = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]

        # 全连接层1
        wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.Variable(self.b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, self.keep_prob)

        # 全连接层2
        wout = tf.get_variable('name', shape=[1024, self.MAX_CAPTCHA* self.CHAR_SET_LEN], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.Variable(self.b_alpha * tf.random_normal([self.MAX_CAPTCHA * self.CHAR_SET_LEN]))
        y_predict = tf.add(tf.matmul(dense, wout), bout)
        return y_predict

    # 训练
    def train_crack_captcha_cnn(self):
        output =self.model()
        # loss
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self.Y))
        # 最后一层用来分类的softmax和sigmoid有什么不同？
        # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        predict = tf.reshape(output, [-1, self.MAX_CAPTCHA, self.CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.MAX_CAPTCHA, self.CHAR_SET_LEN]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver()


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            index.fileindex1 = 0
            step = 0
            while True:
                batch_x, batch_y = self.get_next_batch(step,64)
                _, loss_ = sess.run([optimizer, loss], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})
                print(step, loss_)
                f1 = open('temp1.txt', 'a')
                f1.writelines([str(step), " ", str(loss_), "\n"])
                index.fileindex1 = 1

                # 每100 step计算一次准确率
                if step % 100 == 0:
                    batch_x_test, batch_y_test = self.get_next_batch(step,100)
                    acc = sess.run(accuracy, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                    print(step, "acc=", acc)
                    # 如果准确率大于50%,保存模型,完成训练
                    # if acc > 0.9:
                    if step > 9000:
                        saver.save(sess, "save/crack_capcha.model", global_step=step)
                        break
                step += 1


def main():
    train_image_dir = sample_conf["train_image_dir"]
    # verify_image_dir = sample_conf["test_image_dir"]
    char_set = sample_conf["char_set"]
    tm = TrainModel(train_image_dir, char_set, verify=False)
    tm.train_crack_captcha_cnn()# 开始训练模型
    index.fileindex1=-1
    f1 = open('temp1.txt', 'w')
    f1.close()

if __name__ == '__main__':
    main()