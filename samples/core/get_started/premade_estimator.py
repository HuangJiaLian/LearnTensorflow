# -*- coding:utf-8 -*-
#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # 1.Fetch the data including training set and test set 　获取数据集(训练数据和测试数据)
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    

    # 2. Feature columns describe how to use the input. 创建特征列表
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # 上面这段和下面的功能一样
    # 就像是一个结构体，里面有好几种属性
    # my_feature_columns = [
    # tf.feature_column.numeric_column(key='SepalLength'),
    # tf.feature_column.numeric_column(key='SepalWidth'),
    # tf.feature_column.numeric_column(key='PetalLength'),
    # tf.feature_column.numeric_column(key='PetalWidth')
    # ]


    # 3.指定model的类型: 使用神经网络(fully connected neural network)
    # 使用已经做好的分类class : tf.estimator.DNNClassifier
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        # 因为最后有3种不同的类型
        n_classes=3)

    # 4.Train the Model.
    classifier.train(
        # 训练的方法　
        # train_x: features
        # train_y: labels
        # batch_size: 一个batch里面有多少个examples
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        # 训练重复的次数
        steps=args.train_steps)

    # 5.Evaluate the model. 用测试集去测试分类model的准确性
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # 6.Generate predictions from the model 预测没有标签的example
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        # 和Evaluate的函数用的是一样的，知识没有传入lable参数
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
        # 最有可能是哪个分类
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
