
SNPE  V1.65对dlc的量化模型与非量化模型在8155车机板子的CPU和DSP上进行推理

`config.json` 参数说明：
```python
{
	"model_file": "./multi_label_classification.dlc", # model
	"input_files": "./test_img", # test file fold
	"image_width": 448,   # input image width
	"image_height": 448,  # input image height
	"output_node": "outPutNode", # output node
	"use_quant": false #  quantization enabled
}
```

模型对应的项目连接为 [Multi-label-classification](https://github.com/youngx123/Multi-label-classification)

未量化模型和量化模型在精度和推理时间的结果过对比
| 未量化模型                                                | 量化模型                                                       |
| --------------------------------------------------------- | -------------------------------------------------------------- |
| model file : ./multi_label/multi_label_classification.dlc | model file: ./multi_label/multi_label_classification-quant.dlc |
| use   quant: 0                                            | use quant : 2                                                  |
| use none quantization                                     | use quantization                                               |
| SNPE Version : 1.65.qnx                                   | SNPE Version: 1.65.qnx                                         |
| InputShape: 1, 448, 448,  3                               | InputShape: 1, 448, 448, 3                                     |
| name : outPutNodebuffer size : 36                         | name: outPutNodebuffer size: 36                                |
| file name : ./test_multi/157347722184_5.jpg               | file name: ./test_multi/157347722184_5.jpg                     |
| 0  --> 0.001410                                           | 0  --> 0.000000                                                |
| 1  --> 0.000000                                           | 1  --> 0.000000                                                |
| 2  --> 0.124404                                           | 2  --> 0.137255                                                |
| 3  --> 0.608466                                           | 3  --> 0.545098                                                |
| 4  --> 0.001935                                           | 4  --> 0.003922                                                |
| 5  --> 0.000001                                           | 5  --> 0.000000                                                |
| 6  --> 0.887029                                           | 6  --> 0.886275                                                |
| 7  --> 0.003464                                           | 7  --> 0.003922                                                |
| 8  --> 0.000001                                           | 8  --> 0.000000                                                |
| file name : ./test_multi/15734772266_12.jpg               | file name : ./test_multi/15734772266_12.jpg                    |
| 0  --> 0.000576                                           | 0  --> 0.000000                                                |
| 1  --> 0.000000                                           | 1  --> 0.000000                                                |
| 2  --> 0.589654                                           | 2  --> 0.592157                                                |
| 3  --> 0.105775                                           | 3  --> 0.098039                                                |
| 4  --> 0.061643                                           | 4  --> 0.070588                                                |
| 5  --> 0.000001                                           | 5  --> 0.000000                                                |
| 6  --> 0.728276                                           | 6  --> 0.717647                                                |
| 7  --> 0.006237                                           | 7  --> 0.007843                                                |
| 8  --> 0.000000                                           | 8  --> 0.000000                                                |
| file name : ./test_multi/15734773309459999_128.jpg        | file name : ./test_multi/15734773309459999_128.jpg             |
| 0  --> 0.011253                                           | 0  --> 0.011765                                                |
| 1  --> 0.000000                                           | 1  --> 0.000000                                                |
| 2  --> 0.407571                                           | 2  --> 0.407843                                                |
| 3  --> 0.165056                                           | 3  --> 0.160784                                                |
| 4  --> 0.078271                                           | 4  --> 0.070588                                                |
| 5  --> 0.000030                                           | 5  --> 0.000000                                                |
| 6  --> 0.353301                                           | 6  --> 0.364706                                                |
| 7  --> 0.001757                                           | 7  --> 0.000000                                                |
| 8  --> 0.000006                                           | 8  --> 0.000000                                                |
| file name : ./test_multi/15734776914190001_106.jpg        | file name : ./test_multi/15734776914190001_106.jpg             |
| 0  --> 0.001395                                           | 0  --> 0.000000                                                |
| 1  --> 0.000037                                           | 1  --> 0.000000                                                |
| 2  --> 0.061333                                           | 2  --> 0.070588                                                |
| 3  --> 0.983202                                           | 3  --> 0.980392                                                |
| 4  --> 0.017695                                           | 4  --> 0.019608                                                |
| 5  --> 0.003111                                           | 5  --> 0.003922                                                |
| 6  --> 0.053079                                           | 6  --> 0.050980                                                |
| 7  --> 0.525852                                           | 7  --> 0.501961                                                |
| 8  --> 0.000221                                           | 8  --> 0.000000                                                |
| file name : ./test_multi/15734776934899998_109.jpg        | file name : ./test_multi/15734776934899998_109.jpg             |
| 0  --> 0.001334                                           | 0  --> 0.000000                                                |
| 1  --> 0.000987                                           | 1  --> 0.000000                                                |
| 2  --> 0.007705                                           | 2  --> 0.007843                                                |
| 3  --> 0.992148                                           | 3  --> 0.992157                                                |
| 4  --> 0.001849                                           | 4  --> 0.000000                                                |
| 5  --> 0.976295                                           | 5  --> 0.976471                                                |
| 6  --> 0.237823                                           | 6  --> 0.247059                                                |
| 7  --> 0.006454                                           | 7  --> 0.007843                                                |
| 8  --> 0.000469                                           | 8  --> 0.000000                                                |
| total  image number is: 5                                 | total image number is: 5                                       |
| infer  time  : 2860.0 ms, mean time is : 572.0            | infer time: 37.0  ms,  mean time : 7.4                         |
| proecess time : 373.0 ms, mean time : 74.6 ms             | proecess time: 428.0 ms, mean time : 85.6 ms                   |
| per image mean time : 646.6 ms                            | per image mean time : 93.0 ms                                  |
