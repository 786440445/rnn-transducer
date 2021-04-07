# RNN-Transducer

- 模型训练过程

1. 网络结构
    网络结构主要采取多层LSTM，encoder使用五层，decoder使用两层，每层之间加入一层非线性变换Relu。
      #参数设计
      encoder：
        hidden_size:1024
        project_size:300
        layer_norm
        
      decoder:
        hidden_size:512
        project_size:300
        layer_norm

      jointNet:
        hidden_size:2048
       
2. 预训练方法
    1. 声学模型采用CTC进行预训练。原始paper里采用层级CTC预训练。从多个维度对语音特征进行关系映射。这里直接对中文word进行预训练即可。得到语音特征到文字的映射关系。
    2. 语言模型采用LSTM进行预训练即可。从前文对下个节点进行结果预测。
    
3. 模型细节：
- 帧移25ms，帧长10ms
- 输入：40维度的Fbank特征，[B,L,40]，L最大设置为1600
- Low Frame Rate(10, 5), [B,L//5, 400], 对语音帧进行合并，合并10帧，然后跳过5帧，继续合并10帧。
- Encoder:
    5 * （LN + LSTM + projection_layer） = [B, 320， 400]
    输出：[B, 320, 320]
- Decoder：
    2 * （LN + LSTM + projection_layer） = [B, 50, 512]
    输出：[B, 50, 320]
- Joint：
    concat + forward(2048) + tanh + RNNT_loss解码

4. 模型优化点：
    1. load_randomly_augmented_audio，语音增益 与 语速增益
    2. encoder预训练采用原始数据，收敛后，再添加40%的specAugment(from google)扩增数据，直到收敛。
    3. 收敛后，融合decoder使用带specAugment扩增的数据进行训练。

3. 三个模块的参数组合设置
    1. 三个模块不同参数组合对是否能收敛影响很大。
    2. 编码器数量最少要是解码器参数的四倍以上。
    3. JointNetWork参数比解码器参数要大。
    4. 需要先将编码器encoder loss训练到最优地步，loss<0.1
    5. decoder训练到loss<2
    6. 联合encoder，decoder，进行训练