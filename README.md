# AI CUP 桌球戰術與結果預測系統

本專案針對桌球比賽中的時序擊球資料進行分析，建構基於深度學習的預測模型。系統利用整合後的時序特徵，預測下一拍的關鍵戰術流向（包括球種與落點位置），並評估該回合的最終勝負結果。

## 核心功能與預測任務

1. 下一拍球種預測 (actionId)：分析歷史擊球行為，預測下一拍採用的技術動作（如發球、切球、拉球等）。
2. 下一拍落點預測 (pointId)：結合空間調動與站位資訊，預測下一拍球落於球桌的九宮格區域。
3. 回合勝負預測 (serverGetPoint)：評估當前擊球序列的因果關係，預測發球者是否最終贏得該回合。

## 模型架構

本系統採用多任務學習 (Multi-Task Learning) 架構，核心設計包含：
* 共享嵌入層 (Shared Embeddings)：透過由 `nn.ModuleList` 管理的獨立 Embedding 層，將 11 項特徵分別轉換為 32 維的連續型高維向量，隨後將其拼接為單一的時序表徵。
* 主幹循環網路 (LSTM Backbone)：採用雙層（2 Layers）、具備 256 維隱藏層節點（Hidden Dimension）並加入 Dropout (0.3) 的雙層 LSTM 網路，用以提取時序脈絡資訊。
* 注意力機制層 (Attention Module)：針對回合最終勝負預測任務，引入專屬的 Attention 模組。該模組對 LSTM 的歷史輸出序列進行雙層線性映射與 Tanh 激活，配合 Masking 機制排除填充點的權重，計算出各拍的注意力分佈（Softmax），並加權融合出全域的上下文向量（Context Vector）。
* 獨立任務預測頭 (Task-Specific Heads)：
  * 球種預測頭（act_head）與落點預測頭（pt_head）直接作用於 LSTM 每一拍的序列輸出（多對多推論）。
  * 勝負預測頭（rly_head）則作用於融合後的注意力上下文向量，推論整個回合的發球得分機率（多對一推論）。

## 資料前處理與特徵工程

* 類別特徵編碼：從輸入資料中提取 11 項核心類別特徵（包括 sex, handId, strengthId, spinId, pointId, actionId, positionId, strikeId, scoreSelf, scoreOther, strikeNumber）。系統基於訓練集建立類別對照字典，將所有離散特徵統一進行類別編碼，並對拍數（strikeNumber）進行上下界截斷處理（限制於 0 至 40 拍之間）。
* 動態序列對齊 (Padding & Packing)：針對不同擊球長度的桌球回合（Rally）進行時序對齊。使用固定長度補齊（pad1d 與 pad2d）將序列填充至當前最大長度，並結合 PyTorch 的 pack_padded_sequence 與 pad_packed_sequence 機制，使 LSTM 在前向傳播中能忽略填充標記（PAD_TOKEN = 0），有效捕捉真實的時序動態。
* 類別不平衡處理：
  * 在戰術任務（球種與落點）上，計算訓練集的類別出現頻率，建構類別倒數根號之加權損失權重（Loss Weights），以抑制常見類別並提升稀有戰術的辨識度。
  * 在勝負預測任務上，計算發球得分與失分之比例（pos_weight），並於 DataLoader 中配置加權隨機抽樣器（WeightedRandomSampler），動態平衡訓練批次中的正負樣本分佈。

## 環境需求

* Python 3.x
* PyTorch
* Pandas
* NumPy
* Scikit-learn
