# 🎥 マルチカメラ品質管理システム

YOLOを使用した製品とパッケージの自動品質管理システムです。複数のUSBカメラで製品とパッケージを検出し、正しい組み合わせかをリアルタイムでチェックします。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)

## ✨ 主な機能

- 🎯 **複数カメラ対応** - 無制限にカメラを追加可能
- 🤖 **YOLO検出** - 高精度なリアルタイム物体検出
- 🔄 **連続検査** - ノンストップで品質管理
- ⚠️ **ライン停止警告** - 生産ライン停止を即座に検知
- 📊 **統計表示** - 合格率・不合格率をリアルタイム表示
- 🇯🇵 **日本語UI** - 完全日本語対応

## 🚀 クイックスタート

### 1. 必要なソフトウェア

- Python 3.8以上
- USBカメラ（2台以上推奨）
- 学習済みYOLOモデル（.ptまたは.onnx）

### 2. インストール

```bash
# リポジトリをクローン
git clone https://github.com/Rikiza89/Multi_camera_quality_control.git
cd Multi_camera_quality_control

# 必要なパッケージをインストール
pip install -r requirements.txt
```

**requirements.txt:**
```
Flask
Flask-SocketIO
opencv-python
numpy
ultralytics
python-socketio
eventlet
```

### 3. プロジェクト構造

```
quality-control-system/
├── app.py                 # メインアプリケーション
├── templates/
│   └── index.html        # Webインターフェース
├── qc_config.json        # 設定ファイル（自動生成）
├── requirements.txt      # 依存パッケージ
└── README.md            # このファイル
```

### 4. 実行

```bash
python app.py
```

ブラウザで http://localhost:5000 を開く

## 📖 使い方

### ステップ1: モデルの読み込み

1. YOLOモデルのパスを入力（例: `yolov8n.pt`）
2. 「モデル読込」ボタンをクリック
3. モデル名が表示されれば成功

### ステップ2: カメラの追加

1. カメラインデックスを選択（通常0から開始）
2. カメラの役割を選択：
   - **製品検出** - 製品を検出するカメラ
   - **パッケージ検出** - パッケージを検出するカメラ
3. 「カメラ追加」ボタンをクリック
4. カメラ映像が表示されれば成功

### ステップ3: 製品-パッケージペアの設定

1. 製品名を入力（例: `コカコーラ`）
2. 対応するパッケージ名を入力（例: `コカコーラ箱`）
3. 「追加」ボタンをクリック
4. 必要なペアを全て登録

### ステップ4: システム開始

1. 「システム開始」ボタンをクリック
2. リアルタイムで品質管理が開始されます
3. 不一致があれば即座にログに表示されます

## 🎯 動作の仕組み

```
製品カメラ → 製品検出 → 検出キュー
                              ↓
                        順次ペアチェック → 合格/不合格判定
                              ↓
パッケージカメラ → パッケージ検出 → 検出キュー
```

1. **製品カメラ**が製品を検出してキューに追加
2. **パッケージカメラ**がパッケージを検出してキューに追加
3. システムが自動的に順番にペアをチェック
4. 正しい組み合わせなら**合格**、違う場合は**不合格**

## ⚠️ ライン停止警告

設定した時間（デフォルト5秒）以上、カメラで検出がない場合：

- 🚨 赤い警告バナーが表示
- 停止中のカメラ情報を表示
- 生産ライン異常を即座に通知

## 📊 統計情報

リアルタイムで以下を表示：

- **検査総数** - チェックした製品の総数
- **合格** - 正しいペアの数と割合
- **不合格** - 間違ったペアの数と割合

## 🔧 設定

### ライン停止閾値の変更

ヘッダーの「ライン停止閾値」で秒数を変更できます（1-60秒）

### チェック間隔

デフォルトは1秒間隔でチェックします。`app.py`の`check_interval`で変更可能。

## 📝 YOLOモデルの準備

### オプション1: 事前学習済みモデルを使用

```bash
# YOLOv8 nanoモデル（最も軽量）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# YOLOv8 smallモデル（より高精度）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

### オプション2: カスタムモデルを学習

```python
from ultralytics import YOLO

# モデルの学習
model = YOLO('yolov8n.pt')
results = model.train(
    data='your_dataset.yaml',  # データセット設定
    epochs=100,
    imgsz=640
)
```

## 🛠️ トラブルシューティング

### カメラが検出されない

```bash
# 接続されているカメラを確認
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

### モデルが読み込めない

- ファイルパスが正しいか確認
- `.pt`または`.onnx`形式か確認
- ultralyticsがインストールされているか確認

### ポート5000が使用中

`app.py`の最後の行を変更：

```python
socketio.run(app, debug=True, host='0.0.0.0', port=8080)  # ポート変更
```

## 🌟 応用例

### 例1: 飲料工場

- カメラ1: ペットボトルの種類を検出
- カメラ2: 段ボール箱の種類を検出
- ペア: コカコーラ → コカコーラ箱

### 例2: 電子部品工場

- カメラ1: 基板の型番を検出
- カメラ2: 梱包袋のラベルを検出
- ペア: 基板A → 梱包袋A

### 例3: 食品工場

- カメラ1: 商品パッケージを検出
- カメラ2: 出荷用箱を検出
- ペア: お菓子A → 出荷箱A

## 📄 ライセンス

GNU Affero General Public License v3.0 (AGPL-3.0) License

## 🤝 貢献

プルリクエスト歓迎！バグ報告や機能提案はIssueでお願いします。

---

**開発者**: Rikiza89
**バージョン**: 1.0.0  
**最終更新**: 2026年1月
