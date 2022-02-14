# Example Code

あとで書く

<!-- TOC -->

- [Example Code](#example-code)
    - [概要](#%E6%A6%82%E8%A6%81)
        - [各ディレクトリの意味](#%E5%90%84%E3%83%87%E3%82%A3%E3%83%AC%E3%82%AF%E3%83%88%E3%83%AA%E3%81%AE%E6%84%8F%E5%91%B3)
        - [ルートディレクトリ下の各スクリプトの役割](#%E3%83%AB%E3%83%BC%E3%83%88%E3%83%87%E3%82%A3%E3%83%AC%E3%82%AF%E3%83%88%E3%83%AA%E4%B8%8B%E3%81%AE%E5%90%84%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%97%E3%83%88%E3%81%AE%E5%BD%B9%E5%89%B2)
    - [学習環境（navstack-gym）について](#%E5%AD%A6%E7%BF%92%E7%92%B0%E5%A2%83navstack-gym%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6)
        - [ざっくり](#%E3%81%96%E3%81%A3%E3%81%8F%E3%82%8A)
        - [リポジトリ](#%E3%83%AA%E3%83%9D%E3%82%B8%E3%83%88%E3%83%AA)
        - [基本的なつかいかた](#%E5%9F%BA%E6%9C%AC%E7%9A%84%E3%81%AA%E3%81%A4%E3%81%8B%E3%81%84%E3%81%8B%E3%81%9F)
        - [gym.makeで設定できる項目](#gymmake%E3%81%A7%E8%A8%AD%E5%AE%9A%E3%81%A7%E3%81%8D%E3%82%8B%E9%A0%85%E7%9B%AE)
        - [env.resetで設定できる項目](#envreset%E3%81%A7%E8%A8%AD%E5%AE%9A%E3%81%A7%E3%81%8D%E3%82%8B%E9%A0%85%E7%9B%AE)
        - [タスク環境のID一覧](#%E3%82%BF%E3%82%B9%E3%82%AF%E7%92%B0%E5%A2%83%E3%81%AEid%E4%B8%80%E8%A6%A7)

<!-- /TOC -->

## 概要

### 各ディレクトリの意味
- ルートディレクトリ直下
  - 学習用スクリプトとか評価用スクリプトとかの置き場
  - 頻繁に触る
- algo
  - 上記のスクリプトで使うDrQアルゴリズムとかその他細かいモジュール置き場
- conf
  - 実験条件の設定ファイル置き場
  - 環境生成条件やハイパラの設定（hydra形式）がまとまって存在する
- log
  - 学習ログ置き場
- movie
  - 評価時の動画（gif）置き場
- work
  - 学習開始時に作られるワークスペース
  - 各モデルやリプレイバッファや学習結果の保存先

### ルートディレクトリ下の各スクリプトの役割
- `train.py`: 
  - 学習開始用スクリプト
- `visualize.ipynb`:
  - 学習過程のプロット用
- `eval.ipynb`:
  - 学習済みエージェントの評価用（動画出力）
- `test_env.ipynb`, `test_obs.ipynb`:
  - 環境の動作確認したくなったときに触るやつ
- その他:
  - 名前のはじめに _ がついているものは実装途中とか微妙なやつ


## 学習環境（navstack-gym）について

### ざっくり
- グリッドワールド状の移動ロボット制御タスク環境
  - 空間はグレイスケールの値域で画像形式で表される（現時点では通行可能・障害物のいずれか）
  - ロボットは「通行可能」色のピクセル上でのみ移動可能
- NavStackライクなシステムを内包
  - 疑似LiDARからOccupancyMapを生成
  - move_baseっぽく目標姿勢指定による自律移動制御

### リポジトリ

- 本体：
  - [navstack\-gym](https://github.com/wwwshwww/navstack-gym)（Gym環境本体）
- 要素：
  - [nav\-sim\-modules](https://github.com/wwwshwww/nav-sim-modules)（シミュレーション部分）
  - [randoor](https://github.com/wwwshwww/randoor)（部屋生成エンジン）


### 基本的なつかいかた

```python
import gym
import navstack_gym

env = gym.make('VisibleTreasureChestRoom-v0')
env.reset()
```

以下のタイミングでの引数の指定によってタスク条件を変更可
- Envインスタンス生成時：
  - 該当メソッド：`gym.make()`
  - 設定対象：シミュレーションの動作設定（地図サイズ等・経路探索上限回数など）
- 環境リセット時：
  - 該当メソッド：`env.reset()`
  - 設定対象：部屋生成条件・再生成の可否

### `gym.make()`で設定できる項目
|引数名|型|デフォルト値|意味|感覚的な解釈|
|-|-|-:|----|-|
|id|str|-|タスク環境のID（後述）|-|
|map_size|int|256|地図のサイズ|実質的な世界の大きさ|
|map_resolution|float|0.1|地図1マスあたりの距離（m）|-|
|spawn_extension|float|0.3|エージェントの生成領域を障害物からどれだけ離すか（m）|どれだけ安全な内地からサンプルされやすくするか|
|path_exploration|int|4,000|経路探索時の隣接ノード探索回数上限|1回の経路探索をどれだけ長く行うか|
|path_planning_count|int|10|経路探索回数上限|経路探索のやり直しを何回まで許容するか|
|path_turnable|float|$\frac{\pi}{8}$|1ピクセル移動ごとの回転可能角度（rad）|ロボットの旋回性能|
|allowable_goal_error_norm|float|0.5|ゴール到達判定の閾値（m）|-|
|avoidance_size|int|3|障害物マスから周囲何ピクセルを避けるか|ロボットの体の大きさ|
|move_limit|int|-1|1回の移動可能距離上限（ピクセル単位・-1で上限なし）|自律移動の制限時間的な役割|
|movable_discount|float|5|目標位置の指定可能距離上限（地図サイズの何分割分まで指定できるようにするか）|大きくするほど指定可能範囲が狭まる|
|found_threshold|float|0.75|オブジェクトの発見判定閾値（m）|-|
|passable_color|int|0|環境の通行可能領域の値設定|-|
|map_obs_val|int|100|地図に記録する障害物属性|-|
|map_pass_val|int|0|地図に記録する通行可能属性|-|
|map_unk_val|int|-1|地図に記録する未知属性|-|

### `env.reset()`で設定できる項目
|引数名|型|デフォルト値|意味|感覚的な解釈|
|-|-|-:|-|-|
|is_generate_pose|bool|True|エージェント初期位置の再サンプルの可否|-|
|is_generate_room|bool|True|部屋構造の再生成の可否|-|
|scene_obstacle_count|int|10|部屋内に生成する障害物の個数|-|
|scene_obstacle_size|float|0.7|障害物のサイズ（m）|-|
|scene_target_size|float|0.2|宝箱のサイズ（m）|-|
|scene_key_size|float|0.2|鍵のサイズ（m）|-|
|scene_obstacle_zone_thresh|float|1.5|障害物クラスタリングの閾値（m）|どこまで障害物群とみなすか|
|scene_distance_key_placing|float|0.7|鍵配置領域を障害物・壁からどれだけ離すか（m）|-|
|scene_range_key_placing|float|0.3|鍵配置領域の幅（m）|-|
|scene_room_length_max|float|9|部屋ポリゴンの外接円の最大直径（m）|部屋のだいたいの大きさ|
|scene_room_wall_thickness|float|0.05|壁の厚さ（m）|-| 
|scene_wall_threshold|float|0.1|オブジェクト生成を壁からどれだけ離すか（m）|高くするほど全オブジェクトの配置が内地に寄る|

### タスク環境のID一覧

以下参照。  
[navstack_gym/__init__.py](https://github.com/wwwshwww/navstack-gym/blob/master/navstack_gym/__init__.py)


- `Invisible`：
  - 宝箱・鍵に衝突判定が付与されない
  - 地図に記録されない
- `Visible`：
  - 全てオブジェクトに衝突判定が付与される
  - 地図に記録され、獲得されると衝突判定が消失（地図からも消える）