# environment construction

基本的にpythonのバージョン管理はpyenvを使うこと。  
必要なコマンド(pyenv,pipenv,conda)のインストールについては省略する。

注意1：  
2018/11/16時点で、Anaconda Cloud上に

- ChainerMN
- Chainer v5 (v5からChainerMNがマージされた)

が存在しないため、anacondaを使う場合でもpipを使用してChainerMNをインストールする必要がある。  
環境が壊れる可能性を承知した上で使用すること。  
参考：http://onoz000.hatenablog.com/entry/2018/02/11/142347

注意2：  
2018/11/16時点で、pyenvを使ってanacondaをインストールしてある場合、pipenvによるインストールは失敗するという情報がある。  
このバグの修正版は将来的にリリースされるらしい。  
参考：https://github.com/pypa/pipenv/issues/3044

## pipenv (recommended)

簡単かつ確実なインストール方法。

環境変数`PIPENV_VENV_IN_PROJECT`を1に設定すると、  
pipenvで作成されるpythonの仮想環境がこのプロジェクトの直下に作成される。(`/path/to/HDNNP/.venv/`)  
以下のコマンドを実行するか、`~/.bashrc`に追記してプロセスを再起動することで変更が適用される。
```
export PIPENV_VENV_IN_PROJECT=1
```

```
$ git clone https://github.com/ogura-edu/HDNNP.git
$ cd HDNNP/
$ pyenv install 3.6.7
$ pyenv local 3.6.7
$ pipenv install

# activate
$ pipenv shell

(HDNNP) $ hdnnpy training

# deactivate
(HDNNP) $ exit
```

## anaconda

最適化されたバイナリを取得できるので、実行速度が速い。  
しかし、上記の理由からpipと混在した形になることや、  
マシンによってはインストールがうまくいかないことがあるため注意すること。

`conda env create --file condaenv.yaml`の実行が終了すると、  
各々の環境に合わせてactivationの仕方がいくつか提示されるので好きなものを選ぶ。  
以下の例では`~/.bashrc`に1文追記する方法を選択している。

```
$ git clone https://github.com/ogura-edu/HDNNP.git
$ cd HDNNP/
$ pyenv install anaconda-x.x.x
$ pyenv local anaconda-x.x.x
$ conda env create -n HDNNP --file condaenv.yaml
$ echo ". ${HOME}/.pyenv/versions/anaconda-x.x.x/etc/profile.d/conda.sh" > ~/.bashrc

# activate
$ conda activate HDNNP

# install this program using pip
(HDNNP) $ pip install --editable .

(HDNNP) $ hdnnpy training

# deactivate
(HDNNP) $ conda deactivate
```

## pip install only

`Pipfile`または`condaenv.yaml`に記述されている依存関係を元に、  
パッケージを個別に`pip install`することももちろん可能。  
この場合は`virtualenv`を使って自分で仮想環境を管理することを推奨する。
```
$ git clone https://github.com/ogura-edu/HDNNP.git
$ cd HDNNP/
$ pip install PKG1 PKG2 ...
$ pip install -e .
```

または、慣れた人であれば依存関係を`setup.py`に書き加えるだけで済む。
```
$ git clone https://github.com/ogura-edu/HDNNP.git
$ cd HDNNP/
$ vim setup.py  #=> setup()の引数にinstall_requiresを追加
$ pip install -e .
```
