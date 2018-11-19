# environment construction

pipenvによる環境構築と、anacondaによる環境構築が選択できる。

基本的にはanacondaによるインストールの方が実行速度が速い。
ただし、個々のモジュールを適切にビルドした場合はそちらの方が速くなる可能性はある。

どちらの場合も、pyenvによるバージョン管理と組み合わせるのが賢い。

pyenv,pipenv,anacondaのインストールについては省略する。

注意1：
2018/11/16時点で、
Anaconda Cloud上に

- ChainerMN
- Chainer v5 (v5からChainerMNがマージされた)

が存在しないため、
anacondaを使う場合でもpipを使用してChainerMNをインストールする必要がある。
環境が壊れる可能性を承知した上で使用してください。
参考：http://onoz000.hatenablog.com/entry/2018/02/11/142347

注意2：
2018/11/16時点で、
pyenvを使ってanacondaをインストールしてある場合、pipenvによるインストールは失敗するという情報がある。
このバグの修正版は将来的にリリースされるらしい。
参考：https://github.com/pypa/pipenv/issues/3044

## pipenv

```
$ git clone https://github.com/ogura-edu/HDNNP.git
$ cd HDNNP/
$ pyenv install <VERSION>  # VERSION >= 3.6
$ pyenv local <VERSION>
$ pipenv install

# activate
$ pipenv shell

# deactivate
(HDNNP) $ exit
```

## anaconda

`conda env create --file condaenv.yaml`の実行が終了すると、
各々の環境に合わせてactivationの仕方がいくつか提示されるので好きなものを選ぶ。
以下の例では`~/.bashrc`に1文追記する方法を選択している。

```
$ git clone https://github.com/ogura-edu/HDNNP.git
$ cd HDNNP/
$ pyenv install <anacondaVERSION>
$ pyenv local <anacondaVERSION>
$ conda env create --file condaenv.yaml
$ echo ". ${HOME}/.pyenv/versions/<anacondaVERSION>/etc/profile.d/conda.sh" > ~/.bashrc

# activate
$ conda activate HDNNP

# deactivate
(HDNNP) $ conda deactivate
```

## environment variable

- `PIPENV_VENV_IN_PROJECT=1`
この環境変数を設定すると、pipenvで作成されるpythonの仮想環境がこのディレクトリの直下に作成される(`/path/to/HDNNP/.venv/`)

- `PATH=/path/to/HDNNP:${PATH}`
- `PYTHONPATH=/path/to/HDNNP:${PYTHONPATH}`
`hdnnpy`をこのディレクトリの直下*****以外**で実行する場合、この環境変数を設定してください。
