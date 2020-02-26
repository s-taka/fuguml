FuguML
=======
Excelを入出力するAutoMLプログラムです。（現時点で開発中です。）

Windows環境でPyinstallerによりexe化可能な構成としています。
実行には下記ライブラリが必要です。
* pandas
* scikit-learn
* lightgbm
* optuna
* openpyxl
* xlrd

Pyinstallerの実行には上記に加えpypiwin32が必要です。

Example 
------
1. python3 app/excel_automl.py tests/test_2.xlsx 