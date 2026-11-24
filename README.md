# PDF草案比較アプリ（Streamlit配備用）

このコピーはStreamlit Community Cloudで試すための構成です。起動時に `documents/` 内の公開済みPDF 3本（COP31 2026-11-16、COP31 2026-11-19、CMA8 cover decision annotated）を読み込み、追加のPDFも画面からアップロードできます。固定サンプルを使わず別の文書セットを比較する場合は「全データをクリア」後にPDFをアップロードします。アプリ本体と同じディレクトリに保存する自動保存は無効にし、作業内容は利用者がプロジェクトZIPとしてダウンロードする方式にしています。

## GitHubへ置くファイル

このフォルダ全体をGitHubリポジトリのルートとして登録します。`streamlit_app.py`、`requirements.txt`、`documents/`、`.gitignore`、このREADMEを含めます。PDFをリポジトリに含めると、リポジトリの公開範囲に従って誰でも取得できます。ここに入っているPDFはいずれも公開資料として扱う前提です。

## Streamlit Community Cloudへ配備

1. GitHubに新しいリポジトリを作り、このフォルダ内のファイルを登録します。
2. Streamlit Community CloudでGitHub連携し、リポジトリ・ブランチを選びます。
3. Main file pathに `streamlit_app.py` を指定してDeployします。
4. 配備URLから動作を確認します。更新はGitHubへpushしたコードに応じて反映されます。

## PDFと作業データ

元プロジェクトの自動保存にはPDFが4本ありましたが、配備用コピーには現在表示対象になっていた3本を入れています。非表示だったもう1本のCMA8 PDFは含めていません。

- 同梱PDFは公開ソース文書を固定サンプルとして配備するため、GitHub上のリポジトリにも保存されます。差し替えたい場合は `documents/` を更新します。
- 利用者が追加アップロードしたPDFはサーバー側で処理されます。セッションをまたいだ永続保存にはなりません。
- Streamlitの各ブラウザー接続は別セッションです。利用者間でメモ、Watch、タグ訂正を共有・共同編集するデータベースはこの版には含めていません。
- 作業を引き継ぐ場合は「プロジェクトZIP」をダウンロードし、必要な人がアプリ上で復元します。
- PDFや注釈を個別に保存するための認証・ユーザー別クラウドストレージは未設定です。共同編集にする場合は別途追加が必要です。

## ローカル起動

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Windowsでは仮想環境を有効にするコマンドが異なります。
