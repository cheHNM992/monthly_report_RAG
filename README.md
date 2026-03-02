# monthly_report_RAG

ローカルHTML向けRAG検索システム（MVP）です。OpenAI Embeddings + GPT を使って、インデックス作成と自然言語検索を行います。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai
```

環境変数に API Key を設定してください。

```bash
export OPENAI_API_KEY="your_api_key"
```

## 使い方

### インデックス作成 / 差分更新

```bash
python monthly_report_RAG.py index --root ./docs --index ./rag_index
```

### 検索 + 回答生成

```bash
python monthly_report_RAG.py query --index ./rag_index "有給申請の方法は？"
```

### 主なオプション

- `--chunk-size`
- `--top-k`
- `--model`
- `--embedding-model`
- `--max-tokens`

## 保存データ

- `rag_index/documents.json`
- `rag_index/chunks.json`
