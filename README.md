# Legal-text-decoder

```bash
docker run --rm --gpus all `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\output:/app/output" `
  -p 8051:8501 `
  legal-text-app:1.0
```