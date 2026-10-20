# Benchmark de transcrição — mit-1806-lecture01

- Data: 2026-10-20 17:00
- Versão do projeto: 0.1.0 · Python 3.14.8 · faster-whisper 1.2.1
- CPU: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD · GPU: NVIDIA GeForce RTX 2060
- Normalização antes do WER: minúsculas, apóstrofos unificados, pontuação/hífens removidos, espaços colapsados
- *Tempo real* = tempo de transcrição ÷ duração do áudio (menor é mais rápido). O tempo inclui a decodificação do áudio com FFmpeg e não inclui a carga do modelo, medida à parte em *Carga*.

## mit_1806_lecture01

Áudio: `mit_1806_lecture01.webm` · duração 39min 48.8s · idioma en · com referência

| Configuração | Efetivo | Carga | Tempo | Tempo real | WER | S | D | I | Palavras |
|---|---|---|---|---|---|---|---|---|---|
| `small:cpu:int8` | cpu, int8 | 2.0s | 11min 42.1s | 0.294x | 9.2% | 227 | 98 | 130 | 4982 |
| `small:cuda:float16` | cuda, float16 | 1.0s | 1min 44.8s | 0.044x | 10.3% | 218 | 98 | 195 | 5047 |
| `large-v3-turbo:cuda:float16` | cuda, float16 | 58.2s | 1min 24.0s | 0.035x | 8.9% | 118 | 70 | 254 | 5134 |

## Substituições mais frequentes

### `small:cpu:int8` · mit_1806_lecture01

| Referência | Modelo escreveu | Vezes |
|---|---|---|
| one | 1 | 42 |
| two | 2 | 31 |
| zero | 0 | 22 |
| okay | ok | 21 |
| three | 3 | 17 |
| a | the | 5 |
| twelve | 12 | 5 |
| four | 4 | 4 |
| ys | y's | 4 |
| zs | z's | 4 |

### `small:cuda:float16` · mit_1806_lecture01

| Referência | Modelo escreveu | Vezes |
|---|---|---|
| one | 1 | 39 |
| two | 2 | 28 |
| zero | 0 | 22 |
| okay | ok | 19 |
| three | 3 | 15 |
| four | 4 | 4 |
| xs | x's | 4 |
| ys | y's | 4 |
| zs | z's | 4 |
| a | ax | 3 |

### `large-v3-turbo:cuda:float16` · mit_1806_lecture01

| Referência | Modelo escreveu | Vezes |
|---|---|---|
| okay | ok | 22 |
| 0 | zero | 9 |
| a | the | 5 |
| 2 | two | 4 |
| xs | x's | 4 |
| ys | y's | 4 |
| zs | z's | 4 |
| 1 | one | 3 |
| 2x | minus | 3 |
| a | ax | 3 |

