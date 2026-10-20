# Benchmark de transcrição — mit-1806-lecture01

- Data: 2026-10-20 17:33
- Versão do projeto: 0.1.0 · Python 3.14.8 · faster-whisper 1.2.1
- CPU: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD · GPU: NVIDIA GeForce RTX 2060
- WER bruto: minúsculas, apóstrofos unificados, pontuação/hífens removidos, espaços colapsados
- WER normalizado: o bruto mais, nos dois textos, por idioma (en): números por extenso → dígitos (0–999 e milhares simples), okay → ok, x's → xs, separador de milhar removido
- *Após a ref.* = palavras do final da transcrição gerada sem correspondência na referência (a referência pode terminar antes da fala); contadas à parte, fora das inserções e do WER.
- *Tempo real* = tempo de transcrição ÷ duração do áudio (menor é mais rápido). O tempo inclui a decodificação do áudio com FFmpeg e não inclui a carga do modelo, medida à parte em *Carga*.

## mit_1806_lecture01

Áudio: `mit_1806_lecture01.webm` · duração 39min 48.8s · idioma en · com referência

| Configuração | Efetivo | Carga | Tempo | Tempo real | WER bruto | S | D | I | WER norm. | S | D | I | Após a ref. | Palavras |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `small:cpu:int8` | cpu, int8 | 2.5s | 11min 40.3s | 0.293x | 8.9% | 227 | 98 | 116 | 5.6% | 61 | 98 | 116 | 14 | 4982 |
| `small:cuda:float16` | cuda, float16 | 1.1s | 1min 35.8s | 0.040x | 9.9% | 216 | 96 | 176 | 6.8% | 62 | 98 | 178 | 18 | 5048 |
| `large-v3-turbo:cuda:float16` | cuda, float16 | 2.9s | 1min 22.5s | 0.035x | 8.5% | 118 | 70 | 235 | 7.4% | 63 | 70 | 235 | 19 | 5134 |

## Substituições mais frequentes

### `small:cpu:int8` · mit_1806_lecture01 · WER bruto

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

### `small:cpu:int8` · mit_1806_lecture01 · WER normalizado

| Referência | Modelo escreveu | Vezes |
|---|---|---|
| a | the | 5 |
| a | ax | 3 |
| and | in | 3 |
| as | is | 2 |
| onto | to | 2 |
| x | equal | 2 |
| 06 | 1806 | 1 |
| 1and | and | 1 |
| 2y | plus | 1 |
| 3 | say | 1 |

### `small:cuda:float16` · mit_1806_lecture01 · WER bruto

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

### `small:cuda:float16` · mit_1806_lecture01 · WER normalizado

| Referência | Modelo escreveu | Vezes |
|---|---|---|
| a | ax | 3 |
| a | the | 3 |
| and | in | 3 |
| as | is | 2 |
| onto | to | 2 |
| x | equal | 2 |
| 06 | 1806 | 1 |
| 1and | and | 1 |
| 3 | some | 1 |
| a | equals | 1 |

### `large-v3-turbo:cuda:float16` · mit_1806_lecture01 · WER bruto

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

### `large-v3-turbo:cuda:float16` · mit_1806_lecture01 · WER normalizado

| Referência | Modelo escreveu | Vezes |
|---|---|---|
| a | the | 5 |
| 2x | minus | 3 |
| a | ax | 3 |
| and | in | 2 |
| as | is | 2 |
| equations | equation | 2 |
| onto | to | 2 |
| x | equal | 2 |
| 06 | 1806 | 1 |
| 1 | point | 1 |

## Palavras após o fim da referência

- `small:cpu:int8` · mit_1806_lecture01: 14 palavras, começando em: “find out because elimination fails find out when there isn't a solution…”
- `small:cuda:float16` · mit_1806_lecture01: 18 palavras, começando em: “find out because the because elimination if elimination fails find out when…”
- `large-v3-turbo:cuda:float16` · mit_1806_lecture01: 19 palavras, começando em: “find out because the system because eliminate if elimination fails find out…”

