# PLAN_UPDATE_v2.md — Mudanças na proposta (apresentação v1 → v2)

Leia junto com `PLAN_BRAZIL.md`. **Este arquivo tem precedência** onde os dois divergirem. Antes de mudar código, leia `docs/repro_log.md` e o estado atual do fork: algumas fases podem já estar feitas.

---

## 1. O que mudou na proposta

| Tema | v1 (plano original) | v2 (atual) |
|---|---|---|
| Motivação | Energia eólica no NE | **Brisa marítima** e **robustez** do método: grade grossa, rede esparsa, mais falhas |
| Pergunta central | O método transfere para o Brasil? | O método funciona com grade grossa, rede esparsa e falhas? E recupera o ciclo da brisa? |
| Filtro de disponibilidade | Em aberto (default 0,8) | **Decidido: 0,6 → 26 estações** (0,8 dá só 15) |
| E3 (brisa) | Diagnóstico secundário | **Resultado principal** |
| E4 (densidade) | Retirar estações | Retirar estações **e** variar o filtro de disponibilidade |
| E5 (antecedência) | Opcional | **Removido** (fica só l = 1 h) |
| Experimento extra | — | **Novo: NE-EUA degradado** (§3) |
| Variáveis de energia (vento a 100 m etc.) | Mencionadas | **Fora do escopo** |
| Ressalva "downscaling" | Só do modelo Brasil | Vale também para os resultados ERA5/HRRR-A do artigo; documentar assim |

Números de referência (já calculados, não recalcule para o texto):
- NE-EUA: 442.800 km², 358 estações, 8,1 por 10 mil km², ~35 km de espaçamento.
- NE do Brasil: 390.800 km², 26 estações a 0,6, 0,67 por 10 mil km², ~123 km de espaçamento.

---

## 2. Mudanças no código e na configuração

### 2.1 Configuração Brasil
- Em `configs/ne_brazil.yaml`, `madis_control_ratio: 0.6`. Os demais campos seguem o `PLAN_BRAZIL.md` §5.
- `use_is_real_mask: true` é **obrigatório** (com 0,6, mais horas são preenchidas).
- Remover configs e scripts de E5 (`lead_time_sweep.py`), se já existirem. Não apague nada que já tenha resultado salvo; mova para `scripts/eval/_unused/`.

### 2.2 Persistência e referências
- `scripts/eval/baselines.py` precisa existir para Brasil **e** NE-EUA: ERA5 interpolado no instante-alvo e persistência ŵ(t+l) = w(t).
- Todas as métricas usam apenas horas reais (`IsReal_*`).

### 2.3 E3 — brisa (prioridade máxima entre as análises)
- `scripts/eval/diurnal_analysis.py`:
  - erro por **hora local** (UTC−3) para modelo, ERA5 interpolado e persistência;
  - separar estações costeiras e do interior;
  - figura principal: três curvas (modelo, ERA5, persistência) por grupo, erro do vetor vento.
- **Perguntar ao usuário o limiar de "costeira"** antes de fixar (sugestão: 20 km da linha de costa). Logar quantas estações caem em cada grupo; com 26 estações, esse número pode ser muito pequeno.
- Adicionar, se barato: o ciclo médio de u e v (observado, ERA5, modelo) por hora local nas costeiras. Mostra diretamente se a rotação da brisa é recuperada.
- Intervalos de confiança por bootstrap de **dias**.

### 2.4 E4 — cobertura e falhas
Duas varreduras independentes, treinando um modelo por ponto:
- **Densidade:** `station_keep_fraction` ∈ {1.0, 0.75, 0.5, 0.25} × 3 seeds, filtro 0,6.
- **Filtro:** `madis_control_ratio` ∈ {0.6, 0.7, 0.8}. Registrar o número de estações retidas em cada um.

O conjunto de avaliação precisa ser comparável entre pontos. Use um conjunto fixo de estações de teste presente em todos os casos. Se não for possível, **pergunte ao usuário** antes de escolher o desenho.

Com 26 estações, 25% são ~6 estações. Rode mesmo assim e reporte; não esconda o ponto.

### 2.5 Experimento extra — NE-EUA degradado (novo)
Objetivo: separar o efeito de (a) rede esparsa, (b) falhas e (c) grade grossa do efeito do clima tropical, usando os dados do artigo. Não depende dos dados do Brasil.

Fatores (cada um ligável por argumento, com default = sem degradação):

1. **`station_keep_n: 26`** — subamostrar a rede para 26 estações, espalhadas no espaço (amostragem espacialmente estratificada ou por farthest-point, não aleatória pura), com seed. O nome do shapefile filtrado deve incluir n e seed.
2. **Falhas sintéticas** — `gap_injection: {target_real_fraction, mean_gap_hours, seed}`.
   - Remover blocos contíguos de horas por estação (as falhas do INMET são sobretudo quedas de estação, não horas isoladas), até a fração real alvo.
   - Usar como referência as frações observadas no Brasil (pegar do relatório do notebook 02).
   - Implementar **depois** da leitura do arquivo processado: marcar `*_is_real = False`, re-preencher com o mesmo ffill/bfill de `Madis.createFile`, e não reescrever os arquivos originais.
3. **Grade grossa** — `era5_coarsen_deg` ∈ {0.25, 0.5, 1.0}.
   - Agregar o ERA5 com média em blocos (`xarray.coarsen(...).mean()`, fator 2 e 4) **antes** da interpolação para as estações.
   - Gravar em pasta própria (`era5_region: NEUS_c050`, `NEUS_c100`) para não colidir com os arquivos de cache existentes.

Desenho mínimo: base, cada fator isolado e os três juntos, ou seja, 5 rodadas com Transformer + ERA5, l = 1 h. Se sobrar tempo, adicionar a GNN.

Resultado esperado: tabela de erro em que a última linha (tudo degradado) pode ser comparada com o resultado do Brasil.

### 2.6 Testes
- `test_gap_injection.py`: fração real final dentro de ±1% do alvo; blocos contíguos; reprodutível por seed.
- `test_coarsen.py`: grade 0,5° tem metade das células por eixo; média preservada no domínio.
- `test_station_keep_n.py`: exatamente n estações; mesma seleção com a mesma seed.

---

## 3. Documentação a atualizar

- `docs/results_brazil.md`: E2, E3 e E4 com filtro 0,6.
- Novo `docs/results_degraded_neus.md` para o experimento extra.
- Em toda documentação: com l = 1 h e grade no instante-alvo, os resultados são **downscaling de reanálise**, como os resultados ERA5/HRRR-A do artigo. Só o HRRR-F do artigo é previsão real.
- `make_tables.py` gera as tabelas LaTeX de E2, E3 (resumo), E4 e do extra.

## 4. Ordem de trabalho

- [x] Ler `repro_log.md` e o estado do fork; listar o que já existe
- [x] Ajustar config Brasil para 0,6 e confirmar 26 estações (`configs/ne_brazil_{vit,gnn}.yaml`)
- [x] `baselines.py` (Brasil e NE-EUA)
- [x] E3 `diurnal_analysis.py` (limiar decidido: 20 km, Natural Earth 10 m)
- [x] Extra: `station_keep_n`, injeção de falhas, `era5_coarsen_deg` + testes → rodadas no notebook 03 (+ `sparse_gaps`, `coarse_050`)
- [x] E4 densidade e filtro (desenho decidido: 6 estações fixas mantidas em toda rede; filtro avaliado nas estações de 0,8) — código e notebook 04; rodadas pendentes
- [ ] Documentação e tabelas (`make_tables.py` e esqueletos em `docs/` prontos; faltam os números)
